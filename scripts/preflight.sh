#!/bin/bash
# =============================================================================
# Hanna — Pre-flight Health Check
# =============================================================================
# Validates all services and dependencies before starting.
# Run before docker compose up or after install.
#
# Usage: bash scripts/preflight.sh
# Exit code: 0 = all OK, 1 = issues found
# =============================================================================

set -u
HANNA_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

ok()   { echo -e "  ${GREEN}✅${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠️${NC}  $1"; WARNINGS=$((WARNINGS+1)); }
fail() { echo -e "  ${RED}❌${NC} $1"; ERRORS=$((ERRORS+1)); }

echo "=== Hanna Pre-flight Check ==="
echo ""

# --- 1. .env file ---
echo "[1/7] Environment"
if [ -f "$HANNA_DIR/.env" ]; then
    ok ".env exists"
    # Check critical keys
    for key in OPENAI_API_KEY ANTHROPIC_API_KEY GRAPH_TENANT_ID GRAPH_CLIENT_ID GRAPH_CLIENT_SECRET GRAPH_USER_EMAIL; do
        val=$(grep "^${key}=" "$HANNA_DIR/.env" 2>/dev/null | cut -d'=' -f2-)
        if [ -z "$val" ]; then
            fail "$key not set in .env"
        fi
    done
    # Optional keys
    for key in GOOGLE_API_KEY DISCORD_BOT_TOKEN COHERE_API_KEY; do
        val=$(grep "^${key}=" "$HANNA_DIR/.env" 2>/dev/null | cut -d'=' -f2-)
        if [ -z "$val" ]; then
            warn "$key not set (optional)"
        fi
    done
else
    fail ".env not found — run: cp .env.example .env"
fi

# --- 2. Docker ---
echo ""
echo "[2/7] Docker"
if command -v docker &>/dev/null; then
    ok "Docker installed ($(docker --version | head -1))"
    if docker info &>/dev/null; then
        ok "Docker daemon running"
    else
        fail "Docker daemon not running"
    fi
else
    fail "Docker not installed"
fi

# --- 3. Docker services ---
echo ""
echo "[3/7] Docker Services"
for svc in hanna-db hanna-backend hanna-langfuse; do
    state=$(docker inspect -f '{{.State.Status}}' "$svc" 2>/dev/null || echo "not found")
    if [ "$state" = "running" ]; then
        ok "$svc running"
    elif [ "$state" = "not found" ]; then
        warn "$svc not created (run: docker compose up -d)"
    else
        fail "$svc status: $state"
    fi
done

# --- 4. Native GPU services ---
echo ""
echo "[4/7] Native GPU Services"

check_service() {
    local name=$1 port=$2
    if curl -s --max-time 3 "http://localhost:$port/health" >/dev/null 2>&1; then
        local info=$(curl -s --max-time 3 "http://localhost:$port/health" 2>/dev/null)
        ok "$name :$port — responding"
    else
        fail "$name :$port — not responding"
    fi
}

check_service "BGE-M3 Search"   8104
check_service "BGE-M3 Ingest"   8114
check_service "BGE Reranker"    8102

# --- 5. Database connectivity ---
echo ""
echo "[5/7] Database"

# PostgreSQL via Docker
if docker exec hanna-db pg_isready -U klara -d hanna_oetp &>/dev/null 2>&1; then
    ok "hanna_oetp DB ready"
    chunk_count=$(docker exec hanna-db psql -U klara -d hanna_oetp -t -c "SELECT count(*) FROM chunks" 2>/dev/null | tr -d ' ')
    if [ -n "$chunk_count" ] && [ "$chunk_count" -gt 0 ] 2>/dev/null; then
        ok "chunks table: $chunk_count entries"
    else
        warn "chunks table empty (run scraper/ingest)"
    fi
else
    warn "hanna_oetp DB not reachable (container down?)"
fi

# --- 6. Service directories ---
echo ""
echo "[6/7] Native Service Files"

for dir_name in "BGE-M3 Search:local_llm/bge_m3" "BGE-M3 Ingest:local_llm/bge_m3_ingest" "Reranker:local_llm/reranker"; do
    name="${dir_name%%:*}"
    rel="${dir_name##*:}"
    parent="$(dirname "$HANNA_DIR")"
    dir="$parent/$rel"
    if [ -d "$dir" ]; then
        ok "$name dir exists ($dir)"
        if [ -f "$dir/app.py" ] || [ -f "$dir/main.py" ]; then
            ok "$name server script found"
        else
            fail "$name server script not found in $dir"
        fi
    else
        fail "$name dir missing ($dir)"
    fi
done

# --- 7. LaunchAgents ---
echo ""
echo "[7/7] LaunchAgents"

for la in com.openclaw.bge-m3 com.openclaw.bge-m3-ingest com.openclaw.hanna-reranker com.openclaw.hanna-healthcheck com.openclaw.nffku-monitor; do
    if launchctl list "$la" &>/dev/null; then
        ok "$la loaded"
    else
        warn "$la not loaded"
    fi
done

# --- Summary ---
echo ""
echo "==========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Passed with $WARNINGS warning(s)${NC}"
else
    echo -e "${RED}$ERRORS error(s), $WARNINGS warning(s)${NC}"
fi
echo "==========================================="

exit $ERRORS
