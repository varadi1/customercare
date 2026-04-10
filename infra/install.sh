#!/bin/bash
# =============================================================================
# CustomerCare — Full Install Script
# =============================================================================
# Sets up everything needed to run CustomerCare on a new machine:
#   1. .env from template
#   2. Native GPU services (BGE-M3 embedding + reranker)
#   3. LaunchAgents (auto-start services)
#   4. Docker services (backend + DB + Langfuse)
#   5. Pre-flight health check
#
# Usage: cd ~/DEV/customercare && bash infra/install.sh
# =============================================================================

set -e
CC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_LLM="$(dirname "$CC_DIR")/local_llm"
PLIST_DIR="$HOME/Library/LaunchAgents"

echo "============================================"
echo "  CustomerCare Install"
echo "  Directory: $CC_DIR"
echo "============================================"
echo ""

# ─── 1. .env ─────────────────────────────────────────────────────────────────
echo "[1/6] Environment"
if [ ! -f "$CC_DIR/.env" ]; then
    cp "$CC_DIR/.env.example" "$CC_DIR/.env"
    echo "  ⚠️  Created .env from template — fill in API keys before starting!"
else
    echo "  ✅ .env exists"
fi

# ─── 2. Native GPU services ──────────────────────────────────────────────────
echo ""
echo "[2/6] Native GPU Services"

setup_bge_m3() {
    local dir="$LOCAL_LLM/bge_m3"
    local ingest_dir="$LOCAL_LLM/bge_m3_ingest"

    if [ -d "$dir" ] && [ -f "$dir/app.py" ]; then
        echo "  ✅ BGE-M3 search service exists ($dir)"
    else
        echo "  📦 Setting up BGE-M3 search service..."
        mkdir -p "$dir"
        cp "$CC_DIR/infra/gpu_services/bge_m3_app.py" "$dir/app.py" 2>/dev/null || true
    fi

    if [ -d "$ingest_dir" ] && [ -f "$ingest_dir/app.py" ]; then
        echo "  ✅ BGE-M3 ingest service exists ($ingest_dir)"
    else
        echo "  📦 Setting up BGE-M3 ingest service..."
        mkdir -p "$ingest_dir"
        cp "$CC_DIR/infra/gpu_services/bge_m3_ingest_app.py" "$ingest_dir/app.py" 2>/dev/null || true
    fi

    # Create shared venv if missing
    local venv="$dir/.venv"
    if [ ! -d "$venv" ] && [ ! -L "$venv" ]; then
        echo "  📦 Creating BGE-M3 venv..."
        python3 -m venv "$venv"
        "$venv/bin/pip" install -q -r "$CC_DIR/infra/gpu_services/requirements-bge-m3.txt"
        echo "  ✅ BGE-M3 venv ready"
        # Symlink for ingest
        ln -sf "$venv" "$ingest_dir/.venv" 2>/dev/null || true
    else
        echo "  ✅ BGE-M3 venv exists"
    fi
}

setup_reranker() {
    local dir="$LOCAL_LLM/reranker"

    if [ -d "$dir" ] && [ -f "$dir/main.py" ]; then
        echo "  ✅ Reranker service exists ($dir)"
    else
        echo "  📦 Setting up Reranker service..."
        mkdir -p "$dir"
        cp "$CC_DIR/infra/gpu_services/reranker_main.py" "$dir/main.py" 2>/dev/null || true
    fi

    local venv="$dir/.venv"
    if [ ! -d "$venv" ] && [ ! -L "$venv" ]; then
        echo "  📦 Creating Reranker venv..."
        python3 -m venv "$venv"
        "$venv/bin/pip" install -q -r "$CC_DIR/infra/gpu_services/requirements-reranker.txt"
        echo "  ✅ Reranker venv ready"
    else
        echo "  ✅ Reranker venv exists"
    fi
}

mkdir -p "$LOCAL_LLM"
setup_bge_m3
setup_reranker

# ─── 3. LaunchAgents ─────────────────────────────────────────────────────────
echo ""
echo "[3/6] LaunchAgents"
mkdir -p "$PLIST_DIR"
for plist in "$CC_DIR"/infra/*.plist; do
    name=$(basename "$plist")
    # Substitute paths for current install location
    sed \
        -e "s|/Users/varadiimre/DEV/customercare|$CC_DIR|g" \
        -e "s|/Users/varadiimre/DEV/local_llm|$LOCAL_LLM|g" \
        -e "s|/Users/varadiimre/.openclaw/jogszabaly-rag/bge_m3_service/.venv|$LOCAL_LLM/bge_m3/.venv|g" \
        -e "s|/Users/varadiimre/.venv/reranker|$LOCAL_LLM/reranker/.venv|g" \
        "$plist" > "$PLIST_DIR/$name"
    launchctl unload "$PLIST_DIR/$name" 2>/dev/null || true
    launchctl load "$PLIST_DIR/$name" 2>/dev/null || true
    echo "  ✅ $name"
done

# ─── 4. Docker ───────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Docker Services"
cd "$CC_DIR"

echo "  Building backend image..."
docker compose build backend 2>&1 | tail -2

echo "  Starting services..."
docker compose up -d 2>&1 | tail -3

echo "  Waiting for health checks..."
sleep 10

# ─── 5. Model download warmup ────────────────────────────────────────────────
echo ""
echo "[5/6] Model Warmup"
echo "  Waiting for GPU services to load models (first run may take 2-5 min)..."

for attempt in $(seq 1 30); do
    bge_ok=0; rnk_ok=0
    curl -s --max-time 3 http://localhost:8104/health >/dev/null 2>&1 && bge_ok=1
    curl -s --max-time 3 http://localhost:8102/health >/dev/null 2>&1 && rnk_ok=1

    if [ $bge_ok -eq 1 ] && [ $rnk_ok -eq 1 ]; then
        echo "  ✅ All GPU services ready"
        break
    fi
    if [ $attempt -eq 30 ]; then
        echo "  ⚠️  GPU services still loading — check /tmp/bge_m3_service.log"
    fi
    sleep 10
done

# ─── 6. Pre-flight ───────────────────────────────────────────────────────────
echo ""
echo "[6/6] Pre-flight Check"
bash "$CC_DIR/scripts/preflight.sh"

echo ""
echo "============================================"
echo "  Install complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Fill in .env with API keys (if not done)"
echo "  2. Ingest documents: docker exec cc-backend python3 /app/scripts/scrape_nffku_oetp.py"
echo "  3. Test: cd backend && python3 -m pytest tests/ -v"
echo "  4. Enable processing: set AUTO_PROCESS_ENABLED=true in .env"
