#!/bin/bash
# Hanna — Install script
# Installs LaunchAgents, builds Docker image, creates .env template
#
# Usage: cd ~/DEV/hanna && bash infra/install.sh

set -e
HANNA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"

echo "=== Hanna Install ==="
echo "Directory: $HANNA_DIR"
echo ""

# 1. .env check
if [ ! -f "$HANNA_DIR/.env" ]; then
    echo "[1/5] Creating .env template..."
    cat > "$HANNA_DIR/.env" << 'ENVEOF'
# LLM Providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
COHERE_API_KEY=

# Email (MS Graph API)
GRAPH_TENANT_ID=
GRAPH_CLIENT_ID=
GRAPH_CLIENT_SECRET=
GRAPH_USER_EMAIL=
SHARED_MAILBOXES=lakossagitarolo@neuzrt.hu

# OETP MySQL (readonly)
OETP_DB_PASSWORD=
OETP_DB_ENABLED=false

# Discord bot
DISCORD_BOT_TOKEN=
DISCORD_CHANNEL_ID=

# Autonomous processing
AUTO_PROCESS_ENABLED=false
ENVEOF
    echo "  ⚠️  Fill in .env before starting!"
else
    echo "[1/5] .env exists ✅"
fi

# 2. Install LaunchAgents
echo "[2/5] Installing LaunchAgents..."
for plist in "$HANNA_DIR"/infra/*.plist; do
    name=$(basename "$plist")
    # Update paths in plist to current HANNA_DIR
    sed "s|/Users/varadiimre/DEV/hanna|$HANNA_DIR|g" "$plist" > "$PLIST_DIR/$name"
    launchctl unload "$PLIST_DIR/$name" 2>/dev/null || true
    launchctl load "$PLIST_DIR/$name" 2>/dev/null
    echo "  ✅ $name"
done

# 3. Build Docker image
echo "[3/5] Building Docker image..."
cd "$HANNA_DIR"
docker compose build backend 2>&1 | tail -2

# 4. Python dependencies (for local testing)
echo "[4/5] Checking Python dependencies..."
pip3 install pytest pytest-asyncio asyncpg pymysql 2>/dev/null | tail -1

# 5. Verify
echo "[5/5] Verifying..."
docker compose up -d backend 2>&1 | tail -2
sleep 8

# Health checks
for port in 8101 8102 8104 8114; do
    if curl -s --max-time 3 "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "  ✅ :$port"
    else
        echo "  ❌ :$port (not responding)"
    fi
done

echo ""
echo "=== Install complete ==="
echo "Next steps:"
echo "  1. Fill in .env with API keys"
echo "  2. Run: docker compose up -d"
echo "  3. Run: python3 backend/scripts/migrate_reasoning.py"
echo "  4. Test: python3 -m pytest backend/tests/ -v"
