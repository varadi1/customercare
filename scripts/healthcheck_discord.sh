#!/bin/bash
# Hanna ecosystem health check — Discord alerts on failure
# Checks: backend (:8101), BGE search (:8104), BGE ingest (:8114), reranker (:8102)
# Run via LaunchAgent every 5 minutes

# Discord bot config
DISCORD_BOT_TOKEN="${DISCORD_BOT_TOKEN:-MTQ2ODk3NTM2MjExMjIyNTQyNA.G8ZQHb.TYV6SSieuUDgX6i06jm9pzkAf2RgxDtWAOzsW0}"
DISCORD_CHANNEL_ID="${DISCORD_CHANNEL_ID:-1468974303159517396}"
STATE_FILE="/tmp/hanna_healthcheck_state"

# Services: name|url|kill_pattern (kill_pattern optional — for native processes that can freeze)
SERVICES=(
    "Hanna-backend|http://localhost:8101/livez|"
    "BGE-M3-search|http://localhost:8104/health|bge_m3/app.py"
    "BGE-M3-ingest|http://localhost:8114/health|bge_m3_ingest/app.py"
    "Reranker|http://localhost:8102/health|uvicorn main:app.*--port 8102"
)

send_discord() {
    curl -s -X POST \
        "https://discord.com/api/v10/channels/${DISCORD_CHANNEL_ID}/messages" \
        -H "Authorization: Bot ${DISCORD_BOT_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" \
        >/dev/null 2>&1
}

# Load previous state (key=status per line)
touch "$STATE_FILE"

check_prev() {
    grep -q "^$1=down$" "$STATE_FILE" 2>/dev/null && echo "down" || echo "ok"
}

# Check each service
NEW_STATE=""
ALL_OK=true

for entry in "${SERVICES[@]}"; do
    IFS='|' read -r name url kill_pattern <<< "$entry"
    prev=$(check_prev "$name")

    if curl -sf --max-time 5 "$url" >/dev/null 2>&1; then
        NEW_STATE="${NEW_STATE}${name}=ok\n"
        if [ "$prev" = "down" ]; then
            send_discord "✅ **${name}** — helyreállt"
            echo "[$(date)] ${name} recovered"
        fi
    else
        NEW_STATE="${NEW_STATE}${name}=down\n"
        ALL_OK=false
        # Kill frozen process if pattern defined (launchd KeepAlive will restart)
        if [ -n "$kill_pattern" ]; then
            pkill -f "$kill_pattern" 2>/dev/null && echo "[$(date)] ${name} killed (frozen)" || true
        fi
        if [ "$prev" != "down" ]; then
            send_discord "🚨 **${name}** — NEM ELÉRHETŐ → kill + restart (${url})"
            echo "[$(date)] ${name} DOWN — killed & alert sent"
        else
            echo "[$(date)] ${name} still down (no repeat alert)"
        fi
    fi
done

# Save current state
printf "$NEW_STATE" > "$STATE_FILE"

if $ALL_OK; then
    echo "[$(date)] All services OK"
fi
