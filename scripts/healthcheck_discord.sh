#!/bin/bash
# Hanna ecosystem health check — Discord alerts on failure + auto-restart
# Checks: backend (:8101), DB (pg_isready), BGE search/ingest, reranker
# Run via LaunchAgent every 5 minutes

HANNA_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Discord bot config
DISCORD_BOT_TOKEN="${DISCORD_BOT_TOKEN:-MTQ2ODk3NTM2MjExMjIyNTQyNA.G8ZQHb.TYV6SSieuUDgX6i06jm9pzkAf2RgxDtWAOzsW0}"
DISCORD_CHANNEL_ID="${DISCORD_CHANNEL_ID:-1468974303159517396}"
STATE_FILE="/tmp/hanna_healthcheck_state"

# Services: name|check_type|url_or_cmd|restart_cmd
# check_type: http = curl, docker = docker healthcheck, process = kill pattern
SERVICES=(
    "Hanna-DB|docker|hanna-db|docker restart hanna-db"
    "Hanna-backend|http|http://localhost:8101/livez|docker restart hanna-backend"
    "BGE-M3-search|http|http://localhost:8104/health|kill:bge_m3/app.py"
    "BGE-M3-ingest|http|http://localhost:8114/health|kill:bge_m3_ingest/app.py"
    "Reranker|http|http://localhost:8102/health|kill:uvicorn main:app.*--port 8102"
)

send_discord() {
    curl -s -X POST \
        "https://discord.com/api/v10/channels/${DISCORD_CHANNEL_ID}/messages" \
        -H "Authorization: Bot ${DISCORD_BOT_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" \
        >/dev/null 2>&1
}

touch "$STATE_FILE"

check_prev() {
    grep -q "^$1=down$" "$STATE_FILE" 2>/dev/null && echo "down" || echo "ok"
}

check_service() {
    local check_type="$1"
    local target="$2"

    case "$check_type" in
        http)
            curl -sf --max-time 5 "$target" >/dev/null 2>&1
            ;;
        docker)
            # Check: container running + healthy
            local status
            status=$(docker inspect --format='{{.State.Health.Status}}' "$target" 2>/dev/null)
            [ "$status" = "healthy" ]
            ;;
    esac
}

restart_service() {
    local restart_cmd="$1"

    if [[ "$restart_cmd" == kill:* ]]; then
        local pattern="${restart_cmd#kill:}"
        pkill -f "$pattern" 2>/dev/null || true
    elif [[ "$restart_cmd" == docker* ]]; then
        eval "$restart_cmd" 2>/dev/null || true
    fi
}

# ── DB-specific check: verify hanna_oetp database exists ──
check_db_data() {
    local count
    count=$(docker exec hanna-db psql -U klara -d hanna_oetp -t -c "SELECT COUNT(*) FROM chunks;" 2>/dev/null | tr -d ' ')
    if [ -z "$count" ] || [ "$count" = "0" ]; then
        return 1
    fi
    return 0
}

NEW_STATE=""
ALL_OK=true

for entry in "${SERVICES[@]}"; do
    IFS='|' read -r name check_type target restart_cmd <<< "$entry"
    prev=$(check_prev "$name")

    if check_service "$check_type" "$target"; then
        # Service is up — extra DB data check
        if [ "$name" = "Hanna-DB" ]; then
            db_prev=$(check_prev "Hanna-DB-data")
            if check_db_data; then
                NEW_STATE="${NEW_STATE}Hanna-DB-data=ok\n"
                if [ "$db_prev" = "down" ]; then
                    send_discord "✅ **Hanna-DB adatok** — helyreállt (chunks tábla elérhető)"
                fi
            else
                NEW_STATE="${NEW_STATE}Hanna-DB-data=down\n"
                if [ "$db_prev" != "down" ]; then
                    send_discord "🚨 **Hanna-DB** — fut, de a chunks tábla ÜRES vagy nem elérhető! Init script probléma?"
                fi
            fi
        fi

        NEW_STATE="${NEW_STATE}${name}=ok\n"
        if [ "$prev" = "down" ]; then
            send_discord "✅ **${name}** — helyreállt"
            echo "[$(date)] ${name} recovered"
        fi
    else
        NEW_STATE="${NEW_STATE}${name}=down\n"
        ALL_OK=false

        # Auto-restart
        restart_service "$restart_cmd"

        if [ "$prev" != "down" ]; then
            send_discord "🚨 **${name}** — NEM ELÉRHETŐ → restart indítva"
            echo "[$(date)] ${name} DOWN — restart triggered"
        else
            echo "[$(date)] ${name} still down (no repeat alert)"
        fi
    fi
done

# ── Backend DB connectivity check ──
# Backend may be "up" but can't reach DB (pool error)
if curl -sf --max-time 5 "http://localhost:8101/livez" >/dev/null 2>&1; then
    health_json=$(curl -sf --max-time 10 "http://localhost:8101/health" 2>/dev/null)
    chunk_count=$(echo "$health_json" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('collection_count',0))" 2>/dev/null || echo "0")
    db_conn_prev=$(check_prev "Backend-DB-conn")

    if [ "$chunk_count" -gt 0 ] 2>/dev/null; then
        NEW_STATE="${NEW_STATE}Backend-DB-conn=ok\n"
        if [ "$db_conn_prev" = "down" ]; then
            send_discord "✅ **Backend→DB kapcsolat** — helyreállt (${chunk_count} chunk)"
        fi
    else
        NEW_STATE="${NEW_STATE}Backend-DB-conn=down\n"
        if [ "$db_conn_prev" != "down" ]; then
            send_discord "🚨 **Backend→DB kapcsolat** — Backend fut, de 0 chunk! DB elérhetetlen? → backend restart"
            docker restart hanna-backend 2>/dev/null
        fi
    fi
fi

printf "$NEW_STATE" > "$STATE_FILE"

if $ALL_OK; then
    echo "[$(date)] All services OK"
fi
