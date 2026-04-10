#!/bin/bash
# Obsidian vault daily sync for CustomerCare RAG
# Called by launchd (com.openclaw.obsidian-daily-sync)
# Ensures Docker + container are running before triggering ingest

LOG="/tmp/obsidian-daily-sync.log"
CONTAINER="cc-backend"
API_URL="http://localhost:8101"
MAX_RETRIES=3

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG"
}

log "=== Obsidian sync starting ==="

# 1. Check if Docker daemon is running
if ! /usr/local/bin/docker info >/dev/null 2>&1; then
    log "ERROR: Docker is not running. Attempting to start Docker Desktop..."
    open -a Docker
    # Wait up to 60 seconds for Docker to start
    for i in $(seq 1 12); do
        sleep 5
        if /usr/local/bin/docker info >/dev/null 2>&1; then
            log "Docker started after ${i}x5 seconds"
            break
        fi
    done
    if ! /usr/local/bin/docker info >/dev/null 2>&1; then
        log "ERROR: Docker failed to start after 60s. Aborting."
        exit 1
    fi
fi

# 2. Check if cc-backend container is running
STATUS=$(/usr/local/bin/docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)
if [ "$STATUS" != "running" ]; then
    log "WARNING: Container $CONTAINER is '$STATUS'. Starting it..."
    cd /Users/varadiimre/DEV/customercare && /usr/local/bin/docker compose up -d backend
    sleep 15  # wait for startup
    STATUS=$(/usr/local/bin/docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)
    if [ "$STATUS" != "running" ]; then
        log "ERROR: Container failed to start (status: $STATUS). Aborting."
        exit 1
    fi
    log "Container started successfully."
fi

# 3. Wait for API to be responsive
for i in $(seq 1 $MAX_RETRIES); do
    HEALTH=$(/usr/bin/curl -s --max-time 10 "${API_URL}/health" 2>&1)
    if echo "$HEALTH" | grep -q '"status":"ok"'; then
        break
    fi
    log "API not ready (attempt $i/$MAX_RETRIES), waiting 10s..."
    sleep 10
done

if ! echo "$HEALTH" | grep -q '"status":"ok"'; then
    log "ERROR: API not healthy after $MAX_RETRIES attempts. Aborting."
    exit 1
fi

# 4. Trigger ingest
log "Triggering Obsidian ingest..."
RESULT=$(/usr/bin/curl -s --max-time 30 -X POST "${API_URL}/obsidian/ingest?vault_path=/app/obsidian-vault" 2>&1)
log "Ingest response: $RESULT"

# 5. Wait and check result (ingest is async, poll for completion)
sleep 30
for i in $(seq 1 60); do
    STATUS_JSON=$(/usr/bin/curl -s --max-time 15 "${API_URL}/obsidian/ingest/status" 2>&1)
    if echo "$STATUS_JSON" | grep -q '"running":false'; then
        log "Ingest completed: $STATUS_JSON"
        break
    fi
    if [ $i -eq 60 ]; then
        log "WARNING: Ingest still running after 30 minutes. Check manually."
        break
    fi
    sleep 30
done

log "=== Obsidian sync finished ==="
