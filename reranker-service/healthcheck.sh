#!/bin/bash
# Reranker health check — kill if unresponsive, launchd will auto-restart
RESPONSE=$(curl -s --max-time 5 http://localhost:8102/health 2>/dev/null)
if [ $? -ne 0 ] || ! echo "$RESPONSE" | grep -q '"status":"ok"'; then
    echo "[$(date)] Reranker unresponsive, killing..."
    pkill -f "uvicorn main:app.*--port 8102" 2>/dev/null
fi
