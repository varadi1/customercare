#!/bin/bash
# Daily email history ingest for Hanna RAG
# Fetches sent items from the last 48h and ingests as Q&A chunks

MAILBOX="lakossagitarolo@neuzrt.hu"
SINCE=$(date -u -v-48H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '-48 hours' '+%Y-%m-%dT%H:%M:%SZ')
URL="http://localhost:8101/emails/history/ingest?mailbox=${MAILBOX}&dry_run=false&since=${SINCE}&max_items=500"

echo "[$(date)] Starting daily email ingest..."
RESULT=$(curl -s -X POST "$URL" 2>&1)
echo "$RESULT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'Fetched: {d.get(\"sent_items_fetched\",0)} | Ingested: {d.get(\"ingested\",0)} | Chunks: {d.get(\"chunks_created\",0)} | Errors: {d.get(\"errors\",0)}')
except:
    print('Error parsing result')
"
echo "[$(date)] Done."
