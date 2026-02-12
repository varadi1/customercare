#!/bin/bash
# Start the Contextual AI Reranker Service (native, MPS)

cd "$(dirname "$0")"

# Activate the reranker venv
source ~/.venv/reranker/bin/activate

# Start uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8102
