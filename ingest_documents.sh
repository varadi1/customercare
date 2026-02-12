#!/bin/bash
# Hanna OETP - Document Ingestion Script
# Ingests all OETP pályázati dokumentumok with proper metadata

API="http://localhost:8101"
DOCS="/Users/varadiimre/.openclaw/hanna/documents"

echo "=== Hanna OETP Document Ingestion ==="
echo ""

# 1. Pályázati felhívás (módosított, egységes szerkezetben)
echo "📄 1/7 Pályázati felhívás (1. módosítás, egységes szerkezetben)..."
curl -s -X POST "$API/ingest/pdf" \
  -F "file=@$DOCS/Felhivas_OETP_260129_1szmodositas.pdf" \
  -F "category=felhívás" \
  -F "chunk_type=document" \
  -F "valid_from=2026-01-29" \
  -F "version=2" | python3 -m json.tool
echo ""

# 2. Gyakran Ismételt Kérdések (GyIK)
echo "📄 2/7 GYIK (Gyakran Ismételt Kérdések)..."
curl -s -X POST "$API/ingest/pdf" \
  -F "file=@$DOCS/OEPT_GYIK_20260204.pdf" \
  -F "category=gyik" \
  -F "chunk_type=faq" \
  -F "valid_from=2026-02-04" \
  -F "version=1" | python3 -m json.tool
echo ""

# 3. Kitöltési segédlet
echo "📄 3/7 Kitöltési segédlet..."
curl -s -X POST "$API/ingest/pdf" \
  -F "file=@$DOCS/OETP_Palyazat_kitoltesi_segedlet.pdf" \
  -F "category=útmutató" \
  -F "chunk_type=document" \
  -F "valid_from=2026-02-02" \
  -F "version=1" | python3 -m json.tool
echo ""

# 4. Kitöltőfelület mezők (v3)
echo "📄 4/7 Kitöltőfelület mezők (v3)..."
curl -s -X POST "$API/ingest/pdf" \
  -F "file=@$DOCS/OETP_kitoltofelulet_mezok_rovid_v3.pdf" \
  -F "category=útmutató" \
  -F "chunk_type=document" \
  -F "valid_from=2026-01-27" \
  -F "version=3" | python3 -m json.tool
echo ""

# 5. Gazdasági példák
echo "📄 5/7 Gazdasági példák..."
curl -s -X POST "$API/ingest/pdf" \
  -F "file=@$DOCS/Gazdasagi_peldak.pdf" \
  -F "category=útmutató" \
  -F "chunk_type=document" \
  -F "valid_from=2026-01-26" \
  -F "version=1" | python3 -m json.tool
echo ""

echo "=== PDF ingestion complete ==="
echo ""
echo "Remaining to ingest via /ingest/text:"
echo "  6. Közlemények (nffku.hu announcements)"
echo "  7. EU Bizottsági közlemény (CELEX:52016XC0719(05))"
