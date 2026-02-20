# Ingest QA Checklist — Kritikus dokumentumok

## Minden dokumentum ingest ELŐTT:
1. **Supersedes check:** Van-e korábbi verzió a RAG-ban? → `supersedes` paraméter beállítása
2. **chunk_type megadás:** MINDIG explicit megadni (palyazat_felhivas, segedlet, gyik, kozlemeny, palyazat_melleklet)
3. **category megadás:** MINDIG explicit megadni
4. **valid_from:** A dokumentum érvényesség kezdete

## Minden dokumentum ingest UTÁN:
1. **Chunk count ellenőrzés:** Hány chunk jött létre? Reális szám?
2. **Sample search:** Keress rá a dokumentum kulcsszavaira — megjelenik-e a top 5-ben?
3. **Old version check:** Az invalidált (superseded) dokumentum tényleg expired?
4. **BM25 rebuild:** `POST /bm25/rebuild`
5. **Authority check:** A chunk_type helyes authority weight-et kap?

## Verziókezelés szabály:
- Felhívás módosítás → `supersedes="régi_felhivas.pdf"`, chunk_type=palyazat_felhivas
- Segédlet frissítés → `supersedes="régi_segedlet.pdf"`, chunk_type=segedlet
- GYIK frissítés → `supersedes="régi_gyik.pdf"`, chunk_type=gyik
- Közlemény frissítés → `supersedes="régi_kozlemeny"`, chunk_type=document, category=közlemények
