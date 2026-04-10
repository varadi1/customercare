# CustomerCare — Generikus RAG-alapú Ügyfélszolgálati Asszisztens

Többrétegű RAG (Retrieval-Augmented Generation) framework ügyfélszolgálati email-kezeléshez. Beérkező emailek feldolgozása, tudásbázis-keresés, válasz-draft generálás, és folyamatos tanulás emberi korrekcióból.

**Forked from [Hanna](https://github.com/varadi1/hanna)** — az OETP (Otthoni Energiatároló Program) ügyfélszolgálati asszisztens.

---

## Miben különbözik Hannától?

| | Hanna | CustomerCare |
|---|---|---|
| **Cél** | OETP ügyfélszolgálat (production) | Bármilyen program/szervezet |
| **Konfiguráció** | Hardcoded OETP | `config/program.yaml` |
| **Scraper** | nffku.hu | Tetszőleges URL |
| **Guardrails** | OETP-specifikus 7 szabály | Konfigurálható |
| **Nyelv** | Magyar | Bármilyen |

## Gyors Start

```bash
# 1. Klónozás
git clone https://github.com/varadi1/customercare.git
cd customercare

# 2. Program konfiguráció
cp config/program.example.yaml config/program.yaml
# Szerkeszd a program.yaml-t a saját programodhoz

# 3. Környezeti változók
cp .env.example .env
# Töltsd ki az API kulcsokat

# 4. Telepítés
bash infra/install.sh
```

## Konfiguráció

A teljes program-specifikus beállítás a `config/program.yaml` fájlban:

- **program** — Név, szervezet, nyelv
- **scraper** — Honnan töltse le a tudásbázist
- **doc_types** — Dokumentum típusok és authority score-ok
- **email** — Skip szabályok, köszöntés, aláírás
- **llm** — Provider prioritás, system prompt
- **guardrails** — Biztonsági szabályok
- **verification** — Ellenőrzési rétegek (NLI, CoVe, SelfCheck)
- **kg** — Knowledge Graph extraction beállítások
- **learning** — Tanulási rendszer (5 szint)

## Főbb Funkciók

- **15 verifikációs réteg** — hallucináció-mentes válaszok
- **5 szintű tanulási rendszer** — emberi korrekcióból tanul
- **Knowledge Graph** — automatikus entitás-kinyerés és keresés
- **Multi-provider LLM** — Opus 4.6 → GPT-5.4 → Gemini (auto-fallback)
- **Natív GPU** — BGE-M3 embedding + reranker (MPS/CUDA)
- **Outlook 365 integráció** — MS Graph API polling + draft mentés
- **Langfuse observability** — teljes trace minden kérdésre

## Architektúra

Részletes architektúra leírás: [CLAUDE.md](CLAUDE.md) és [DEPLOYMENT.md](DEPLOYMENT.md)

## Státusz

🚧 **Work in progress** — A generalizálás folyamatban. Jelenleg a kódbázis még tartalmaz OETP-specifikus részeket, amelyek fokozatosan kerülnek konfigurálhatóvá.

### Refaktor terv

1. ✅ `config/program.yaml` — program konfiguráció séma
2. ✅ `config.py` — yaml betöltés + Settings integráció (`get_program_config()`, `get_db_config()`)
3. ✅ `radix_client.py` — generikus program DB client (mysql/mssql, schema YAML-ből)
4. ✅ `observability.py` — 15+ Langfuse span típus, token tracking
5. ✅ Teljes `hanna` → `cc` rename (78 fájl)
6. ✅ `ingest.py` — authority map doc_types-ból, KG threshold config-driven
7. ✅ `main.py` — system prompt (Langfuse → YAML → hardcoded), signature YAML-ből
8. ✅ `guardrails.py` — toggle-ök + custom_rules YAML-ből, app_id_pattern config-ból
9. ✅ `skip_filter.py` — skip domains/patterns YAML-ből, processor.py is YAML-t használ
10. ✅ `scrape.py` — generikus config-driven scraper (escalation: httpx→stealth→camoufox, snapshot+change detection, multi-page, PDF download+ingest)
11. ✅ `drafts.py` — signature + fallback YAML-ből

## Licenc

MIT
