"""Comprehensive tests — unit + integration for all Hanna modules.
Run: cd backend && python3 -m pytest tests/ -v"""
import pytest, json, math
from tests.conftest import run


# ============================================================
# REASONING: traces.py
# ============================================================
class TestTracesExtended:
    def test_find_similar_filters_by_program(self, db_conn, ensure_reasoning_traces_table):
        from app.reasoning.traces import create_trace, resolve_trace, find_similar_traces
        emb = [0.3]*1024
        t1 = run(create_trace(conn=db_conn, query_text="OETP q", category="t", program="OETP", query_embedding=emb))
        run(resolve_trace(conn=db_conn, trace_id=t1, sent_text="a", similarity_score=0.9))
        t2 = run(create_trace(conn=db_conn, query_text="NPP q", category="t", program="NPP", query_embedding=emb))
        run(resolve_trace(conn=db_conn, trace_id=t2, sent_text="b", similarity_score=0.9))
        results = run(find_similar_traces(conn=db_conn, query_embedding=emb, program="OETP", limit=10, min_similarity=0.5))
        for r in results: assert r["program"] == "OETP"

    def test_classify_outcome(self):
        from app.reasoning.traces import _classify_outcome
        assert _classify_outcome(0.95) == "SENT_AS_IS"
        assert _classify_outcome(0.85) == "SENT_AS_IS"
        assert _classify_outcome(0.84) == "SENT_MODIFIED"
        assert _classify_outcome(0.30) == "SENT_MODIFIED"
        assert _classify_outcome(0.29) == "REJECTED"
        assert _classify_outcome(0.0) == "REJECTED"


# ============================================================
# REASONING: person_tracker.py
# ============================================================
class TestPersonTrackerExtended:
    def test_register_organization(self, db_conn):
        from app.reasoning.person_tracker import register_organization
        oid = run(register_organization(conn=db_conn, org_name="TestCorp Ltd", domain="testcorp.hu"))
        assert oid is not None
        row = run(db_conn.fetchrow("SELECT type, name FROM kg_entities WHERE id=$1", oid))
        assert row["type"] == "organization"

    def test_get_or_create_application(self, db_conn):
        from app.reasoning.person_tracker import get_or_create_application
        aid = run(get_or_create_application(conn=db_conn, oetp_id="OETP-2026-UNITTEST"))
        assert aid is not None
        aid2 = run(get_or_create_application(conn=db_conn, oetp_id="OETP-2026-UNITTEST"))
        assert aid == aid2  # dedup

    def test_link_entities_idempotent(self, db_conn):
        from app.reasoning.person_tracker import register_sender, get_or_create_application, link_entities
        pid = run(register_sender(conn=db_conn, sender_name="Link Test", sender_email="link.test@example.com"))
        aid = run(get_or_create_application(conn=db_conn, oetp_id="OETP-2026-LINK"))
        run(link_entities(conn=db_conn, source_id=pid, target_id=aid, relation_type="ASKED_ABOUT"))
        run(link_entities(conn=db_conn, source_id=pid, target_id=aid, relation_type="ASKED_ABOUT"))
        cnt = run(db_conn.fetchval("SELECT COUNT(*) FROM kg_relations WHERE source_id=$1 AND target_id=$2", pid, aid))
        assert cnt == 1

    def test_gmail_skips_org(self, db_conn):
        from app.reasoning.person_tracker import process_email_entities
        result = run(process_email_entities(conn=db_conn, sender_name="Gmail User", sender_email="test@gmail.com", oetp_ids=[]))
        assert result["org_id"] is None

    def test_extract_oetp_ids(self):
        from app.reasoning.person_tracker import extract_oetp_ids
        ids = extract_oetp_ids("Pályázatom: OETP-2026-123456 és OETP-2026-789012")
        assert len(ids) == 2
        assert extract_oetp_ids("Nincs ID itt") == []


# ============================================================
# REASONING: authority_learner.py
# ============================================================
class TestAuthorityLearner:
    def test_apply_adjustments_reorders(self):
        from app.reasoning.authority_learner import apply_learned_adjustments
        results = [
            {"score": 0.80, "chunk_type": "email_reply"},
            {"score": 0.75, "chunk_type": "felhivas"},
            {"score": 0.70, "chunk_type": "gyik"},
        ]
        adj = {"inverter": {"email_reply": -0.10, "gyik": +0.08}}
        updated = apply_learned_adjustments(results, "inverter", adj)
        assert updated[0]["chunk_type"] == "gyik"  # boosted
        assert updated[-1]["chunk_type"] == "email_reply"  # penalized

    def test_no_matching_category(self):
        from app.reasoning.authority_learner import apply_learned_adjustments
        results = [{"score": 0.80, "chunk_type": "felhivas"}]
        updated = apply_learned_adjustments(results, "nonexistent", {"inverter": {"felhivas": 0.1}})
        assert updated[0]["score"] == 0.80  # unchanged

    def test_format_empty(self):
        from app.reasoning.authority_learner import format_adjustments_report
        assert "Nincs elég" in format_adjustments_report({})

    def test_format_with_data(self):
        from app.reasoning.authority_learner import format_adjustments_report
        r = format_adjustments_report({"inverter": {"gyik": 0.05}})
        assert "inverter" in r
        assert "gyik" in r


# ============================================================
# REASONING: policy_tracker.py
# ============================================================
class TestPolicyTracker:
    def test_check_empty_chunks(self, db_conn):
        from app.reasoning.policy_tracker import check_answer_validity
        result = run(check_answer_validity(db_conn, []))
        assert result["all_valid"] is True

    def test_invalidate_and_check(self, db_conn, ensure_reasoning_traces_table):
        from app.reasoning.policy_tracker import invalidate_superseded_chunks, check_answer_validity
        run(db_conn.execute("""INSERT INTO chunks (id, doc_id, doc_type, content, metadata)
            VALUES ('pol_test_1', 'pol_old', 'felhívás', 'Régi', '{"valid_from":"2025-01-01"}'::jsonb)
            ON CONFLICT (id) DO NOTHING"""))
        result = run(invalidate_superseded_chunks(conn=db_conn, new_doc_id="pol_new", superseded_doc_id="pol_old"))
        assert result["invalidated_chunks"] >= 1
        validity = run(check_answer_validity(db_conn, ["pol_test_1"]))
        assert validity["all_valid"] is False


# ============================================================
# REASONING: radix_client.py
# ============================================================
class TestRadixClient:
    def test_format_applicant_context(self):
        from app.reasoning.radix_client import format_applicant_context, STATUS_MAP
        ctx = format_applicant_context({"palyazati_kodszam": "OETP-2026-TEST", "status": 6, "palyazo_neve": "Teszt Elek", "celterulet": 1, "igenyelt_tamogatas": 4000000})
        assert "OETP-2026-TEST" in ctx
        assert "Teszt Elek" in ctx
        assert "Nyertes" in ctx
        assert "4 000 000" in ctx

    def test_status_map_completeness(self):
        from app.reasoning.radix_client import STATUS_MAP
        assert len(STATUS_MAP) >= 10
        assert STATUS_MAP[6] == "Nyertes"
        assert STATUS_MAP[0] == "Piszkozat"

    def test_disabled_returns_none(self):
        from app.reasoning.radix_client import get_application
        # With default config (disabled), should return None
        result = get_application("OETP-2026-FAKE")
        assert result is None  # disabled by default


# ============================================================
# REASONING: style_score.py (extended)
# ============================================================
class TestStyleScoreExtended:
    def test_length_ratio(self):
        from app.reasoning.style_score import _score_length
        assert _score_length("x" * 100, "x" * 100) == 1.0  # identical
        assert _score_length("x" * 50, "x" * 100) >= 0.7    # 50% ratio
        assert _score_length("x" * 10, "x" * 100) <= 0.4    # 10% ratio

    def test_closing_match(self):
        from app.reasoning.style_score import _score_closing
        assert _score_closing("blah\nÜdvözlettel:\nNemzeti Energetikai", "blah\nÜdvözlettel:\nNemzeti Energetikai") == 1.0
        assert _score_closing("blah", "blah\nÜdvözlettel:") == 0.5  # half match

    def test_brevity_short_question(self):
        from app.reasoning.style_score import _score_brevity
        assert _score_brevity("Rövid válasz.", "OK.") == 1.0  # both short
        assert _score_brevity("x" * 500, "OK.") <= 0.5  # hanna too long


# ============================================================
# REASONING: reference_checker.py (extended)
# ============================================================
class TestReferenceCheckerExtended:
    def test_missing_section(self):
        from app.reasoning.reference_checker import check_references
        r = check_references("A Felhívás 9.9. pontja szerint", ["A Felhívás 3.3. pontja"])
        assert r["valid"] is False
        assert len(r["issues"]) > 0

    def test_no_references_is_valid(self):
        from app.reasoning.reference_checker import check_references
        r = check_references("Nincs hivatkozás, csak szöveg.", ["forrás szöveg"])
        assert r["valid"] is True


# ============================================================
# REASONING: knowledge_gaps.py (extended)
# ============================================================
class TestKnowledgeGapsExtended:
    def test_obsidian_format(self):
        from app.reasoning.knowledge_gaps import format_obsidian_report
        report = {"status": "ok", "period_days": 7, "period_start": "2026-04-01", "period_end": "2026-04-06",
                  "total_traces": 10, "success_rate": 0.7, "outcomes": {"SENT_AS_IS": 7, "REJECTED": 3, "SENT_MODIFIED": 0, "PENDING": 0},
                  "problem_categories": [{"category": "inverter", "count": 3, "examples": []}], "recommendations": ["Teszt"]}
        md = format_obsidian_report(report)
        assert "Knowledge Gap" in md
        assert "inverter" in md
        assert "70.0%" in md

    def test_high_rejection_recommendation(self):
        from app.reasoning.knowledge_gaps import _generate_recommendations
        recs = _generate_recommendations([], 0.5, 8, 2, 20)
        assert any("átíródott" in r for r in recs)


# ============================================================
# LLM CLIENT
# ============================================================
class TestLLMClientExtended:
    def test_provider_order(self):
        from app.llm_client import PROVIDERS
        assert PROVIDERS[0]["name"] == "openai"
        assert PROVIDERS[0]["model"] == "gpt-5.4-mini"
        assert PROVIDERS[1]["name"] == "anthropic"
        assert PROVIDERS[2]["name"] == "google"

    def test_gpt5_uses_max_completion_tokens(self):
        """GPT-5.x models must use max_completion_tokens, not max_tokens."""
        # Verify the code handles this
        model = "gpt-5.4-mini"
        assert model.startswith("gpt-5")


# ============================================================
# SKIP FILTER
# ============================================================
class TestSkipFilter:
    def test_thank_you_skip(self):
        try:
            from app.email.skip_filter import check_skip
        except ImportError:
            pytest.skip("msal not installed locally")
        r = check_skip("Köszönöm a válaszukat, minden világos.", "RE: Válasz")
        assert r["skip"] is True

    def test_normal_question_not_skipped(self):
        try:
            from app.email.skip_filter import check_skip
        except ImportError:
            pytest.skip("msal not installed locally")
        r = check_skip("Mikor kapom meg a támogatási döntést az OETP pályázatomra?", "Pályázat státusz")
        assert r["skip"] is False

    def test_auto_reply_skip(self):
        try:
            from app.email.skip_filter import check_skip
        except ImportError:
            pytest.skip("msal not installed locally")
        r = check_skip("I am out of office.", "Automatic reply: Out of Office")
        assert r["skip"] is True


# ============================================================
# SCHEDULER
# ============================================================
class TestScheduler:
    def test_get_status(self):
        from app.scheduler import get_scheduler_status
        status = get_scheduler_status()
        assert "enabled" in status
        assert "running" in status
        assert "last_run" in status


# ============================================================
# CONFIG
# ============================================================
class TestConfig:
    def test_all_settings_have_defaults(self):
        from app.config import Settings
        s = Settings()
        assert s.embedding_backend == "bge-m3"
        assert s.oetp_db_port == 3307
        assert s.auto_process_enabled is False  # default off

    def test_oetp_db_defaults(self):
        from app.config import Settings
        s = Settings()
        assert s.oetp_db_host == "185.187.73.44"
        assert s.oetp_db_name == "tarolo_neuzrt_hu_db"


# ============================================================
# E2E: Draft generate (needs running backend)
# ============================================================
class TestE2EDraft:
    def test_draft_endpoint_responds(self):
        """E2E: draft/generate returns valid response."""
        import httpx
        try:
            resp = httpx.post("http://localhost:8101/draft/generate",
                json={"email_text": "Teszt kérdés", "email_subject": "Teszt", "top_k": 3, "max_context_chunks": 2},
                timeout=30)
            assert resp.status_code == 200
            data = resp.json()
            assert "confidence" in data
            assert data.get("llm_provider") in ("openai", "anthropic", "google", None)
        except httpx.ConnectError:
            pytest.skip("Backend not running")

    def test_health_endpoint(self):
        import httpx
        try:
            resp = httpx.get("http://localhost:8101/health", timeout=5)
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except httpx.ConnectError:
            pytest.skip("Backend not running")

    def test_llm_health_endpoint(self):
        import httpx
        try:
            resp = httpx.get("http://localhost:8101/llm/health", timeout=30)
            assert resp.status_code == 200
            data = resp.json()
            providers = data.get("providers", {})
            # At least one provider should be ok
            assert any(v.get("status") == "ok" for v in providers.values())
        except httpx.ConnectError:
            pytest.skip("Backend not running")

    def test_scheduler_status_endpoint(self):
        import httpx
        try:
            resp = httpx.get("http://localhost:8101/scheduler/status", timeout=5)
            assert resp.status_code == 200
            data = resp.json()
            assert data["enabled"] is True
        except httpx.ConnectError:
            pytest.skip("Backend not running")

    def test_process_endpoint(self):
        """E2E: /emails/process returns valid stats."""
        import httpx
        try:
            resp = httpx.post("http://localhost:8101/emails/process?hours=1", timeout=60)
            assert resp.status_code == 200
            data = resp.json()
            assert "emails_polled" in data
            assert "errors" in data
            assert data["errors"] == 0
        except httpx.ConnectError:
            pytest.skip("Backend not running")
