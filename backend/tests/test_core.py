"""Core tests for reasoning memory + person tracking + style score."""
import pytest
from tests.conftest import run

class TestTraces:
    def test_create_and_resolve(self, db_conn, ensure_reasoning_traces_table):
        from app.reasoning.traces import create_trace, resolve_trace
        tid = run(create_trace(conn=db_conn, query_text="Teszt kérdés", category="altalanos"))
        assert tid is not None
        run(resolve_trace(conn=db_conn, trace_id=tid, sent_text="Válasz", similarity_score=0.9))
        row = run(db_conn.fetchrow("SELECT outcome FROM reasoning_traces WHERE id=$1", tid))
        assert row["outcome"] == "SENT_AS_IS"

    def test_outcome_thresholds(self, db_conn, ensure_reasoning_traces_table):
        from app.reasoning.traces import create_trace, resolve_trace
        for sim, expected in [(0.95, "SENT_AS_IS"), (0.55, "SENT_MODIFIED"), (0.1, "REJECTED")]:
            tid = run(create_trace(conn=db_conn, query_text=f"test {sim}", category="test"))
            run(resolve_trace(conn=db_conn, trace_id=tid, sent_text="x", similarity_score=sim))
            row = run(db_conn.fetchrow("SELECT outcome FROM reasoning_traces WHERE id=$1", tid))
            assert row["outcome"] == expected, f"sim={sim}: expected {expected}, got {row['outcome']}"

    def test_find_similar_excludes_pending(self, db_conn, ensure_reasoning_traces_table):
        from app.reasoning.traces import create_trace, find_similar_traces
        emb = [0.1]*1024
        run(create_trace(conn=db_conn, query_text="pending", category="t", query_embedding=emb))
        results = run(find_similar_traces(conn=db_conn, query_embedding=emb, limit=3, min_similarity=0.5))
        for r in results: assert r["outcome"] != "PENDING"

class TestPersonTracker:
    def test_register_and_dedup(self, db_conn):
        from app.reasoning.person_tracker import register_sender
        id1 = run(register_sender(conn=db_conn, sender_name="Test A", sender_email="test.dedup@example.com"))
        id2 = run(register_sender(conn=db_conn, sender_name="Test B", sender_email="test.dedup@example.com"))
        assert id1 == id2

    def test_process_email_entities(self, db_conn):
        from app.reasoning.person_tracker import process_email_entities
        result = run(process_email_entities(conn=db_conn, sender_name="Eval User", sender_email="eval@solarpro.hu", oetp_ids=["OETP-2026-TEST01"], email_subject="Teszt"))
        assert result["person_id"] is not None
        assert result["org_id"] is not None
        assert len(result["application_ids"]) == 1

class TestStyleScore:
    def test_matching_style(self):
        from app.reasoning.style_score import compute_style_score
        r = compute_style_score("Tisztelt Pályázó!\nVálasz.\nÜdvözlettel:\nNEÜ Zrt.", "Tisztelt Pályázó!\nVálasz.\nÜdvözlettel:\nNEÜ Zrt.")
        assert r["overall"] > 0.8

    def test_wrong_greeting_penalty(self):
        from app.reasoning.style_score import compute_style_score
        r = compute_style_score("Tisztelt Meghatalmazott!\nX", "Tisztelt Pályázó!\nX")
        assert r["components"]["greeting"] < 0.5

class TestReferenceChecker:
    def test_valid_reference(self):
        from app.reasoning.reference_checker import check_references
        r = check_references("A Felhívás 3.3. pontja szerint", ["A Felhívás 3.3. pontja alapján"])
        assert r["valid"] == True

    def test_wrong_program(self):
        from app.reasoning.reference_checker import check_references, should_downgrade_confidence
        r = check_references("Az Otthonfelújítási Program keretében", ["OETP"])
        assert r["wrong_program"] == True
        assert should_downgrade_confidence(r) == True

class TestKnowledgeGaps:
    def test_recommendations(self):
        from app.reasoning.knowledge_gaps import _generate_recommendations
        recs = _generate_recommendations([("inverter", 5)], 0.3, 10, 8, 20)
        assert any("KRITIKUS" in r for r in recs)

class TestLLMClient:
    def test_provider_config(self):
        from app.llm_client import PROVIDERS
        assert len(PROVIDERS) == 3
        assert PROVIDERS[0]["name"] == "openai"
        assert PROVIDERS[1]["name"] == "anthropic"
        assert PROVIDERS[2]["name"] == "google"
