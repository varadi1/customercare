"""Langfuse observability for Hanna (v2 SDK + v2 server)."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

_langfuse = None
_enabled = False


def _get_langfuse():
    global _langfuse, _enabled
    if _langfuse is not None:
        return _langfuse

    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "http://hanna-langfuse:3000")

    if not pk or not sk:
        _enabled = False
        return None

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
        _enabled = True
        print("[langfuse] Connected")
        return _langfuse
    except Exception as e:
        print(f"[langfuse] Init failed: {e}")
        _enabled = False
        return None


def get_prompt(name: str, fallback: str = "") -> str:
    lf = _get_langfuse()
    if not lf or not _enabled:
        return fallback
    try:
        prompt = lf.get_prompt(name)
        return prompt.compile()
    except Exception:
        return fallback


class DraftTrace:
    def __init__(self, trace):
        self._trace = trace
        self._t0 = time.time()

    def search(self, results: list[dict], query: str = ""):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="rag_search",
                input={"query": query[:200]},
                output={
                    "result_count": len(results),
                    "top_score": results[0].get("score", 0) if results else 0,
                    "chunk_types": [r.get("chunk_type", "?") for r in results[:5]],
                },
            )
        except Exception:
            pass

    def llm(self, model: str = "", provider: str = "", **kwargs):
        if not self._trace:
            return
        try:
            self._trace.generation(
                name="llm_call",
                model=model,
                metadata={"provider": provider},
            )
        except Exception:
            pass

    def verify(self, nli_result=None, cove_result=None, guardrails=None):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="verification",
                output={
                    "nli": nli_result.get("overall_verdict") if nli_result else "skipped",
                    "cove": cove_result.get("overall") if cove_result else "skipped",
                    "guardrails_pass": guardrails.get("pass") if guardrails else True,
                },
            )
        except Exception:
            pass

    def output(self, body_html: str = "", confidence: str = "", category: str = ""):
        if not self._trace:
            return
        try:
            self._trace.update(
                output={
                    "confidence": confidence,
                    "category": category,
                    "body_length": len(body_html),
                    "duration_s": round(time.time() - self._t0, 1),
                },
            )
        except Exception:
            pass


@asynccontextmanager
async def trace_draft(email_text: str = "", subject: str = "", sender: str = ""):
    lf = _get_langfuse()
    trace = None
    if lf and _enabled:
        try:
            trace = lf.trace(
                name="draft_generate",
                input={"email_text": email_text[:500], "subject": subject, "sender": sender},
                metadata={"service": "hanna"},
            )
        except Exception:
            pass

    dt = DraftTrace(trace)
    try:
        yield dt
    finally:
        if lf:
            try:
                lf.flush()
            except Exception:
                pass
