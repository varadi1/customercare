"""Langfuse observability integration for Hanna pipeline.

Traces every draft generation with:
- Input (email text, subject)
- RAG results (search scores, chunk types)
- LLM call (model, provider, tokens)
- Verification (NLI, CoVe, guardrails)
- Output (confidence, draft category)

Gracefully no-ops if Langfuse is not configured.

Usage:
    from app.observability import trace_draft

    async with trace_draft(email_text, subject) as t:
        t.search(results, scores)
        t.llm(model, provider, tokens)
        t.verify(nli_result, cove_result, guardrails)
        t.output(body_html, confidence)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

_langfuse = None
_enabled = False


def _get_langfuse():
    """Lazy-init Langfuse client."""
    global _langfuse, _enabled
    if _langfuse is not None:
        return _langfuse

    host = os.getenv("LANGFUSE_HOST", "http://hanna-langfuse:3000")
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")

    if not pk or not sk:
        _enabled = False
        return None

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=pk,
            secret_key=sk,
            host=host,
        )
        _enabled = True
        print("[langfuse] Connected")
        return _langfuse
    except Exception as e:
        print(f"[langfuse] Init failed: {e}")
        _enabled = False
        return None


class DraftTrace:
    """Wrapper for tracing a single draft generation."""

    def __init__(self, trace):
        self._trace = trace
        self._t0 = time.time()

    def search(self, results: list[dict], query: str = ""):
        """Log RAG search results."""
        if not self._trace:
            return
        try:
            self._trace.span(
                name="rag_search",
                input={"query": query[:500]},
                output={
                    "result_count": len(results),
                    "top_score": results[0].get("score", 0) if results else 0,
                    "chunk_types": [r.get("chunk_type", "?") for r in results[:5]],
                },
            )
        except Exception:
            pass

    def llm(self, model: str = "", provider: str = "", prompt_tokens: int = 0, completion_tokens: int = 0):
        """Log LLM generation."""
        if not self._trace:
            return
        try:
            self._trace.generation(
                name="draft_generation",
                model=model,
                metadata={"provider": provider},
                usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens} if prompt_tokens else None,
            )
        except Exception:
            pass

    def verify(self, nli_result: dict | None = None, cove_result: dict | None = None, guardrails: dict | None = None):
        """Log verification results."""
        if not self._trace:
            return
        try:
            self._trace.span(
                name="verification",
                output={
                    "nli": nli_result.get("overall_verdict") if nli_result else "skipped",
                    "cove": cove_result.get("overall") if cove_result else "skipped",
                    "guardrails_pass": guardrails.get("pass") if guardrails else True,
                    "guardrails_warnings": len(guardrails.get("warnings", [])) if guardrails else 0,
                },
            )
        except Exception:
            pass

    def output(self, body_html: str = "", confidence: str = "", category: str = ""):
        """Log final output."""
        if not self._trace:
            return
        try:
            duration = time.time() - self._t0
            self._trace.update(
                output={
                    "confidence": confidence,
                    "category": category,
                    "body_length": len(body_html),
                    "duration_s": round(duration, 1),
                },
            )
        except Exception:
            pass

    def score(self, name: str, value: float, comment: str = ""):
        """Add a score to the trace."""
        if not self._trace:
            return
        try:
            self._trace.score(name=name, value=value, comment=comment)
        except Exception:
            pass


def get_prompt(name: str, fallback: str = "") -> str:
    """Get prompt from Langfuse prompt management. Falls back to code if unavailable.

    Args:
        name: Prompt name in Langfuse (e.g. "draft_generate_system")
        fallback: Default prompt text if Langfuse unavailable

    Returns:
        Prompt text (from Langfuse if available, otherwise fallback)
    """
    lf = _get_langfuse()
    if not lf or not _enabled:
        return fallback

    try:
        prompt = lf.get_prompt(name)
        compiled = prompt.compile()
        return compiled
    except Exception:
        return fallback


@asynccontextmanager
async def trace_draft(email_text: str = "", subject: str = "", sender: str = ""):
    """Context manager for tracing a draft generation."""
    lf = _get_langfuse()
    trace = None

    if lf and _enabled:
        try:
            trace = lf.trace(
                name="draft_generate",
                input={
                    "email_text": email_text[:500],
                    "subject": subject,
                    "sender": sender,
                },
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
