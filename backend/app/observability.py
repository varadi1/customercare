"""Langfuse observability for CustomerCare (v2 SDK + v2 server).

Provides tracing for the entire draft generation pipeline:
- Root trace per draft/email
- Spans for pipeline steps (RAG, verification, external services, etc.)
- Generations for LLM calls (with token counts + cost)
- Autonomous processor traces
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

_langfuse = None
_enabled = False


def _get_langfuse():
    global _langfuse, _enabled
    if _langfuse is not None:
        return _langfuse

    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "http://cc-langfuse:3000")

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


def flush():
    """Flush Langfuse events (call at end of request or batch)."""
    lf = _get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Usage normalization (different providers return different formats)
# ---------------------------------------------------------------------------

def _normalize_usage(usage: dict | None, provider: str = "") -> dict:
    """Normalize token usage from different providers to Langfuse format."""
    if not usage:
        return {}
    # Anthropic: {"input_tokens": N, "output_tokens": N}
    # OpenAI: {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    # Google: {"promptTokenCount": N, "candidatesTokenCount": N, "totalTokenCount": N}
    return {
        "input": (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("promptTokenCount")
        ),
        "output": (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("candidatesTokenCount")
        ),
        "total": (
            usage.get("total_tokens")
            or usage.get("totalTokenCount")
            or (
                (usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                + (usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            )
        ),
    }


# ---------------------------------------------------------------------------
# DraftTrace — main tracing class for draft generation pipeline
# ---------------------------------------------------------------------------

class DraftTrace:
    """Trace wrapper for draft generation pipeline.

    All methods are no-op safe — they silently skip if Langfuse is not enabled.
    """

    def __init__(self, trace):
        self._trace = trace
        self._t0 = time.time()

    # --- RAG Search ---
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

    # --- Primary LLM call (draft generation) ---
    def llm(self, model: str = "", provider: str = "", usage: dict | None = None, **kwargs):
        if not self._trace:
            return
        try:
            gen_kwargs = {
                "name": "draft_llm",
                "model": model,
                "metadata": {"provider": provider},
            }
            norm = _normalize_usage(usage, provider)
            if norm:
                gen_kwargs["usage"] = norm
            self._trace.generation(**gen_kwargs)
        except Exception:
            pass

    # --- Verification summary ---
    def verify(self, nli_result=None, cove_result=None, guardrails=None,
               selfcheck=None, alignment=None, legal=None):
        if not self._trace:
            return
        try:
            output = {
                "nli": nli_result.get("overall_verdict") if nli_result else "skipped",
                "cove": cove_result.get("overall") if cove_result else "skipped",
                "guardrails_pass": guardrails.get("pass") if guardrails else True,
            }
            if selfcheck:
                output["selfcheck_consistent"] = selfcheck.get("consistent", True)
                output["selfcheck_avg_sim"] = selfcheck.get("avg_similarity")
            if alignment:
                output["alignment_verdict"] = alignment.get("verdict", "skipped")
            if legal:
                output["legal_risk"] = legal.get("risk_level", "none")
            self._trace.span(name="verification", output=output)
        except Exception:
            pass

    # --- Draft output ---
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

    # --- CoVe LLM call ---
    def cove(self, result: dict | None = None, usage: dict | None = None,
             model: str = "", provider: str = "", duration_ms: int = 0):
        if not self._trace:
            return
        try:
            gen_kwargs = {
                "name": "cove_verification",
                "model": model,
                "metadata": {"provider": provider},
                "output": result or {},
            }
            norm = _normalize_usage(usage, provider)
            if norm:
                gen_kwargs["usage"] = norm
            if duration_ms:
                gen_kwargs["metadata"]["duration_ms"] = duration_ms
            self._trace.generation(**gen_kwargs)
        except Exception:
            pass

    # --- SelfCheck LLM calls ---
    def selfcheck(self, result: dict | None = None, n_calls: int = 0,
                  total_usage: dict | None = None, model: str = "", provider: str = ""):
        if not self._trace:
            return
        try:
            gen_kwargs = {
                "name": "selfcheck",
                "model": model,
                "metadata": {"provider": provider, "n_samples": n_calls},
                "output": result or {},
            }
            norm = _normalize_usage(total_usage, provider)
            if norm:
                gen_kwargs["usage"] = norm
            self._trace.generation(**gen_kwargs)
        except Exception:
            pass

    # --- Answer-Question Alignment LLM call ---
    def alignment(self, result: dict | None = None, usage: dict | None = None,
                  model: str = "", provider: str = "", duration_ms: int = 0):
        if not self._trace:
            return
        try:
            gen_kwargs = {
                "name": "answer_alignment",
                "model": model,
                "metadata": {"provider": provider},
                "output": result or {},
            }
            norm = _normalize_usage(usage, provider)
            if norm:
                gen_kwargs["usage"] = norm
            self._trace.generation(**gen_kwargs)
        except Exception:
            pass

    # --- Legal risk check ---
    def legal_check(self, result: dict | None = None, duration_ms: int = 0):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="legal_risk_check",
                output=result or {},
                metadata={"duration_ms": duration_ms},
            )
        except Exception:
            pass

    # --- External service calls (VerbatimRAG, NLI) ---
    def external_service(self, name: str, result: Any = None,
                         duration_ms: int = 0, status: str = "ok", url: str = ""):
        if not self._trace:
            return
        try:
            self._trace.span(
                name=name,
                output={"status": status, "result_preview": str(result)[:300]} if result else {"status": status},
                metadata={"duration_ms": duration_ms, "url": url},
            )
        except Exception:
            pass

    # --- Skip filter decision ---
    def skip_filter(self, skipped: bool, reason: str = "", details: dict | None = None):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="skip_filter",
                output={"skipped": skipped, "reason": reason, **(details or {})},
            )
        except Exception:
            pass

    # --- Guardrails (rule-by-rule) ---
    def guardrails(self, result: dict | None = None, warnings: list | None = None):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="guardrails",
                output={
                    "pass": result.get("pass", True) if result else True,
                    "warning_count": len(warnings) if warnings else 0,
                    "warnings": [w.get("rule", "?") for w in (warnings or [])],
                },
            )
        except Exception:
            pass

    # --- Program DB enrichment ---
    def db_enrichment(self, app_ids: list[str] | None = None,
                      found: int = 0, duration_ms: int = 0):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="program_db_enrichment",
                input={"app_ids": app_ids or []},
                output={"found": found},
                metadata={"duration_ms": duration_ms},
            )
        except Exception:
            pass

    # --- Draft save to Outlook ---
    def draft_save(self, success: bool, draft_id: str = "",
                   confidence: str = "", duration_ms: int = 0):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="outlook_draft_save",
                output={"success": success, "draft_id": draft_id[:30], "confidence": confidence},
                metadata={"duration_ms": duration_ms},
            )
        except Exception:
            pass

    # --- Greeting generation ---
    def greeting(self, greeting_text: str = "", source: str = ""):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="greeting",
                output={"greeting": greeting_text, "source": source},
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Draft trace context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def trace_draft(email_text: str = "", subject: str = "", sender: str = ""):
    lf = _get_langfuse()
    trace = None
    if lf and _enabled:
        try:
            trace = lf.trace(
                name="draft_generate",
                input={"email_text": email_text[:500], "subject": subject, "sender": sender},
                metadata={"service": "customercare"},
            )
        except Exception:
            pass

    dt = DraftTrace(trace)
    try:
        yield dt
    finally:
        flush()


# ---------------------------------------------------------------------------
# Autonomous processor trace
# ---------------------------------------------------------------------------

@asynccontextmanager
async def trace_processor(mailbox: str = "", batch_size: int = 0):
    """Root trace for autonomous email processing batch."""
    lf = _get_langfuse()
    trace = None
    if lf and _enabled:
        try:
            trace = lf.trace(
                name="email_processor",
                input={"mailbox": mailbox, "batch_size": batch_size},
                metadata={"service": "customercare", "type": "autonomous"},
            )
        except Exception:
            pass

    ctx = ProcessorTrace(trace)
    try:
        yield ctx
    finally:
        flush()


class ProcessorTrace:
    """Trace wrapper for autonomous email processor."""

    def __init__(self, trace):
        self._trace = trace
        self._t0 = time.time()

    def poll(self, mailbox: str, email_count: int, duration_ms: int = 0):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="email_poll",
                input={"mailbox": mailbox},
                output={"email_count": email_count},
                metadata={"duration_ms": duration_ms},
            )
        except Exception:
            pass

    def email_processed(self, subject: str = "", result: str = "",
                        confidence: str = "", duration_ms: int = 0):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="email_processed",
                input={"subject": subject[:100]},
                output={"result": result, "confidence": confidence},
                metadata={"duration_ms": duration_ms},
            )
        except Exception:
            pass

    def email_skipped(self, subject: str = "", reason: str = ""):
        if not self._trace:
            return
        try:
            self._trace.span(
                name="email_skipped",
                input={"subject": subject[:100]},
                output={"reason": reason},
            )
        except Exception:
            pass

    def batch_complete(self, total: int = 0, drafted: int = 0, skipped: int = 0, errors: int = 0):
        if not self._trace:
            return
        try:
            self._trace.update(
                output={
                    "total": total,
                    "drafted": drafted,
                    "skipped": skipped,
                    "errors": errors,
                    "duration_s": round(time.time() - self._t0, 1),
                },
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Feedback analytics trace
# ---------------------------------------------------------------------------

@asynccontextmanager
async def trace_feedback(operation: str = "check"):
    """Root trace for feedback analytics operations."""
    lf = _get_langfuse()
    trace = None
    if lf and _enabled:
        try:
            trace = lf.trace(
                name=f"feedback_{operation}",
                metadata={"service": "customercare", "type": "learning"},
            )
        except Exception:
            pass

    yield trace
    flush()


def feedback_generation(trace, name: str, model: str = "", provider: str = "",
                        usage: dict | None = None, result: Any = None):
    """Log an LLM generation within a feedback trace."""
    if not trace:
        return
    try:
        gen_kwargs = {
            "name": name,
            "model": model,
            "metadata": {"provider": provider},
            "output": result if isinstance(result, (dict, str)) else str(result)[:500],
        }
        norm = _normalize_usage(usage, provider)
        if norm:
            gen_kwargs["usage"] = norm
        trace.generation(**gen_kwargs)
    except Exception:
        pass
