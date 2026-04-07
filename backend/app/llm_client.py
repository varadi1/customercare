"""
Multi-provider LLM client with automatic fallback.

Priority: Anthropic (claude-opus-4-6) → OpenAI (gpt-5.4) → Google (gemini-flash-latest)
Each provider is tried in order; if one fails, the next is used.

Config via .env:
  OPENAI_API_KEY=sk-...
  ANTHROPIC_API_KEY=sk-ant-...
  GOOGLE_API_KEY=AIza...

Health check: GET /llm/health — tests all providers
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Provider configs
PROVIDERS = [
    {
        "name": "anthropic",
        "model": "claude-opus-4-6",
        "url": "https://api.anthropic.com/v1/messages",
        "key_attr": "anthropic_api_key",
        "format": "anthropic",
    },
    {
        "name": "openai",
        "model": "gpt-5.4",
        "url": "https://api.openai.com/v1/chat/completions",
        "key_attr": "openai_api_key",
        "format": "openai",
    },
    {
        "name": "google",
        "model": "gemini-flash-latest",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
        "key_attr": "google_api_key",
        "format": "google",
    },
]

# Vision model (for image analysis)
VISION_MODEL = "gpt-5.4"
VISION_URL = "https://api.openai.com/v1/chat/completions"


async def chat_completion(
    messages: list[dict],
    temperature: float = 0.15,
    max_tokens: int = 1000,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Call LLM with automatic fallback across providers.

    Returns: {"content": str, "provider": str, "model": str, "duration_ms": int}
    Raises: RuntimeError if all providers fail.
    """
    errors = []

    import asyncio as _aio

    for provider in PROVIDERS:
        api_key = getattr(settings, provider["key_attr"], "")
        if not api_key:
            continue

        # Retry up to 2 times per provider (with backoff)
        for attempt in range(2):
            try:
                t0 = time.time()
                content = await _call_provider(
                    provider, api_key, messages, temperature, max_tokens, json_mode,
                )
                duration_ms = int((time.time() - t0) * 1000)

                return {
                    "content": content,
                    "provider": provider["name"],
                    "model": provider["model"],
                    "duration_ms": duration_ms,
                }
            except Exception as e:
                logger.warning("LLM provider %s attempt %d failed: %s", provider["name"], attempt + 1, e)
                errors.append(f"{provider['name']}[{attempt+1}]: {e}")
                if attempt == 0:
                    await _aio.sleep(2)  # Wait before retry
                continue

    raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")


async def vision_completion(
    image_url: str,
    prompt: str,
    max_tokens: int = 500,
) -> str:
    """Call Vision model for image analysis. OpenAI only (gpt-5.4)."""
    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("OpenAI API key not configured for vision")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            VISION_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": VISION_MODEL,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def health_check() -> dict[str, Any]:
    """Test all providers and return status."""
    results = {}

    for provider in PROVIDERS:
        api_key = getattr(settings, provider["key_attr"], "")
        if not api_key:
            results[provider["name"]] = {"status": "no_key", "model": provider["model"]}
            continue

        try:
            t0 = time.time()
            content = await _call_provider(
                provider, api_key,
                [{"role": "user", "content": "Respond with just 'ok'."}],
                temperature=0, max_tokens=5, json_mode=False,
            )
            duration = int((time.time() - t0) * 1000)
            results[provider["name"]] = {
                "status": "ok",
                "model": provider["model"],
                "duration_ms": duration,
                "response": content[:20],
            }
        except Exception as e:
            results[provider["name"]] = {
                "status": "error",
                "model": provider["model"],
                "error": str(e)[:100],
            }

    return results


async def _call_provider(
    provider: dict,
    api_key: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    """Call a specific provider's API."""
    fmt = provider["format"]

    async with httpx.AsyncClient(timeout=60) as client:
        if fmt == "openai":
            return await _call_openai(client, provider, api_key, messages, temperature, max_tokens, json_mode)
        elif fmt == "anthropic":
            return await _call_anthropic(client, provider, api_key, messages, temperature, max_tokens, json_mode)
        elif fmt == "google":
            return await _call_google(client, provider, api_key, messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown format: {fmt}")


async def _call_openai(client, provider, api_key, messages, temperature, max_tokens, json_mode) -> str:
    model = provider["model"]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # GPT-5.x models use max_completion_tokens instead of max_tokens
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = await client.post(
        provider["url"],
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_anthropic(client, provider, api_key, messages, temperature, max_tokens, json_mode=False) -> str:
    # Convert OpenAI format to Anthropic format
    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": provider["model"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": user_messages,
    }
    if system_msg:
        if json_mode:
            system_msg += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no code blocks, just the raw JSON object."
        payload["system"] = system_msg

    resp = await client.post(
        provider["url"],
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["content"][0]["text"]

    return text


async def _call_google(client, provider, api_key, messages, temperature, max_tokens) -> str:
    # Convert OpenAI format to Google format
    contents = []
    system_instruction = None
    for m in messages:
        if m["role"] == "system":
            system_instruction = {"parts": [{"text": m["content"]}]}
        else:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    url = f"{provider['url']}?key={api_key}"
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
