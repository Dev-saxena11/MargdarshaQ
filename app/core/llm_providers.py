"""
llm_providers.py
----------------
Pluggable LLM backends for the in-dashboard AI Assistant.

Why this module exists
======================
The assistant previously had Gemini and OpenAI calls inlined in
`assistant.py`, which made it awkward to add a provider or to reuse the same
LLM plumbing from anywhere else. Everything LLM-shaped now lives here behind
one interface, so:

  * adding a provider is a new subclass + one registry entry,
  * anything else that needs an LLM (e.g. the document-grounded chatbot in
    issue #33) can reuse this layer instead of re-implementing HTTP plumbing,
  * the assistant keeps working with no provider configured at all.

Provider selection
==================
`get_provider()` returns the first provider whose API key is present, in the
order listed by `PROVIDER_PRIORITY`. OpenRouter is first because it gives the
project free models without a billing account. If no key is set anywhere the
function returns None and the assistant falls back to its local deterministic
engine, which is always available.

Free-tier policy
================
This project deliberately runs on free models only. `OpenRouterProvider`
enforces that: a model id must be `openrouter/free` (OpenRouter's auto-router
over free models) or carry the `:free` suffix. Anything else raises
`ValueError` at construction, so a paid model can't be enabled by a typo in an
env var.

Configuration (all optional, all via environment):
    OPENROUTER_API_KEY   - https://openrouter.ai/keys
    OPENROUTER_MODEL     - default "openrouter/free"
    GEMINI_API_KEY
    OPENAI_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

# Free reasoning models (which OpenRouter's free router often selects) spend a
# large share of their completion budget on hidden reasoning tokens before
# emitting any visible text. Measured against `openrouter/free`: a 150-token cap
# produced 179 reasoning tokens and a 10-character truncated answer, while an
# 800-token cap returned a complete 418-character answer. Keep this generous.
DEFAULT_MAX_TOKENS = 800

# A reply shorter than this that was cut off by the token limit is treated as a
# failed generation rather than shown to the user as a stub answer.
MIN_USABLE_REPLY_CHARS = 40

HTTP_TIMEOUT_SECONDS = 25.0


class LLMProvider(ABC):
    """One external text-generation backend."""

    name: str = "unknown"

    @abstractmethod
    def is_configured(self) -> bool:
        """True when this provider has the credentials it needs."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> Optional[str]:
        """
        Generate a reply, or return None if the call failed or produced
        nothing usable. Implementations must not raise for ordinary network or
        API errors — returning None lets the caller fall back cleanly.
        """

    # -- shared helper ----------------------------------------------------

    @staticmethod
    def _post_json(url: str, payload: dict, headers: dict) -> Optional[dict]:
        """POST JSON and decode the response, or None on any failure."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            logger.warning("LLM HTTP %s from %s: %s", e.code, url, body)
        except Exception as e:
            logger.warning("LLM request to %s failed: %s", url, e)
        return None


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter (https://openrouter.ai) — free models only.

    `openrouter/free` is an auto-router that picks among currently-available
    free models, so the project keeps working even when one specific free model
    is retired or rate-limited.
    """

    name = "openrouter"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "openrouter/free"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL") or self.DEFAULT_MODEL
        if not self.is_free_model(self.model):
            raise ValueError(
                f"OPENROUTER_MODEL={self.model!r} is not a free model. This project "
                f"runs on free models only: use 'openrouter/free' or a model id "
                f"ending in ':free'."
            )

    @staticmethod
    def is_free_model(model_id: str) -> bool:
        """Free models are the auto-router itself or anything tagged ':free'."""
        if not model_id:
            return False
        return model_id == "openrouter/free" or model_id.endswith(":free")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> Optional[str]:
        if not self.is_configured():
            return None

        res = self._post_json(
            self.API_URL,
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # OpenRouter uses these for attribution on its dashboard.
                "HTTP-Referer": "https://sih-26137.vercel.app",
                "X-Title": "QuantaRoute SIH26137",
            },
        )
        if not res:
            return None
        if "error" in res:
            logger.warning("OpenRouter error: %s", str(res["error"])[:200])
            return None

        try:
            choice = res["choices"][0]
            text = (choice["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError):
            logger.warning("OpenRouter returned an unexpected response shape")
            return None

        if not text:
            # Reasoning models can spend the whole budget before emitting text.
            logger.warning("OpenRouter returned empty content (model=%s)", res.get("model"))
            return None

        if choice.get("finish_reason") == "length" and len(text) < MIN_USABLE_REPLY_CHARS:
            logger.warning(
                "OpenRouter reply truncated to %d chars (model=%s); discarding",
                len(text), res.get("model"),
            )
            return None

        return text


class GeminiProvider(LLMProvider):
    """Google Gemini (gemini-1.5-flash)."""

    name = "gemini"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> Optional[str]:
        if not self.is_configured():
            return None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        res = self._post_json(
            url,
            {
                "contents": [
                    {"role": "user",
                     "parts": [{"text": f"{system_prompt}\nUser query: {user_prompt}"}]}
                ],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
            },
            {"Content-Type": "application/json"},
        )
        if not res:
            return None
        try:
            return (res["candidates"][0]["content"]["parts"][0]["text"] or "").strip() or None
        except (KeyError, IndexError, TypeError):
            logger.warning("Gemini returned an unexpected response shape")
            return None


class OpenAIProvider(LLMProvider):
    """OpenAI chat completions (gpt-4o-mini by default)."""

    name = "openai"
    API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> Optional[str]:
        if not self.is_configured():
            return None

        res = self._post_json(
            self.API_URL,
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        if not res:
            return None
        try:
            return (res["choices"][0]["message"]["content"] or "").strip() or None
        except (KeyError, IndexError, TypeError):
            logger.warning("OpenAI returned an unexpected response shape")
            return None


# Order matters: the first configured provider wins. OpenRouter leads because
# it is the one that costs nothing to run.
PROVIDER_PRIORITY = [OpenRouterProvider, GeminiProvider, OpenAIProvider]


def available_providers() -> List[LLMProvider]:
    """Every provider that currently has credentials, in priority order."""
    out: List[LLMProvider] = []
    for cls in PROVIDER_PRIORITY:
        try:
            provider = cls()
        except ValueError as e:
            # e.g. a non-free OPENROUTER_MODEL. Skip rather than break the app.
            logger.warning("Skipping %s: %s", cls.__name__, e)
            continue
        if provider.is_configured():
            out.append(provider)
    return out


def get_provider() -> Optional[LLMProvider]:
    """
    The highest-priority configured provider, or None when the deployment has
    no LLM credentials at all (in which case the assistant uses its local
    deterministic engine).
    """
    providers = available_providers()
    return providers[0] if providers else None
