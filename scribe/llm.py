"""The AI layer: transcript cleanup and summarization.

Two paths:
  * A free, offline **regex fallback** (`regex_clean`) that removes filler words,
    collapses whitespace, fixes spacing, and capitalizes sentences. Always works,
    no model needed.
  * Pluggable **LLM providers** (Ollama / Anthropic / OpenAI) behind a tiny
    `complete(system, user) -> str` interface, for higher-quality cleanup and
    summaries.

All three providers speak plain HTTP via httpx — one client, minimal deps, and a
uniform shape across a local model (Ollama) and two hosted APIs. The Anthropic
path follows the documented Messages API wire format.
"""

from __future__ import annotations

import re
from typing import Optional, Protocol

import httpx

from .config import Settings

# ---------------------------------------------------------------------------
# Regex fallback cleaner (no LLM)
# ---------------------------------------------------------------------------

_FILLERS = ["umm", "um", "uhh", "uhm", "uh", "erm", "hmm", "you know", "i mean"]
_FILLER_RE = re.compile(
    r"\s*,?\s*\b(?:" + "|".join(_FILLERS) + r")\b\s*,?\s*",
    re.IGNORECASE,
)


def _capitalize_sentences(text: str) -> str:
    text = re.sub(r"^\s*([a-z])", lambda m: m.group(1).upper(), text)
    text = re.sub(r"([.!?]\s+)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def regex_clean(text: str) -> str:
    """Conservative, offline transcript cleanup."""
    out = _FILLER_RE.sub(" ", text)
    out = re.sub(r"\s+", " ", out)          # collapse whitespace
    out = re.sub(r"\s+([,.!?])", r"\1", out)  # no space before punctuation
    out = re.sub(r",{2,}", ",", out)         # collapse repeated commas
    out = out.strip()
    return _capitalize_sentences(out)


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

_TIMEOUT = 120.0


class Provider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OllamaProvider:
    """Local, free model served by Ollama (https://ollama.com)."""

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


class AnthropicProvider:
    """Claude via the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()


class OpenAIProvider:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def get_provider(settings: Settings) -> Optional[Provider]:
    """Build the configured provider, or None to use the regex fallback."""
    p = settings.llm_provider
    if p == "ollama":
        return OllamaProvider(settings.ollama_host, settings.ollama_model)
    if p == "anthropic" and settings.anthropic_api_key:
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    if p == "openai" and settings.openai_api_key:
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)
    return None


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------

_CLEAN_SYSTEM = (
    "You are a transcript editor. Clean up the given raw speech-to-text transcript: "
    "remove filler words (um, uh, you know), fix punctuation and capitalization, and "
    "break it into readable paragraphs. Do NOT summarize, paraphrase, omit content, or "
    "add commentary. Preserve the speaker's words and meaning. Return only the cleaned "
    "transcript."
)

_SUMMARY_SYSTEM = (
    "You are a meeting-notes assistant. Given a transcript, produce concise notes in "
    "Markdown with three sections: a one-paragraph '## TL;DR', a '## Key Points' bullet "
    "list, and an '## Action Items' bullet list (write 'None' if there are none). "
    "Base everything strictly on the transcript."
)


def clean_transcript(text: str, provider: Optional[Provider]) -> str:
    """Clean a transcript with the LLM if available, else the regex fallback."""
    if provider is None:
        return regex_clean(text)
    return provider.complete(_CLEAN_SYSTEM, text)


def summarize(text: str, provider: Optional[Provider]) -> Optional[str]:
    """Summarize a transcript, or None if no LLM is configured."""
    if provider is None:
        return None
    return provider.complete(_SUMMARY_SYSTEM, text)
