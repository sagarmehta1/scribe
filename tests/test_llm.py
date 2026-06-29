"""Tests for the LLM layer: regex fallback, provider factory, clean/summarize."""

from scribe import config, llm


# ---- regex fallback cleaner (no LLM needed) ----

def test_regex_clean_removes_leading_filler():
    assert llm.regex_clean("Um, hello.") == "Hello."


def test_regex_clean_removes_midsentence_filler_phrase():
    assert llm.regex_clean("This is, you know, great.") == "This is great."


def test_regex_clean_collapses_whitespace():
    assert llm.regex_clean("Hello     world") == "Hello world"


def test_regex_clean_capitalizes_sentences():
    assert llm.regex_clean("hello. how are you?") == "Hello. How are you?"


# ---- provider factory ----

def test_get_provider_none():
    assert llm.get_provider(config.Settings(llm_provider="none")) is None


def test_get_provider_ollama():
    p = llm.get_provider(config.Settings(llm_provider="ollama"))
    assert isinstance(p, llm.OllamaProvider)


def test_get_provider_anthropic_requires_key():
    # No key -> falls back to None rather than a broken provider
    assert llm.get_provider(config.Settings(llm_provider="anthropic")) is None
    p = llm.get_provider(config.Settings(llm_provider="anthropic",
                                         anthropic_api_key="sk-x"))
    assert isinstance(p, llm.AnthropicProvider)


# ---- clean_transcript / summarize dispatch ----

class FakeProvider:
    def __init__(self):
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return "FAKE_OUTPUT"


def test_clean_transcript_uses_provider_when_present():
    fake = FakeProvider()
    out = llm.clean_transcript("um hello", provider=fake)
    assert out == "FAKE_OUTPUT"
    assert len(fake.calls) == 1


def test_clean_transcript_falls_back_to_regex_without_provider():
    assert llm.clean_transcript("Um, hello.", provider=None) == "Hello."


def test_summarize_returns_none_without_provider():
    assert llm.summarize("some transcript", provider=None) is None


def test_summarize_uses_provider():
    fake = FakeProvider()
    out = llm.summarize("transcript text", provider=fake)
    assert out == "FAKE_OUTPUT"


def test_ollama_provider_builds_request(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json

        class R:
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "hi there"}}
        return R()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    p = llm.OllamaProvider(host="http://localhost:11434", model="llama3.1")
    assert p.complete("sys", "user") == "hi there"
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["model"] == "llama3.1"
