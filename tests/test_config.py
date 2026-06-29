"""Tests for settings loading, saving, and redaction."""

from scribe import config


def test_defaults_when_no_env_no_file(tmp_path):
    s = config.load(path=tmp_path / "config.json", env={})
    assert s.whisper_model == "base"
    assert s.llm_provider == "none"
    assert s.denoise is False
    assert s.diarize is False


def test_env_provides_defaults(tmp_path):
    env = {
        "SCRIBE_WHISPER_MODEL": "small",
        "SCRIBE_LLM_PROVIDER": "ollama",
        "SCRIBE_OLLAMA_MODEL": "llama3.1",
    }
    s = config.load(path=tmp_path / "config.json", env=env)
    assert s.whisper_model == "small"
    assert s.llm_provider == "ollama"
    assert s.ollama_model == "llama3.1"


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "config.json"
    s = config.Settings(whisper_model="medium", denoise=True, llm_provider="openai",
                        openai_api_key="sk-secret")
    config.save(s, path=path)
    loaded = config.load(path=path, env={})
    assert loaded.whisper_model == "medium"
    assert loaded.denoise is True
    assert loaded.openai_api_key == "sk-secret"


def test_file_overrides_env(tmp_path):
    path = tmp_path / "config.json"
    config.save(config.Settings(whisper_model="large-v3"), path=path)
    s = config.load(path=path, env={"SCRIBE_WHISPER_MODEL": "tiny"})
    assert s.whisper_model == "large-v3"


def test_public_redacts_secrets():
    s = config.Settings(openai_api_key="sk-secret", anthropic_api_key="",
                        hf_token="hf_token123")
    pub = s.public()
    # Raw secrets must never be exposed to the frontend
    assert "sk-secret" not in str(pub)
    assert "hf_token123" not in str(pub)
    # Instead, booleans indicate whether a key is configured
    assert pub["openai_api_key_set"] is True
    assert pub["anthropic_api_key_set"] is False
    assert pub["hf_token_set"] is True
