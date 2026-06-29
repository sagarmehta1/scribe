"""Application settings: defaults from environment, overridable via config.json.

Layering (lowest to highest priority):
    1. Hard-coded defaults on the Settings model
    2. Environment variables (SCRIBE_* — typically from a .env file)
    3. config.json written by the settings UI (runtime-editable)

Secrets (API keys, HF token) are never sent to the frontend; ``Settings.public()``
returns a redacted dict with ``*_set`` booleans instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Mapping, Optional

from pydantic import BaseModel

LLMProvider = Literal["none", "ollama", "anthropic", "openai"]

DATA_DIR = Path(os.environ.get("SCRIBE_DATA_DIR", "data"))
DEFAULT_CONFIG_PATH = DATA_DIR / "config.json"

# Map Settings field -> environment variable name.
_ENV_MAP = {
    "llm_provider": "SCRIBE_LLM_PROVIDER",
    "ollama_host": "SCRIBE_OLLAMA_HOST",
    "ollama_model": "SCRIBE_OLLAMA_MODEL",
    "anthropic_api_key": "SCRIBE_ANTHROPIC_API_KEY",
    "anthropic_model": "SCRIBE_ANTHROPIC_MODEL",
    "openai_api_key": "SCRIBE_OPENAI_API_KEY",
    "openai_model": "SCRIBE_OPENAI_MODEL",
    "whisper_model": "SCRIBE_WHISPER_MODEL",
    "hf_token": "SCRIBE_HF_TOKEN",
}

_SECRET_FIELDS = ("anthropic_api_key", "openai_api_key", "hf_token")


class Settings(BaseModel):
    # AI layer
    llm_provider: LLMProvider = "none"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Transcription
    whisper_model: str = "base"

    # Optional stages
    denoise: bool = False
    diarize: bool = False
    hf_token: str = ""

    def public(self) -> dict:
        """Redacted view safe to send to the browser."""
        data = self.model_dump()
        for field in _SECRET_FIELDS:
            data[f"{field}_set"] = bool(data.pop(field))
        return data


def _env_overrides(env: Mapping[str, str]) -> dict:
    out = {}
    for field, var in _ENV_MAP.items():
        if env.get(var):
            out[field] = env[var]
    return out


def load(path: Optional[Path] = None, env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build Settings from defaults <- env <- config.json."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if env is None:
        env = os.environ

    values = _env_overrides(env)

    path = Path(path)
    if path.exists():
        file_values = json.loads(path.read_text(encoding="utf-8"))
        values.update({k: v for k, v in file_values.items() if v is not None})

    return Settings(**values)


def save(settings: Settings, path: Optional[Path] = None) -> None:
    """Persist settings to config.json (creating parent dirs as needed)."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
