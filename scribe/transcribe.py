"""Speech-to-text via faster-whisper.

Models are loaded lazily and cached per (size, device, compute) so repeated jobs
reuse the same in-memory model. Device is auto-detected: CUDA + float16 when a GPU
is available, otherwise CPU + int8 (broadly compatible, no GPU needed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

_model_cache: dict = {}

# Where pre-downloaded models live: <project>/models/faster-whisper-<size>/
_MODELS_DIR = Path(__file__).parent.parent / "models"


def _resolve_model(model_size: str, models_dir: Optional[Path] = None) -> str:
    """Resolve a model name to a local folder if one is present.

    Lets a bare name like "small" load from ``models/faster-whisper-small`` without
    touching the network. An explicit path (or a name with no local copy) is returned
    unchanged, so faster-whisper downloads it as usual for normal setups.
    """
    if os.path.sep in model_size or os.path.altsep and os.path.altsep in model_size:
        return model_size  # already a path
    base = Path(models_dir) if models_dir is not None else _MODELS_DIR
    local = base / f"faster-whisper-{model_size}"
    if (local / "model.bin").exists():
        return str(local)
    return model_size


def _auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_model(model_size: str = "base"):
    """Load (and cache) a WhisperModel for the given size."""
    device, compute_type = _auto_device()
    resolved = _resolve_model(model_size)
    key = (resolved, device, compute_type)
    if key not in _model_cache:
        from faster_whisper import WhisperModel
        _model_cache[key] = WhisperModel(resolved, device=device, compute_type=compute_type)
    return _model_cache[key]


def _to_segments(whisper_segments) -> List[dict]:
    return [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in whisper_segments
    ]


def transcribe(samples: np.ndarray, model_size: str = "base", model=None) -> List[dict]:
    """Transcribe a 16 kHz mono float32 array into ``{start, end, text}`` segments."""
    if model is None:
        model = get_model(model_size)
    segments, _info = model.transcribe(samples, beam_size=5)
    return _to_segments(segments)
