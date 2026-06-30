"""Speech-to-text via faster-whisper.

Models are loaded lazily and cached per (size, device, compute) so repeated jobs
reuse the same in-memory model. Device is auto-detected: CUDA + float16 when a GPU
is available, otherwise CPU + int8 (broadly compatible, no GPU needed).

Decoding is tuned for speed without meaningfully hurting accuracy on clear speech:
greedy decode (``beam_size=1``), VAD silence-skipping, all CPU cores, and a batched
inference pipeline that transcribes chunks in parallel instead of one serial stream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

_model_cache: dict = {}
_batched_cache: dict = {}

# Decode options shared by the real (batched) and injected code paths.
_DECODE_OPTS = {"beam_size": 1, "vad_filter": True}
# How many VAD-segmented chunks to transcribe at once on the batched pipeline.
_BATCH_SIZE = 8

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
        kwargs = {}
        if device == "cpu":
            # Use every logical core; ctranslate2 otherwise defaults to physical cores only.
            kwargs["cpu_threads"] = os.cpu_count() or 0
        _model_cache[key] = WhisperModel(
            resolved, device=device, compute_type=compute_type, **kwargs)
    return _model_cache[key]


def _get_batched(model_size: str):
    """Wrap (and cache) a model in a batched pipeline for parallel-chunk decoding."""
    base = get_model(model_size)
    key = id(base)
    if key not in _batched_cache:
        from faster_whisper import BatchedInferencePipeline
        _batched_cache[key] = BatchedInferencePipeline(model=base)
    return _batched_cache[key]


def _to_segments(whisper_segments) -> List[dict]:
    return [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in whisper_segments
    ]


def transcribe(samples: np.ndarray, model_size: str = "base", model=None) -> List[dict]:
    """Transcribe a 16 kHz mono float32 array into ``{start, end, text}`` segments."""
    if model is not None:
        # Injected model (tests / custom): call directly with the fast decode options.
        segments, _info = model.transcribe(samples, **_DECODE_OPTS)
    else:
        batched = _get_batched(model_size)
        segments, _info = batched.transcribe(samples, batch_size=_BATCH_SIZE, **_DECODE_OPTS)
    return _to_segments(segments)
