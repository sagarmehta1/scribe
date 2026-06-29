"""Audio loading and denoising.

Decoding goes through PyAV, whose wheels bundle the ffmpeg libraries — so there is
no system ffmpeg binary to install. Any container PyAV/ffmpeg can read (mp3, mp4,
m4a, wav, ...) is decoded, downmixed to mono, and resampled to 16 kHz, which is
exactly what Whisper expects.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

TARGET_SR = 16000


def load_audio(path) -> np.ndarray:
    """Decode any audio/video file to a 16 kHz mono float32 array in [-1, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    container = av.open(str(path))
    try:
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=TARGET_SR
        )
        chunks = []

        def _collect(frames):
            for frame in frames:
                chunks.append(frame.to_ndarray())

        for frame in container.decode(audio=0):
            _collect(resampler.resample(frame))
        _collect(resampler.resample(None))  # flush
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    data = np.concatenate(chunks, axis=1).reshape(-1)
    return (data.astype(np.float32) / 32768.0)


def denoise(samples: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Spectral-gating noise reduction. No-op on empty input."""
    if samples.size == 0:
        return samples
    import noisereduce as nr

    reduced = nr.reduce_noise(y=samples, sr=sr)
    return reduced.astype(np.float32)
