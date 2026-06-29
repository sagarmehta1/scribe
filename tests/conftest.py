"""Shared test fixtures."""

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sample_wav(tmp_path):
    """A 1-second 440 Hz mono tone at 16 kHz, written to a temp .wav file."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, tone.astype(np.float32), sr)
    return path
