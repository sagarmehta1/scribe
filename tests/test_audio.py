"""Tests for audio loading and denoising."""

import numpy as np
import pytest

from scribe import audio


def test_load_audio_returns_float32_mono_16k(sample_wav):
    samples = audio.load_audio(sample_wav)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    # ~1 second at 16 kHz (allow a little slack for resampler edges)
    assert 15000 < len(samples) < 17000
    # The tone is normalized to roughly [-1, 1]
    assert np.max(np.abs(samples)) <= 1.01


def test_load_audio_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        audio.load_audio(tmp_path / "does_not_exist.wav")


def test_denoise_preserves_shape_and_dtype(sample_wav):
    samples = audio.load_audio(sample_wav)
    out = audio.denoise(samples)
    assert out.dtype == np.float32
    assert out.shape == samples.shape


def test_denoise_empty_array_is_noop():
    empty = np.zeros(0, dtype=np.float32)
    assert audio.denoise(empty).size == 0
