"""Tests for the faster-whisper wrapper.

The real model is exercised in the end-to-end run; here we inject a fake model so
the unit tests stay fast and offline.
"""

import numpy as np

from scribe import transcribe


class FakeSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class FakeModel:
    def __init__(self):
        self.called_with = None

    def transcribe(self, audio, **kwargs):
        self.called_with = audio
        return iter([FakeSeg(0.0, 1.5, " hi "), FakeSeg(1.5, 2.0, "there")]), {"language": "en"}


def test_to_segments_strips_text_and_floats_times():
    out = transcribe._to_segments([FakeSeg(0, 1.5, "  hi  ")])
    assert out == [{"start": 0.0, "end": 1.5, "text": "hi"}]


def test_transcribe_uses_injected_model_and_converts():
    model = FakeModel()
    out = transcribe.transcribe(np.zeros(10, dtype=np.float32), model=model)
    assert [s["text"] for s in out] == ["hi", "there"]
    assert out[0]["start"] == 0.0 and out[1]["end"] == 2.0


def test_transcribe_passes_audio_through():
    model = FakeModel()
    audio = np.ones(5, dtype=np.float32)
    transcribe.transcribe(audio, model=model)
    assert model.called_with is audio
