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


def test_resolve_model_prefers_local_dir(tmp_path):
    # A downloaded model at models/faster-whisper-small should be used by name.
    local = tmp_path / "faster-whisper-small"
    local.mkdir()
    (local / "model.bin").write_bytes(b"x")
    assert transcribe._resolve_model("small", models_dir=tmp_path) == str(local)


def test_resolve_model_falls_back_to_name_when_absent(tmp_path):
    # No local copy -> return the name so faster-whisper downloads it.
    assert transcribe._resolve_model("base", models_dir=tmp_path) == "base"


def test_resolve_model_passes_through_explicit_path(tmp_path):
    p = str(tmp_path / "some-dir")
    assert transcribe._resolve_model(p, models_dir=tmp_path) == p
