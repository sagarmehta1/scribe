"""Tests for pipeline orchestration with injected fake stages."""

from scribe import config, pipeline


def _deps(**overrides):
    """Default fake stage implementations; override individually per test."""
    calls = {"denoise": 0}

    def load_audio(path):
        return "AUDIO"

    def denoise(samples):
        calls["denoise"] += 1
        return "DENOISED"

    def transcribe(samples, model_size=None):
        return [{"start": 0.0, "end": 1.0, "text": "hi"},
                {"start": 1.0, "end": 2.0, "text": "there"}]

    def get_provider(settings):
        return None

    def clean_transcript(text, provider):
        return "CLEANED"

    def summarize(text, provider):
        return None

    def diarize_file(path, hf_token):
        return [{"start": 0.0, "end": 2.0, "speaker": "S1"}]

    def assign_speakers(segments, turns):
        return [{**s, "speaker": "S1"} for s in segments]

    deps = dict(load_audio=load_audio, denoise=denoise, transcribe=transcribe,
                get_provider=get_provider, clean_transcript=clean_transcript,
                summarize=summarize, diarize_file=diarize_file,
                assign_speakers=assign_speakers)
    deps.update(overrides)
    return deps, calls


def test_basic_run_no_optional_stages():
    deps, calls = _deps()
    stages = []
    settings = config.Settings(denoise=False, diarize=False, llm_provider="none")
    result = pipeline.run("x.mp3", settings, progress=lambda s, p: stages.append(s), **deps)

    assert calls["denoise"] == 0                       # denoise skipped
    assert all(seg["speaker"] is None for seg in result["segments"])
    assert result["cleaned_text"] == "CLEANED"
    assert result["summary"] is None
    assert "transcribe" in stages and stages[-1] == "done"


def test_denoise_runs_when_enabled():
    deps, calls = _deps()
    settings = config.Settings(denoise=True, diarize=False, llm_provider="none")
    pipeline.run("x.mp3", settings, progress=lambda s, p: None, **deps)
    assert calls["denoise"] == 1


def test_diarize_assigns_speakers_when_enabled():
    deps, _ = _deps()
    settings = config.Settings(denoise=False, diarize=True, hf_token="hf_x",
                               llm_provider="none")
    result = pipeline.run("x.mp3", settings, progress=lambda s, p: None, **deps)
    assert all(seg["speaker"] == "S1" for seg in result["segments"])


def test_summary_flows_through_when_provider_present():
    deps, _ = _deps(get_provider=lambda s: object(),
                    summarize=lambda text, provider: "## TL;DR\nstuff")
    settings = config.Settings(denoise=False, diarize=False, llm_provider="ollama")
    result = pipeline.run("x.mp3", settings, progress=lambda s, p: None, **deps)
    assert "TL;DR" in result["summary"]


def test_diarize_failure_degrades_gracefully():
    def boom(path, hf_token):
        raise RuntimeError("no token")

    deps, _ = _deps(diarize_file=boom)
    settings = config.Settings(denoise=False, diarize=True, llm_provider="none")
    result = pipeline.run("x.mp3", settings, progress=lambda s, p: None, **deps)
    # Falls back to no speaker labels rather than crashing the whole job
    assert all(seg["speaker"] is None for seg in result["segments"])
