"""Orchestrate the full transcription pipeline with progress reporting.

Stages (optional ones skipped by config): extract -> denoise? -> transcribe ->
diarize? -> clean -> summarize -> done. Each stage reports via a
``progress(stage, percent)`` callback so the web layer can show live status.

Stage implementations are injected (defaulting to the real modules) so the
orchestration logic is unit-testable without loading models or audio.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from . import audio as audio_mod
from . import diarize as diarize_mod
from . import llm as llm_mod
from . import transcribe as transcribe_mod
from .config import Settings


def _full_text(segments: List[dict]) -> str:
    return " ".join(s["text"] for s in segments).strip()


def run(
    audio_path,
    settings: Settings,
    progress: Callable[[str, int], None],
    *,
    load_audio=audio_mod.load_audio,
    denoise=audio_mod.denoise,
    transcribe=transcribe_mod.transcribe,
    diarize_file=diarize_mod.diarize_file,
    assign_speakers=diarize_mod.assign_speakers,
    get_provider=llm_mod.get_provider,
    clean_transcript=llm_mod.clean_transcript,
    summarize=llm_mod.summarize,
) -> dict:
    progress("extract", 5)
    samples = load_audio(audio_path)

    if settings.denoise:
        progress("denoise", 20)
        samples = denoise(samples)

    progress("transcribe", 40)
    segments = transcribe(samples, model_size=settings.whisper_model)

    speaker_labelled = False
    if settings.diarize:
        progress("diarize", 70)
        try:
            turns = diarize_file(audio_path, settings.hf_token)
            segments = assign_speakers(segments, turns)
            speaker_labelled = True
        except Exception:
            # Diarization is optional — fall back to no labels rather than failing.
            speaker_labelled = False
    if not speaker_labelled:
        segments = [{**s, "speaker": None} for s in segments]

    provider = get_provider(settings)

    progress("clean", 80)
    cleaned_text = clean_transcript(_full_text(segments), provider)

    progress("summarize", 90)
    summary = summarize(cleaned_text or _full_text(segments), provider)

    progress("done", 100)
    return {"segments": segments, "cleaned_text": cleaned_text, "summary": summary}
