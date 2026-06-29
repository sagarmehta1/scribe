"""Optional speaker diarization via pyannote.audio.

Two parts:
  * ``assign_speakers`` — pure logic mapping speaker turns onto transcript
    segments by maximum time overlap. Always available, fully tested.
  * ``diarize_file`` — thin wrapper around pyannote that produces speaker turns.
    Requires ``pip install pyannote.audio`` and a HuggingFace token; imported
    lazily so the core app works without the heavy dependency.
"""

from __future__ import annotations

from typing import List, Optional

Segment = dict
Turn = dict


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segments: List[Segment], turns: List[Turn]) -> List[Segment]:
    """Return copies of ``segments`` with a ``speaker`` set to the turn that
    overlaps each segment the most (or ``None`` if nothing overlaps)."""
    result = []
    for seg in segments:
        best_speaker: Optional[str] = None
        best_overlap = 0.0
        for turn in turns:
            ov = _overlap(seg["start"], seg["end"], turn["start"], turn["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn["speaker"]
        result.append({**seg, "speaker": best_speaker})
    return result


def is_available() -> bool:
    """Whether pyannote.audio can be imported."""
    try:
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


def diarize_file(wav_path: str, hf_token: str) -> List[Turn]:
    """Run pyannote diarization on a wav file and return speaker turns.

    Raises a clear error if pyannote isn't installed or no token is given.
    """
    if not hf_token:
        raise RuntimeError(
            "Diarization needs a HuggingFace token. Set SCRIBE_HF_TOKEN or "
            "disable speaker labels."
        )
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "Speaker labels need pyannote.audio. Install it with "
            "`pip install pyannote.audio`."
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    annotation = pipeline(wav_path)
    turns: List[Turn] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({"start": segment.start, "end": segment.end, "speaker": speaker})
    return turns
