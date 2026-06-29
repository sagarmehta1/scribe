"""Render a list of transcript segments into various file formats.

A *segment* is a plain dict with keys:
    start   (float, seconds)
    end     (float, seconds)
    text    (str)
    speaker (str or None)

Keeping segments as plain dicts makes JSON serialization and the web layer trivial.
"""

from __future__ import annotations

import json
from typing import Iterable

Segment = dict


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep="," for SRT, "." for VTT)."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _clock(seconds: float) -> str:
    """Short HH:MM:SS clock for human-readable formats."""
    return format_timestamp(seconds, sep=".").split(".")[0]


def to_srt(segments: Iterable[Segment]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"], sep=",")
        end = format_timestamp(seg["end"], sep=",")
        text = seg["text"].strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(segments: Iterable[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp(seg["start"], sep=".")
        end = format_timestamp(seg["end"], sep=".")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def to_txt(segments: Iterable[Segment]) -> str:
    return "\n".join(seg["text"].strip() for seg in segments) + "\n"


def to_markdown(segments: Iterable[Segment]) -> str:
    lines = ["# Transcript", ""]
    for seg in segments:
        ts = _clock(seg["start"])
        speaker = seg.get("speaker")
        prefix = f"**{speaker}** " if speaker else ""
        lines.append(f"{prefix}[{ts}] {seg['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


def to_json(segments: Iterable[Segment]) -> str:
    return json.dumps(list(segments), indent=2, ensure_ascii=False)


_RENDERERS = {
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_txt,
    "md": to_markdown,
    "markdown": to_markdown,
    "json": to_json,
}


def render(segments: Iterable[Segment], fmt: str) -> str:
    """Dispatch to the renderer for ``fmt`` (case-insensitive)."""
    segments = list(segments)
    key = fmt.lower()
    if key not in _RENDERERS:
        raise ValueError(f"Unknown export format: {fmt!r}. "
                         f"Valid: {', '.join(sorted(_RENDERERS))}")
    return _RENDERERS[key](segments)
