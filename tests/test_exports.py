"""Tests for transcript export formats."""

import json

from scribe import exports

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Hello there.", "speaker": "SPEAKER_00"},
    {"start": 2.5, "end": 5.0, "text": "General Kenobi.", "speaker": "SPEAKER_01"},
    {"start": 3661.0, "end": 3663.0, "text": "An hour later.", "speaker": None},
]


def test_format_timestamp_srt():
    # 1h 1m 1s -> 01:01:01,000  (comma decimal separator for SRT)
    assert exports.format_timestamp(3661.0, sep=",") == "01:01:01,000"


def test_format_timestamp_vtt():
    # VTT uses a dot decimal separator
    assert exports.format_timestamp(3661.5, sep=".") == "01:01:01.500"


def test_to_srt_has_indexed_cues_and_arrows():
    out = exports.to_srt(SEGMENTS)
    assert out.startswith("1\n")
    assert "00:00:00,000 --> 00:00:02,500" in out
    assert "Hello there." in out
    # Three cues -> the index "3" must appear
    assert "\n3\n" in out


def test_to_vtt_starts_with_header():
    out = exports.to_vtt(SEGMENTS)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in out


def test_to_txt_is_plain_lines():
    out = exports.to_txt(SEGMENTS)
    lines = out.strip().splitlines()
    assert lines[0] == "Hello there."
    assert lines[1] == "General Kenobi."


def test_to_markdown_includes_speakers_and_timestamps():
    out = exports.to_markdown(SEGMENTS)
    assert "SPEAKER_00" in out
    assert "[00:00:00]" in out
    assert "Hello there." in out


def test_to_json_roundtrips():
    out = exports.to_json(SEGMENTS)
    parsed = json.loads(out)
    assert parsed[0]["text"] == "Hello there."
    assert parsed[1]["speaker"] == "SPEAKER_01"


def test_render_dispatches_by_format():
    assert exports.render(SEGMENTS, "srt") == exports.to_srt(SEGMENTS)
    assert exports.render(SEGMENTS, "vtt") == exports.to_vtt(SEGMENTS)
    assert exports.render(SEGMENTS, "txt") == exports.to_txt(SEGMENTS)


def test_render_rejects_unknown_format():
    try:
        exports.render(SEGMENTS, "docx")
        assert False, "expected ValueError"
    except ValueError:
        pass
