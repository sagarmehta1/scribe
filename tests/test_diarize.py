"""Tests for mapping diarization speaker-turns onto transcript segments."""

from scribe import diarize


def test_assign_speakers_picks_max_overlap():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hi"},
        {"start": 2.0, "end": 4.0, "text": "Bye"},
    ]
    turns = [
        {"start": 0.0, "end": 2.1, "speaker": "SPEAKER_00"},
        {"start": 2.1, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    out = diarize.assign_speakers(segments, turns)
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[1]["speaker"] == "SPEAKER_01"


def test_assign_speakers_partial_overlap_wins_by_majority():
    segments = [{"start": 0.0, "end": 10.0, "text": "long"}]
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": "A"},
        {"start": 3.0, "end": 10.0, "speaker": "B"},  # 7s overlap beats 3s
    ]
    out = diarize.assign_speakers(segments, turns)
    assert out[0]["speaker"] == "B"


def test_assign_speakers_no_overlap_gives_none():
    segments = [{"start": 100.0, "end": 101.0, "text": "orphan"}]
    turns = [{"start": 0.0, "end": 5.0, "speaker": "A"}]
    out = diarize.assign_speakers(segments, turns)
    assert out[0]["speaker"] is None


def test_assign_speakers_does_not_mutate_input():
    segments = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}]
    diarize.assign_speakers(segments, turns)
    assert "speaker" not in segments[0]


def test_assign_speakers_empty_turns_returns_none_speakers():
    segments = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
    out = diarize.assign_speakers(segments, [])
    assert out[0]["speaker"] is None
