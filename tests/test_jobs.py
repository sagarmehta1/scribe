"""Tests for the job store (SQLite metadata + on-disk results)."""

import pytest

from scribe.jobs import JobStore


@pytest.fixture
def store(tmp_path):
    return JobStore(db_path=tmp_path / "jobs.db", data_dir=tmp_path / "data")


def test_create_returns_id_and_queued_status(store):
    job_id = store.create(filename="meeting.mp3", denoise=True, diarize=False)
    job = store.get(job_id)
    assert job["id"] == job_id
    assert job["filename"] == "meeting.mp3"
    assert job["status"] == "queued"
    assert job["denoise"] is True
    assert job["diarize"] is False
    assert job["segments"] is None


def test_job_dir_created(store, tmp_path):
    job_id = store.create(filename="a.mp3")
    assert store.job_dir(job_id).is_dir()


def test_list_returns_created_jobs(store):
    a = store.create(filename="a.mp3")
    b = store.create(filename="b.mp3")
    ids = {j["id"] for j in store.list()}
    assert ids == {a, b}


def test_update_status_sets_stage_and_progress(store):
    job_id = store.create(filename="a.mp3")
    store.update_status(job_id, status="running", stage="transcribe", progress=50)
    job = store.get(job_id)
    assert job["status"] == "running"
    assert job["stage"] == "transcribe"
    assert job["progress"] == 50


def test_save_result_marks_done_and_persists(store):
    job_id = store.create(filename="a.mp3")
    segments = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}]
    store.save_result(job_id, segments=segments, summary="## TL;DR\nA greeting.")
    job = store.get(job_id)
    assert job["status"] == "done"
    assert job["segments"] == segments
    assert "TL;DR" in job["summary"]


def test_update_transcript_persists_edits(store):
    job_id = store.create(filename="a.mp3")
    store.save_result(job_id, segments=[{"start": 0, "end": 1, "text": "old", "speaker": None}],
                      summary=None)
    edited = [{"start": 0, "end": 1, "text": "new", "speaker": None}]
    store.update_transcript(job_id, segments=edited)
    assert store.get(job_id)["segments"][0]["text"] == "new"


def test_fail_records_error(store):
    job_id = store.create(filename="a.mp3")
    store.update_status(job_id, status="failed", error="ffmpeg blew up")
    job = store.get(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "ffmpeg blew up"


def test_delete_removes_job(store):
    job_id = store.create(filename="a.mp3")
    store.delete(job_id)
    assert store.get(job_id) is None
    assert not store.job_dir(job_id).exists()


def test_get_missing_returns_none(store):
    assert store.get("nope") is None
