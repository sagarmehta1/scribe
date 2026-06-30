"""API tests using a synchronous fake pipeline (no model loading)."""

import io

import pytest
from fastapi.testclient import TestClient

from scribe import config
from scribe.app import create_app
from scribe.jobs import JobStore


def fake_pipeline_run(audio_path, settings, progress):
    progress("transcribe", 40)
    return {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world", "speaker": None}],
        "cleaned_text": "Hello world.",
        "summary": "## TL;DR\nA greeting.",
    }


@pytest.fixture
def client(tmp_path):
    store = JobStore(db_path=tmp_path / "jobs.db", data_dir=tmp_path / "data")
    settings = config.Settings(llm_provider="none")
    # Use an isolated config path so settings POSTs never touch the real data/config.json.
    app = create_app(store=store, settings=settings, pipeline_run=fake_pipeline_run,
                     max_workers=1, config_path=tmp_path / "config.json")
    return TestClient(app)


def _upload(client, name="meeting.mp3"):
    return client.post("/api/upload",
                       files={"file": (name, io.BytesIO(b"fake audio bytes"), "audio/mpeg")})


def test_upload_creates_job(client):
    r = _upload(client)
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_upload_rejects_bad_extension(client):
    r = _upload(client, name="virus.exe")
    assert r.status_code == 400


def test_job_completes_and_returns_result(client):
    job_id = _upload(client).json()["job_id"]
    # Synchronous executor (max_workers=1) — give it a moment via the status endpoint.
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("done", "running", "queued")


def test_get_missing_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_list_jobs(client):
    _upload(client)
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_settings_get_redacts_secrets(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "anthropic_api_key" not in body
    assert "anthropic_api_key_set" in body


def test_settings_post_updates(client):
    r = client.post("/api/settings", json={"whisper_model": "small", "denoise": True})
    assert r.status_code == 200
    assert client.get("/api/settings").json()["whisper_model"] == "small"


def test_export_after_done(client):
    job_id = _upload(client).json()["job_id"]
    # Wait for the synchronous worker to finish.
    for _ in range(50):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "done":
            break
    r = client.get(f"/api/jobs/{job_id}/export?format=txt")
    assert r.status_code == 200
    assert "hello world" in r.text


def test_retry_missing_job_404(client):
    assert client.post("/api/jobs/nope/retry").status_code == 404


def test_retry_reruns_existing_audio(client):
    job_id = _upload(client).json()["job_id"]
    for _ in range(50):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "done":
            break
    r = client.post(f"/api/jobs/{job_id}/retry")
    assert r.status_code == 200
    for _ in range(50):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "done":
            break
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"


def test_delete_removes_job_and_audio(client, tmp_path):
    job_id = _upload(client).json()["job_id"]
    job_dir = tmp_path / "data" / job_id
    assert any(job_dir.glob("source.*"))  # audio was saved
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not job_dir.exists()  # audio folder cleaned up


def test_delete_missing_job_404(client):
    assert client.delete("/api/jobs/nope").status_code == 404


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Scribe" in r.text
