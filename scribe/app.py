"""FastAPI web layer: REST endpoints + static frontend.

Jobs run in a ThreadPoolExecutor (Whisper is blocking and CPU/GPU bound). Progress
is reported by the pipeline into the job store; the browser polls GET /api/jobs/{id}.
``create_app`` is a factory so tests can inject a fake store/pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, exports
from . import pipeline as pipeline_mod
from .jobs import JobStore

STATIC_DIR = Path(__file__).parent.parent / "static"

ALLOWED_SUFFIXES = {
    ".mp3", ".mp4", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".oga", ".webm", ".mov",
}

_EXPORT_MEDIA = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "json": "application/json",
}


def _process_job(store: JobStore, settings, pipeline_run, job_id, audio_path,
                 denoise: bool, diarize: bool) -> None:
    job_settings = settings.model_copy(update={"denoise": denoise, "diarize": diarize})

    def progress(stage, pct):
        store.update_status(job_id, "running", stage=stage, progress=pct)

    try:
        result = pipeline_run(str(audio_path), job_settings, progress)
        store.save_result(job_id, result["segments"], result.get("summary"),
                          result.get("cleaned_text"))
    except Exception as exc:  # surface failures to the UI rather than a dead spinner
        store.update_status(job_id, "failed", error=str(exc))


class SettingsUpdate(BaseModel):
    llm_provider: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_model: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    whisper_model: Optional[str] = None
    denoise: Optional[bool] = None
    diarize: Optional[bool] = None
    hf_token: Optional[str] = None


_SECRET_FIELDS = ("anthropic_api_key", "openai_api_key", "hf_token")


def create_app(store: Optional[JobStore] = None, settings=None,
               pipeline_run=pipeline_mod.run, max_workers: int = 1,
               config_path=None) -> FastAPI:
    app = FastAPI(title="Scribe")

    app.state.config_path = config_path
    app.state.store = store or JobStore(
        db_path=config.DATA_DIR / "jobs.db", data_dir=config.DATA_DIR / "jobs")
    app.state.settings = settings or config.load(path=config_path)
    app.state.executor = ThreadPoolExecutor(max_workers=max_workers)
    app.state.pipeline_run = pipeline_run

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...),
                     denoise: Optional[bool] = Form(None),
                     diarize: Optional[bool] = Form(None)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"Unsupported file type: {suffix or 'unknown'}")

        s = app.state.settings
        use_denoise = s.denoise if denoise is None else denoise
        use_diarize = s.diarize if diarize is None else diarize

        store = app.state.store
        job_id = store.create(filename=file.filename, denoise=use_denoise, diarize=use_diarize)
        audio_path = store.audio_path(job_id, suffix)
        audio_path.write_bytes(await file.read())

        app.state.executor.submit(
            _process_job, store, s, app.state.pipeline_run, job_id, audio_path,
            use_denoise, use_diarize)
        return {"job_id": job_id}

    @app.get("/api/jobs")
    def list_jobs():
        return app.state.store.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = app.state.store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/transcript")
    def save_transcript(job_id: str, body: dict):
        if app.state.store.get(job_id) is None:
            raise HTTPException(404, "Job not found")
        app.state.store.update_transcript(job_id, body.get("segments", []))
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        if app.state.store.get(job_id) is None:
            raise HTTPException(404, "Job not found")
        app.state.store.delete(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/export")
    def export_job(job_id: str, format: str = "txt"):
        job = app.state.store.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if not job.get("segments"):
            raise HTTPException(400, "Job has no transcript yet")
        try:
            content = exports.render(job["segments"], format)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        media = _EXPORT_MEDIA.get(format.lower(), "text/plain")
        ext = "md" if format.lower() == "markdown" else format.lower()
        filename = f"{Path(job['filename']).stem}.{ext}"
        return PlainTextResponse(
            content, media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.get("/api/settings")
    def get_settings():
        return app.state.settings.public()

    @app.post("/api/settings")
    def update_settings(update: SettingsUpdate):
        current = app.state.settings.model_dump()
        for field, value in update.model_dump(exclude_none=True).items():
            # Don't overwrite a stored secret with an empty string from the UI.
            if field in _SECRET_FIELDS and value == "":
                continue
            current[field] = value
        new_settings = config.Settings(**current)
        config.save(new_settings, path=app.state.config_path)
        app.state.settings = new_settings
        return new_settings.public()

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


app = create_app()
