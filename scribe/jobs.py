"""Job store: SQLite for metadata + a per-job folder on disk for audio and results.

A job moves through statuses: queued -> running -> done (or failed). Heavy result
data (the segment list) lives in ``<data_dir>/<job_id>/result.json`` rather than
the database, keeping the DB small and the transcript easy to inspect/edit on disk.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    stage       TEXT,
    progress    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    denoise     INTEGER NOT NULL DEFAULT 0,
    diarize     INTEGER NOT NULL DEFAULT 0,
    summary     TEXT,
    cleaned_text TEXT,
    created_at  TEXT NOT NULL
);
"""


class JobStore:
    def __init__(self, db_path, data_dir):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- paths --------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    def _result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result.json"

    def audio_path(self, job_id: str, suffix: str) -> Path:
        return self.job_dir(job_id) / f"source{suffix}"

    # -- lifecycle ----------------------------------------------------------
    def create(self, filename: str, denoise: bool = False, diarize: bool = False) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, filename, status, progress, denoise, diarize, created_at) "
                "VALUES (?, ?, 'queued', 0, ?, ?, ?)",
                (job_id, filename, int(denoise), int(diarize),
                 datetime.now(timezone.utc).isoformat()),
            )
        return job_id

    def update_status(self, job_id: str, status: str, stage: Optional[str] = None,
                      progress: Optional[int] = None, error: Optional[str] = None) -> None:
        sets = ["status = ?"]
        vals: list = [status]
        if stage is not None:
            sets.append("stage = ?"); vals.append(stage)
        if progress is not None:
            sets.append("progress = ?"); vals.append(progress)
        if error is not None:
            sets.append("error = ?"); vals.append(error)
        vals.append(job_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)

    def save_result(self, job_id: str, segments: List[dict], summary: Optional[str],
                    cleaned_text: Optional[str] = None) -> None:
        self._result_path(job_id).write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'done', stage = 'done', progress = 100, "
                "summary = ?, cleaned_text = ? WHERE id = ?",
                (summary, cleaned_text, job_id),
            )

    def update_transcript(self, job_id: str, segments: List[dict]) -> None:
        self._result_path(job_id).write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, job_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    # -- reads --------------------------------------------------------------
    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        job = {
            "id": row["id"],
            "filename": row["filename"],
            "status": row["status"],
            "stage": row["stage"],
            "progress": row["progress"],
            "error": row["error"],
            "denoise": bool(row["denoise"]),
            "diarize": bool(row["diarize"]),
            "summary": row["summary"],
            "cleaned_text": row["cleaned_text"],
            "created_at": row["created_at"],
            "segments": None,
        }
        result_path = self._result_path(row["id"])
        if result_path.exists():
            job["segments"] = json.loads(result_path.read_text(encoding="utf-8"))
        return job

    def get(self, job_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]
