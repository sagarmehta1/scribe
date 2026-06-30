"""Download audio from a URL (YouTube, podcast page, direct media link) via yt-dlp.

We grab a single best-audio stream and save it as ``source.<ext>`` inside the job
folder, matching what a normal upload stores. That keeps the rest of the pipeline
(and Retry, which re-runs ``source.*``) identical for fetched and uploaded jobs.
No system ffmpeg is required because we take one stream as-is and let PyAV decode it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def download_audio(url: str, dest_dir) -> Tuple[Path, Optional[str]]:
    """Fetch the audio of ``url`` into ``dest_dir``; return (path, title)."""
    import yt_dlp

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        title = info.get("title")
    return path, title
