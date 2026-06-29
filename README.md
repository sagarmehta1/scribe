# Scribe

A free, local, self-hosted alternative to Otter.ai. Drop in an audio or video
file and get back a clean, optionally speaker-labeled transcript plus an AI
summary — with **no subscription** and, by default, **nothing leaving your machine**.

- 🎙️ **Transcribe** mp3 / mp4 / m4a / wav / and more (powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- 🔇 **Optional denoising** to clean up noisy recordings before transcribing
- 🗣️ **Optional speaker labels** (diarization via pyannote.audio)
- ✨ **Transcript cleanup + AI summary** (TL;DR / key points / action items)
- 💾 **Export** to Markdown, plain text, SRT, VTT, or JSON
- 🔒 **Private by default** — runs entirely on your computer; no system ffmpeg required (PyAV bundles the codecs)

No GPU required (it runs on CPU; a GPU just makes it faster).

## Quick start

```bash
git clone <your-fork-url> scribe
cd scribe
python -m venv .venv && . .venv/Scripts/activate    # Windows
# python -m venv .venv && source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
uvicorn scribe.app:app
```

Then open http://127.0.0.1:8000, drag in a file, and watch the transcript appear.
The first transcription downloads the Whisper model (a few hundred MB) — that's a
one-time cost.

## Configuration

Everything works out of the box with zero config. To customize, copy
`.env.example` to `.env` and edit it, or use the in-app **Settings** panel
(stored in `data/config.json`).

### Whisper model size

`SCRIBE_WHISPER_MODEL` — `tiny` | `base` (default) | `small` | `medium` | `large-v3`.
Bigger = more accurate but slower. `base` is a good CPU default.

### AI summary + cleanup (optional)

Pick a provider in Settings (or `SCRIBE_LLM_PROVIDER`):

- `none` — regex-based cleanup only (removes filler words, fixes punctuation). Fully free, no setup.
- `ollama` — free, local. Install [Ollama](https://ollama.com), run `ollama pull llama3.1`.
- `anthropic` — Claude. Paste an API key in Settings.
- `openai` — paste an API key in Settings.

### Speaker labels (optional)

Diarization needs an extra dependency and a free HuggingFace token:

```bash
pip install pyannote.audio
```

Accept the model terms at https://hf.co/pyannote/speaker-diarization-3.1, create a
token at https://hf.co/settings/tokens, and set `SCRIBE_HF_TOKEN` (or paste it in
Settings). Then toggle "Detect speakers" on upload. If it's not configured, jobs
simply run without speaker labels.

## Development

```bash
pip install -r requirements.txt
pytest                    # 61 tests, no network/model needed (LLM + Whisper mocked)
```

Architecture: a reusable, independently-tested pipeline (`scribe/`) under a thin
FastAPI layer (`scribe/app.py`) that serves a vanilla-JS frontend (`static/`).

| Module | Responsibility |
|--------|----------------|
| `audio.py` | Decode any file to 16 kHz mono (PyAV) + optional denoise |
| `transcribe.py` | faster-whisper wrapper → timestamped segments |
| `diarize.py` | pyannote turns + overlap-based speaker assignment |
| `llm.py` | Ollama / Anthropic / OpenAI providers + regex fallback |
| `exports.py` | Render segments to md / txt / srt / vtt / json |
| `jobs.py` | SQLite metadata + per-job on-disk results |
| `pipeline.py` | Stage orchestration with progress callbacks |
| `app.py` | REST endpoints + static frontend |

## Troubleshooting

**Model download fails / TLS errors.** Some corporate or antivirus setups
intercept TLS and break Python's in-process HTTPS. As a workaround, download the
model files yourself (e.g. with `curl`) into a folder and point
`SCRIBE_WHISPER_MODEL` at that folder — faster-whisper accepts a local path. The
files live under `https://huggingface.co/Systran/faster-whisper-<size>/`.

## License

MIT — see [LICENSE](LICENSE).
