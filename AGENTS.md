# ViriaRevive — Agent Instructions

## Entry points
- **CLI**: `python main.py <url>` — argparser in `main.py`
- **GUI (debug)**: `python app.py`
- **GUI (prod)**: `pythonw app.pyw` (no console, `--minimized` for tray start)

## Tests
```bash
python run_tests.py [-v]
```
No pytest — plain `unittest`. Tests requiring optional deps (numpy, yt-dlp) skip gracefully with `@skipIf`.

## Developer commands
| Action | Command |
|--------|---------|
| Install deps | `pip install -r requirements.txt` |
| Build exe | `build.bat` (release) / `build.bat debug` |
| Lint / typecheck | none configured |
| Windows startup | `setup_startup.bat` |

## Non-obvious structure
- **Flat package** (no `src/`). Imports work from repo root.
- **`api_bridge.py`** (1966 lines) is the GUI backend — all pipeline orchestration, state, settings, encryption. The frontend calls `pywebview.api.<method>()`.
- **`gui/`** is a vanilla HTML+CSS+JS SPA served by pywebview. No framework.
- **`subprocess_utils.py`** — global cancel infrastructure (`request_cancel()`, `reset_cancel()`) used across all modules. Pipe-draining threads prevent FFmpeg deadlocks.
- **`hwaccel.py`** — probes FFmpeg encoders/decoders once per process (`HwProfile` cached globally). `run_ffmpeg_with_encode_fallback()` retries with libx264 on HW failure.
- **`config.py`** — read at module import time. Gemini key is loaded from `tokens/gemini_key.json` (Fernet-encrypted via keyring).
- **`utils.py`** — shared utilities: `fmt_time()`, `wait_for_file_unlock()`.

## Key dependencies (not in requirements.txt)
- **FFmpeg/ffprobe** — must be in PATH, used heavily via subprocess
- **Ollama** — optional, for AI reranking and title gen. Default model: `qwen2.5:3b`

## Secrets (gitignored, must be manually placed)
| File | Purpose |
|------|---------|
| `client_secrets.json` | Google OAuth 2.0 Desktop app credentials |
| `tokens/*.json` | YouTube OAuth tokens (per account) |
| `tokens/gemini_key.json` | Encrypted Gemini API key |

## AI pipeline
- **Ollama** (local): `ollama_client.py` → `title_generator.py` / `ollama_detector.py`
- **Gemini** (cloud): `gemini_client.py` — alternative provider
- `AI_PROVIDER` in config controls which is used. Auto-fallback on failure.

## Platform
**Windows-only.** Code assumes:
- `CREATE_NO_WINDOW` flag for subprocesses (`subprocess_utils.py`)
- `C:/Windows/Fonts/` for font fallback
- VBS launchers for silent startup
- Windows Credential Manager via `keyring`

## Pipeline flow (CLI)
1. `download_video` (yt-dlp)
2. `find_viral_moments` (audio energy + scene changes)
3. `rerank_moments` (Ollama/Gemini — optionally)
4. For each clip: `extract_audio_clip` → `transcribe_clip` (Whisper) → `generate_subtitles` (ASS) → `extract_clip` (FFmpeg crop + burn subs)
5. `generate_titles_batch` (Ollama/Gemini/heuristic)
6. `upload_to_youtube` / `build_schedule`

## Gaming optimization
The project has been tuned for gaming videos:
- Detection weights: 30% audio / 20% variance / 50% scene changes
- Scene change threshold: 0.15 (more sensitive)
- Defaults: 8 clips, 25s duration, 10s min gap
- Category: Gaming (20), tags include `gaming`/`gameplay`
- Title prompts tuned for gaming slang (clutch, OP, insane)
- AI detector rejects: menu nav, inventory management, loading screens
- Subtitle style "game" preset available
- Effects presets include "streamer" and "hdr"

## Build artifacts
- PyInstaller spec: `viria.spec` — bundles `gui/` + ultralytics data
- Output: `dist/ViriaRevive/ViriaRevive.exe`
- Need to manually copy `music/`, `client_secrets.json` to dist folder
