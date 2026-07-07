"""
ApiBridge  –  Python ↔ JavaScript bridge for the ViriaRevive GUI.

Every public method is exposed to the frontend as  pywebview.api.<method>().
Long-running work runs on a daemon thread; progress is pushed back to
the UI via  window.evaluate_js()  which calls global JS callback functions.
"""

import functools
import http.server
import json
import os
import re
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import yt_dlp
import logging

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

try:
    import keyring
except ImportError:
    keyring = None

import config
from config import (
    BASE_DIR,
    AI_DETECTOR_MODE,
    AI_PROVIDER,
    CLIENT_SECRETS_FILE,
    GEMINI_TOKEN_FILE,
    GEMINI_API_KEY,
    CLIPS_DIR,
    CLIP_DURATION,
    CROP_VERTICAL,
    DOWNLOADS_DIR,
    FFMPEG_PRESET,
    MIN_GAP,
    MUSIC_DIR,
    SENTENCE_BUFFER,
    NUM_CLIPS,
    OLLAMA_DETECTOR_CANDIDATE_MULTIPLIER,
    OLLAMA_DETECTOR_MODEL,
    OLLAMA_DETECTOR_TIMEOUT,
    SUBTITLE_STYLE,
    SUBTITLES_DIR,
    VIDEO_CRF,
    VIDEO_ENCODER,
    VIDEO_DECODER,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    YOLO_DEVICE,
)
import gemini_client
from hwaccel import log_hardware_startup, probe_ffmpeg, get_hardware_summary, video_encode_args, get_gpu_count, select_least_loaded_gpu
from detector import find_viral_moments
from ollama_detector import detector_ready, rerank_moments
from transcriber import transcribe_clip, find_sentence_boundary
from subtitler import generate_subtitles, get_available_styles
from clipper import (
    extract_clip, extract_audio_clip,
    get_effects_list,
    validate_shorts_output,
)
from cropper import get_crop_params, get_crop_params_dynamic, get_dimensions, detect_all_persons
from subprocess_utils import CancelledError
from title_generator import generate_title, generate_titles_batch, list_ollama_models, ensure_model
from uploader import (
    upload_to_youtube,
    build_schedule,
    get_youtube_service,
    is_connected,
    disconnect,
    list_channels,
    list_categories,
    add_account,
    list_accounts,
    DEFAULT_CATEGORIES,
)

logger = logging.getLogger("ViriaRevive")

STATE_FILE = BASE_DIR / "viria_state.json"

# Local video input (GUI file picker) — must match frontend allowed extensions
ALLOWED_INPUT_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm"}


# ── Log interceptor — captures print() and forwards to the GUI console ───────

import sys as _sys


class _LogTee:
    """Wraps stdout/stderr: writes to both the original stream and a callback."""

    def __init__(self, original, callback):
        self._orig = original
        self._cb = callback
        self._encoding = getattr(original, 'encoding', 'utf-8') # type: ignore

    def write(self, text):
        try:
            self._orig.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Windows console can't handle some Unicode chars — strip them
            safe = text.encode('ascii', errors='xmlcharrefreplace').decode('ascii')
            try:
                self._orig.write(safe)
            except Exception as e:
                logger.debug("Failed to write sanitized text to console: %s", e)
        if text and text.strip():
            try:
                self._cb(text.strip())
            except Exception as e:
                logger.debug("Log callback failed: %s", e)
        return len(text)

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


_log_bridge = None  # set by ApiBridge.__init__


_log_push_queue = queue.Queue()
_forwarding = threading.local()

def _install_log_tee(debug=False):
    """Configure Python's logging to push messages to the frontend console."""
    level = logging.DEBUG if debug else logging.INFO

    def _forward(text):
        if not text:
            return
        # Guard against recursion (if the bridge pusher itself triggers a print)
        if getattr(_forwarding, 'active', False):
            return
        _log_push_queue.put(text)

    # Create a custom handler that uses our _forward function
    class ConsoleHandler(logging.Handler):
        def emit(self, record):
            _forward(self.format(record))

    # Configure the logger
    root = logging.getLogger()
    # Remove existing handlers to avoid duplicates on re-init
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(level)
    root.addHandler(ConsoleHandler())
    
    # Suppress noisy third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Also redirect stdout/stderr to the logger for non-logging prints
    _sys.stdout = _LogTee(_sys.__stdout__ or _sys.stdout, _forward) # type: ignore
    _sys.stderr = _LogTee(_sys.__stderr__ or _sys.stderr, _forward) # type: ignore

def _log_pusher_thread():
    """Batches log messages and pushes them to JS at a controlled rate to avoid UI flooding."""
    while True:
        try:
            # Wait for the first log message
            first = _log_push_queue.get()
            lines = [first]
            
            # Collect more messages for a brief period to batch them
            start_batch = time.time()
            while time.time() - start_batch < 0.15 and len(lines) < 100:
                try:
                    lines.append(_log_push_queue.get_nowait())
                except queue.Empty:
                    break
            
            if _log_bridge:
                text = "\n".join(lines)
                safe_text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
                debug_val = 1 if _log_bridge._user_settings.get("debug_logging", False) else 0
                
                _forwarding.active = True
                try:
                    _log_bridge._js(f"window.onConsoleLog({json.dumps(safe_text)}, {debug_val})")
                finally:
                    _forwarding.active = False
        except Exception as e:
            logger.debug("Log pusher error: %s", e)
            time.sleep(0.5)

threading.Thread(target=_log_pusher_thread, daemon=True).start()

# ── Local video server (serves clip files for HTML5 <video> preview) ─────────

class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files from a directory with CORS headers, no logging."""

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass  # Browser closed connection early — harmless


class _SilentHTTPServer(http.server.HTTPServer):
    """HTTPServer that suppresses broken-pipe / connection-reset tracebacks."""

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError, OSError)):
            return  # browser closed connection early — harmless
        super().handle_error(request, client_address)


def _start_video_server(clips_dir: Path):
    """Start a local HTTP server for video previews; returns (port, server)."""
    handler = functools.partial(_SilentHandler, directory=str(clips_dir))
    server = _SilentHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[+] Video preview server on http://127.0.0.1:{port}")
    return port, server


class ApiBridge:
    def __init__(self):
        self._window = None
        self._processing = False
        self._cancel = False
        self._cancel_lock = threading.Lock()
        self._pipeline_queue = queue.Queue()
        self._worker_thread = None
        self._worker_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._scheduled_lock = threading.Lock()
        self._active_item_index = None

        self._results: list[Path] = []
        self._moments: list[dict] = []
        self._scheduled: list[dict] = []
        self._scheduler_running = False
        self._delete_after_upload = False   # auto-delete clips after YouTube upload
        self._user_settings: dict = {}      # user settings persisted to disk
        self._pending_js: list[str] = []    # JS calls queued while window was hidden
        self._uploading_indices: set = set()  # prevent double-upload/delete

        # Install log interceptor so print() output goes to the GUI console
        global _log_bridge
        _log_bridge = self

        # Load persisted state from previous session (before video server init)
        self._load_state()

        # Clips directory: use saved setting or fall back to config default
        saved_clips_path = self._user_settings.get("clips_path")
        if saved_clips_path:
            self._clips_dir = Path(saved_clips_path)
            self._clips_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._clips_dir = CLIPS_DIR

        self._video_port, self._video_server = _start_video_server(self._clips_dir)
        self._music_port, self._music_server = _start_video_server(MUSIC_DIR)

        # Initialize logging based on saved setting
        is_debug = self._user_settings.get("debug_logging", False)
        _install_log_tee(debug=is_debug)

        # Perform hardware startup check in background so the UI doesn't hang on launch
        threading.Thread(target=log_hardware_startup, kwargs={
            "encoder_pref": self._user_settings.get("video_encoder", VIDEO_ENCODER),
            "yolo_device": self._user_settings.get("yolo_device", YOLO_DEVICE),
            "whisper_device": self._user_settings.get("whisper_device", WHISPER_DEVICE),
        }, daemon=True).start()

    # ── Exposed: config / deps ───────────────────────────────────────────

    def run_diagnostics(self):
        """Execute a suite of diagnostic tests to verify the software environment."""
        logger.info("[diag] Starting system diagnostics...")
        diag = {
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "ffprobe": shutil.which("ffprobe") is not None,
            "storage": {},
            "hardware": get_hardware_summary(
                self._user_settings.get("video_encoder", VIDEO_ENCODER),
                self._user_settings.get("yolo_device", YOLO_DEVICE),
                self._user_settings.get("whisper_device", WHISPER_DEVICE)
            ),
            "ollama": detector_ready(OLLAMA_DETECTOR_MODEL)
        }
        for name, path in [("clips", self._clips_dir), ("temp", Path(tempfile.gettempdir()))]:
            try:
                test_file = path / f".viria_test_{int(time.time())}"
                test_file.write_text("ok")
                test_file.unlink()
                diag["storage"][name] = "writable"
            except Exception as e:
                diag["storage"][name] = f"error: {str(e)}"
        return diag

    def test_ffmpeg_encoding(self):
        """Verify if the selected FFmpeg encoder is functional with current settings."""
        preset = self._user_settings.get("ffmpeg_preset", FFMPEG_PRESET)
        crf = str(self._user_settings.get("video_crf", VIDEO_CRF))
        encoder = self._user_settings.get("video_encoder", VIDEO_ENCODER)
        test_out = self._clips_dir / f"test_enc_{int(time.time())}.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=1280x720:rate=30",
            *video_encode_args(preset, crf, encoder),
            str(test_out)
        ]
        from subprocess_utils import run as _srun
        r = _srun(cmd, capture_output=True, text=True)
        success = r.returncode == 0 and test_out.exists()
        if test_out.exists(): test_out.unlink()
        return {"success": success, "encoder": encoder, "error": r.stderr if not success else None}

    def _get_cipher(self):
        """Initialize or retrieve the encryption cipher from Windows Credential Manager."""
        if not Fernet:
            return None

        service_name = "ViriaRevive"
        account_name = "MasterEncryptionKey"

        if keyring:
            try:
                stored_key = keyring.get_password(service_name, account_name)
                if not stored_key:
                    new_key = Fernet.generate_key().decode()
                    keyring.set_password(service_name, account_name, new_key)
                    stored_key = new_key
                return Fernet(stored_key.encode()) # type: ignore
            except Exception as e:
                print(f"[security] Keyring access failed: {e}. Storing secrets without local crypt key.")

        return None

    def _encrypt(self, text):
        cipher = self._get_cipher()
        if not text or not cipher: return text
        return cipher.encrypt(text.encode()).decode()

    def _decrypt(self, text):
        cipher = self._get_cipher()
        if not text or not cipher: return text
        try: return cipher.decrypt(text.encode()).decode()
        except Exception as e:
            logger.error("Failed to decrypt value: %s", e)
            return text

    def get_settings(self):
        """Return user settings (persisted overrides merged with defaults)."""
        defaults = {
            "num_clips": NUM_CLIPS,
            "clip_duration": CLIP_DURATION,
            "sentence_buffer": SENTENCE_BUFFER,
            "min_gap": MIN_GAP,
            "whisper_model": WHISPER_MODEL,
            "whisper_language": WHISPER_LANGUAGE or "",
            "subtitle_style": SUBTITLE_STYLE,
            "ffmpeg_preset": FFMPEG_PRESET,
            "video_crf": VIDEO_CRF,
            "video_encoder": VIDEO_ENCODER,
            "video_decoder": VIDEO_DECODER,
            "yolo_device": YOLO_DEVICE,
            "whisper_device": WHISPER_DEVICE,
            "crop_vertical": True,
            "ai_detector": AI_DETECTOR_MODE,
            "upload_category": "20",
            "upload_privacy": "public",
            "upload_region": "US",
            "upload_tags": "shorts, gaming, gameplay, clips",
            "upload_description": "#shorts #gaming #gameplay",
            "debug_logging": False,
            "clips_path": str(CLIPS_DIR),
        }
        # Merge saved user overrides (from save_settings)
        if self._user_settings:
            defaults.update(self._user_settings)
        # Ensure the Gemini key is included if it exists in the token file
        gemini_key = self._decrypt(config.GEMINI_API_KEY)
        defaults["gemini_key_configured"] = gemini_client.is_available(gemini_key)
        defaults["gemini_key_hint"] = f"...{gemini_key[-4:]}" if gemini_key and len(gemini_key) >= 4 else ""
        defaults["crop_vertical"] = True
        return defaults

    @staticmethod
    def _validate_youtube_credentials(data: dict) -> bool:
        for section in ("installed", "web"):
            creds = data.get(section)
            if isinstance(creds, dict) and creds.get("client_id") and creds.get("client_secret"):
                return True
        return False

    def get_credentials_status(self):
        """Return whether YouTube OAuth and Gemini credentials are configured."""
        yt_ok = False
        if CLIENT_SECRETS_FILE.exists():
            try:
                data = json.loads(CLIENT_SECRETS_FILE.read_text(encoding="utf-8"))
                yt_ok = self._validate_youtube_credentials(data)
            except Exception:
                yt_ok = False

        gemini_key = self._decrypt(config.GEMINI_API_KEY)
        gemini_ok = gemini_client.is_available(gemini_key)
        return {
            "youtube_credentials": yt_ok,
            "gemini_configured": gemini_ok,
            "gemini_key_hint": f"...{gemini_key[-4:]}" if gemini_ok and len(gemini_key) >= 4 else "",
        }

    def upload_youtube_credentials(self):
        """Pick a Google OAuth JSON file and save it as client_secrets.json."""
        import webview

        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("JSON files (*.json)", "All files (*.*)"),
        )
        if not result:
            return {"ok": False, "cancelled": True}

        try:
            data = json.loads(Path(result[0]).read_text(encoding="utf-8"))
            if not self._validate_youtube_credentials(data):
                return {
                    "ok": False,
                    "error": "Invalid OAuth credentials. Expected Google Desktop app JSON with client_id and client_secret.",
                }
            data.pop("gemini_api_key", None)
            CLIENT_SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return {"ok": True}
        except json.JSONDecodeError:
            return {"ok": False, "error": "File is not valid JSON."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def remove_youtube_credentials(self):
        """Delete the saved YouTube OAuth credentials file."""
        if CLIENT_SECRETS_FILE.exists():
            CLIENT_SECRETS_FILE.unlink()
        return {"ok": True}

    def test_gemini_key(self, api_key=None):
        """Test the saved or provided Gemini API key."""
        key = (api_key or "").strip() or self._decrypt(config.GEMINI_API_KEY)
        return gemini_client.test_connection(key)

    def clear_gemini_key(self):
        """Remove the saved Gemini API key."""
        if GEMINI_TOKEN_FILE.exists():
            GEMINI_TOKEN_FILE.unlink()
        config.GEMINI_API_KEY = ""
        return {"ok": True}

    def save_settings(self, settings):
        """Persist user settings to disk so they survive restarts."""
        new_settings = (settings or {}).copy()

        # Extract Gemini key and save to tokens/gemini_key.json instead of viria_state.json
        gemini_key = new_settings.pop("gemini_api_key", None)
        if gemini_key is not None and str(gemini_key).strip():
            try:
                secrets = {}
                if GEMINI_TOKEN_FILE.exists():
                    with open(GEMINI_TOKEN_FILE, 'r', encoding='utf-8') as f:
                        secrets = json.load(f)
                secrets["gemini_api_key"] = self._encrypt(gemini_key.strip())
                with open(GEMINI_TOKEN_FILE, 'w', encoding='utf-8') as f:
                    json.dump(secrets, f, indent=2)

                # Update runtime value so changes take effect without restart
                config.GEMINI_API_KEY = gemini_key.strip()
            except Exception as e:
                logger.exception("Failed to write Gemini key to tokens file")

        # Check if debug mode changed and re-install logging tee
        if new_settings.get("debug_logging") != self._user_settings.get("debug_logging"):
            _install_log_tee(debug=new_settings.get("debug_logging", False))

        # Handle clips path change: restart video server for new directory
        new_clips_path = new_settings.get("clips_path")
        if new_clips_path and str(Path(new_clips_path).resolve()) != str(self._clips_dir.resolve()):
            self._restart_video_server(Path(new_clips_path))

        self._user_settings = new_settings
        self._user_settings["crop_vertical"] = True
        self._save_state()
        return {"ok": True}

    def check_dependencies(self):
        has_ffmpeg = shutil.which("ffmpeg") is not None
        has_ffprobe = shutil.which("ffprobe") is not None
        info = {"ffmpeg": has_ffmpeg, "ffprobe": has_ffprobe, "audio_analysis": has_ffmpeg and has_ffprobe}
        if has_ffmpeg:
            enc_pref = self._user_settings.get("video_encoder", VIDEO_ENCODER)
            prof = probe_ffmpeg(enc_pref)
            info["video_encoder"] = prof.active_encoder
            info["video_encoder_label"] = prof.active_encoder_label
            info["hwaccel_decode"] = prof.active_hwaccel
        return info

    def set_delete_after_upload(self, enabled):
        """Toggle auto-delete clips from disk after successful YouTube upload."""
        self._delete_after_upload = bool(enabled)
        return {"ok": True, "enabled": self._delete_after_upload}

    def get_delete_after_upload(self):
        return {"enabled": self._delete_after_upload}

    # ── Exposed: AI title generation ──────────────────────────────────────

    def generate_titles(self):
        """Generate titles for all clips using LLM (or heuristic fallback).

        If transcripts are missing (e.g. clips from a previous session where
        moments were lost), auto-transcribe the clip audio first.
        """
        from title_generator import ensure_model, DEFAULT_MODEL

        num_clips = len(self._results)
        # Sync moments to match results count exactly
        if len(self._moments) > num_clips:
            self._moments = self._moments[:num_clips]
        while len(self._moments) < num_clips:
            self._moments.append({})

        # Backfill any clips missing transcripts
        # If using Gemini, we only need to backfill if we want titles right now
        results_count = len(self._results)
        if num_clips > results_count:
            num_clips = results_count
        missing = [i for i in range(results_count)
                   if not self._moments[i].get("transcript")]
        if missing:
            for i in missing:
                self._backfill_transcript_single(i)
            self._save_state()

        transcripts = [m.get("transcript", "") for m in self._moments]
        if not any(transcripts):
            return {"titles": [], "error": "No transcripts available — process clips first"}

        titles, llm_available = self._batch_generate_titles(transcripts)
        return {"titles": titles, "llm": llm_available}

    def generate_title_for_clip(self, clip_index):
        """Generate a title for a single clip."""
        # Ensure moments list matches results length
        while len(self._moments) < len(self._results):
            self._moments.append({})

        if clip_index < 0 or clip_index >= len(self._moments):
            return {"title": "", "error": "Invalid clip index"}

        transcript = self._moments[clip_index].get("transcript", "")

        # If no transcript, try to transcribe from the clip file
        if not transcript and clip_index < len(self._results):
            self._backfill_transcript_single(clip_index)
            transcript = self._moments[clip_index].get("transcript", "")

        if not transcript:
            return {"title": "", "error": "No transcript for this clip"}
        title = self._generate_single_title(transcript)
        return {"title": title}

    def rename_clip(self, clip_index, new_title):
        """Rename a clip file on disk to match a new title.

        Returns the new filename, or error.
        """
        if clip_index < 0 or clip_index >= len(self._results):
            return {"error": "Invalid clip index"}
        old_path = self._results[clip_index]
        if not old_path.exists():
            return {"error": "File not found"}

        # Remove emojis and non-ASCII chars that cause issues on Windows
        safe = re.sub(r'[^\x20-\x7E]', '', new_title)
        safe = re.sub(r'[<>:"/\\|?*]', '', safe)
        safe = safe.strip('. ')[:80]
        if not safe:
            return {"error": "Title too short after sanitization"}

        ext = old_path.suffix
        new_name = f"{safe}{ext}"
        new_path = old_path.parent / new_name

        # Avoid collisions
        if new_path.exists() and new_path != old_path:
            counter = 2
            while new_path.exists():
                new_name = f"{safe} ({counter}){ext}"
                new_path = old_path.parent / new_name
                counter += 1

        try:
            old_path.rename(new_path)
            self._results[clip_index] = new_path
            self._save_state() # type: ignore
            print(f"[rename] {old_path.name} → {new_path.name}")
            return {"filename": new_path.name, "path": str(new_path)}
        except Exception as e:
            return {"error": str(e)}

    def generate_and_rename_all(self):
        """Generate AI titles for all clips in a background thread.

        Returns immediately with {"ok": True}. Progress and results are
        pushed to the frontend via window.onTitleProgress and
        window.onTitlesDone callbacks.
        """
        threading.Thread(target=self._run_title_gen, daemon=True).start()
        return {"ok": True}
    
    def generate_and_rename_indices(self, indices):
        """Generate AI titles only for specific clip indices (e.g. a folder).

        Returns immediately with {"ok": True}. Progress and results are
        pushed to the frontend via window.onTitleProgress and
        window.onTitlesDone callbacks.
        """
        threading.Thread(target=self._run_title_gen, args=(indices,), daemon=True).start() # type: ignore
        return {"ok": True}

    def open_devtools(self):
        """Request the frontend to open developer tools."""
        self._js("console.log('[bridge] Developer tools requested via UI')")
        return {"ok": True}

    def _run_title_gen(self, only_indices=None):
        """Background thread: generate titles, rename files, push results to JS.

        If only_indices is provided (list of ints), only those clip indices
        are transcribed and titled. Otherwise all clips are processed.
        """
        try:
            num_clips = len(self._results) # type: ignore
            print(f"[title-gen] {num_clips} clips, {len(self._moments)} moments in state")

            # Trim moments to match results (moments can accumulate beyond results
            # if clips were deleted or state got out of sync)
            if len(self._moments) > num_clips:
                self._moments = self._moments[:num_clips]
            # Pad if fewer
            while len(self._moments) < num_clips:
                self._moments.append({})

            # Determine which indices to process
            target_indices = only_indices if only_indices is not None else list(range(num_clips))
            # Filter to valid range
            target_indices = [i for i in target_indices if 0 <= i < num_clips] # type: ignore
            if not target_indices:
                self._js("window.onTitlesDone && window.onTitlesDone({error: 'No valid clips to process'})")
                return

            print(f"[title-gen] Processing {len(target_indices)} of {num_clips} clips")

            # Backfill any target clips missing transcripts
            missing = [i for i in target_indices # type: ignore
                       if not self._moments[i].get("transcript")] # type: ignore
            if missing:
                print(f"[title-gen] {len(missing)} clips missing transcripts, backfilling...")
                for idx, i in enumerate(missing):
                    self._js(f"window.onTitleProgress && window.onTitleProgress({idx}, {len(missing)}, 'Transcribing clip {i+1}...')") # type: ignore
                    self._backfill_transcript_single(i)
                self._save_state() # type: ignore

            # Build transcripts list — only for target indices, empty for others
            transcripts = [""] * num_clips
            for i in target_indices:
                transcripts[i] = self._moments[i].get("transcript", "")
            if not any(transcripts[i] for i in target_indices):
                self._js("window.onTitlesDone && window.onTitlesDone({error: 'No transcripts available'})")
                return

            for i in target_indices:
                p = self._results[i] # type: ignore
                if i < len(self._moments) and not self._moments[i].get("source_stem"):
                    m = re.match(r'^(.+?)_viral\d+', p.name)
                    self._moments[i]["source_stem"] = m.group(1) if m else p.stem

            def _on_progress(done, total, title):
                self._js(f"window.onTitleProgress && window.onTitleProgress({done}, {total}, {json.dumps(title or '')})")

            titles, llm_available = self._batch_generate_titles(transcripts, on_progress=_on_progress)

            renamed = 0 # type: ignore
            results = []
            for i in target_indices:
                title = titles[i] if i < len(titles) else ""
                if not title:
                    results.append({"index": i, "title": "", "renamed": False})
                    continue
                r = self.rename_clip(i, title)
                ok = "filename" in r
                if ok: # type: ignore
                    renamed += 1
                results.append({
                    "index": i,
                    "title": title,
                    "renamed": ok,
                    "filename": r.get("filename", self._results[i].name if i < len(self._results) else ""),
                }) # type: ignore

            self._save_state()

            # Push results to frontend
            payload = json.dumps({"titles": results, "renamed": renamed, "llm": llm_available, "total": len(target_indices)})
            self._js(f"window.onTitlesDone && window.onTitlesDone({payload})")

        except Exception as e:
            logger.exception("Error in title generation")
            self._js(f"window.onTitlesDone && window.onTitlesDone({{error: {json.dumps('An internal error occurred during title generation.')}}})")

    def _backfill_transcripts(self): # type: ignore
        """Transcribe clips that are missing transcripts (e.g. from previous sessions)."""
        print("[title-gen] Backfilling missing transcripts from clip audio...")
        for i, p in enumerate(self._results): # type: ignore
            if i < len(self._moments) and self._moments[i].get("transcript"):
                continue  # already has transcript
            self._backfill_transcript_single(i)
        self._save_state()  # persist backfilled transcripts

    def _backfill_transcript_single(self, clip_index):
        """Transcribe a single clip to fill in its transcript."""
        import tempfile
        if clip_index >= len(self._results): # type: ignore
            return
        p = self._results[clip_index]
        if not p.exists():
            return

        # Ensure moments slot exists
        while len(self._moments) <= clip_index:
            self._moments.append({})
        
        try:
            wav = Path(tempfile.gettempdir()) / f"viria_backfill_{clip_index}.wav"
            extract_audio_clip(p, 0, 60, wav)  # max 60s
            try:
                st = wav.stat()
            except OSError:
                st = None
            if st and st.st_size > 1000: # type: ignore
                words = transcribe_clip(
                    wav, model_size=WHISPER_MODEL, language=None, device_pref=WHISPER_DEVICE,
                ) # type: ignore
                transcript = " ".join(w.get("text", "") for w in words).strip()
                if transcript:
                    self._moments[clip_index]["transcript"] = transcript
                    print(f"  [+] Clip {clip_index + 1}: {len(transcript)} chars transcribed")
            try:
                wav.unlink(missing_ok=True)
            except Exception:
                pass # type: ignore
        except Exception as e:
            print(f"  [!] Backfill failed for clip {clip_index + 1}: {e}")

    def _batch_generate_titles(self, transcripts, on_progress=None):
        """Helper to route batch title generation to the selected AI provider."""
        ai_provider = self._user_settings.get("ai_provider", AI_PROVIDER)
        gemini_key = self._decrypt(config.GEMINI_API_KEY)
        use_gemini = ai_provider == "gemini" and gemini_client.is_available(gemini_key)
        lang = self._user_settings.get("title_language") or self._user_settings.get("whisper_language")

        if use_gemini:
            titles = gemini_client.generate_titles_batch(
                transcripts, gemini_key, language=lang, on_progress=on_progress
            ) # type: ignore
            return titles, True
        else:
            model = self._user_settings.get("ollama_detector_model", OLLAMA_DETECTOR_MODEL)
            llm_ready = ensure_model(model)
            titles = generate_titles_batch(
                transcripts, model=model, language=lang, on_progress=on_progress
            )
            return titles, llm_ready

    def _generate_single_title(self, transcript):
        """Helper to route single title generation to the selected AI provider."""
        ai_provider = self._user_settings.get("ai_provider", AI_PROVIDER)
        gemini_key = self._decrypt(config.GEMINI_API_KEY)
        use_gemini = ai_provider == "gemini" and gemini_client.is_available(gemini_key)
        lang = self._user_settings.get("title_language") or self._user_settings.get("whisper_language")

        if use_gemini:
            prompt = (
                f"Create a viral gaming YouTube Short title in {lang or 'English'} for this gameplay clip. "
                f"Make it hype and gamer-friendly. Use gaming slang if appropriate "
                f"(clutch, OP, insane, wipeout, GG). Keep it under 50 chars.\n\n"
                f"Transcript: {transcript}"
            )
            return gemini_client.generate(prompt, gemini_key)
        else:
            model = self._user_settings.get("ollama_detector_model", OLLAMA_DETECTOR_MODEL)
            return generate_title(transcript, model=model, language=lang)

    def get_ollama_models(self):
        """Return available Ollama models for title generation."""
        models = list_ollama_models()
        return {"models": models, "available": len(models) > 0}

    def ensure_ollama_model(self, model=None):
        """Ensure the title generation model is downloaded. Auto-pulls if needed."""
        from title_generator import DEFAULT_MODEL
        model = model or DEFAULT_MODEL
        ready = ensure_model(model)
        return {"ready": ready, "model": model}


    # ── Exposed: YouTube connection ───────────────────────────────────────

    def connect_youtube(self):
        """Add a YouTube account via OAuth flow. Supports multiple accounts."""
        try:
            result = add_account()
            return {"ok": True, "account": result}
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"Connection failed: {e}"}

    def add_youtube_account(self):
        """Alias for connect_youtube — adds another account."""
        return self.connect_youtube()

    def disconnect_youtube(self, account_id=None):
        """Disconnect a specific account, or all accounts if no ID given."""
        disconnect(account_id)
        return {"ok": True}

    def youtube_status(self):
        return {"connected": is_connected(), "accounts": list_accounts()}

    def get_channels(self):
        try:
            channels = list_channels()
            accounts = list_accounts()
            return {"channels": channels, "accounts": accounts, "account_count": len(accounts), "error": None}
        except Exception as e:
            accounts = list_accounts()
            return {"channels": [], "accounts": accounts, "account_count": len(accounts), "error": str(e)}

    def get_categories(self):
        try:
            cats = list_categories()
            return {"categories": cats if cats else DEFAULT_CATEGORIES}
        except Exception as e:
            return {"error": str(e), "categories": DEFAULT_CATEGORIES}

    def get_subtitle_styles(self):
        """Return available subtitle styles for the UI picker."""
        return {"styles": get_available_styles()}

    def get_effects(self):
        """Return available video effect presets."""
        return {"effects": get_effects_list()}

    def list_music(self):
        """List audio files in the music/ folder."""
        tracks = []
        if MUSIC_DIR.exists():
            for p in sorted(MUSIC_DIR.iterdir()):
                if p.suffix.lower() in ('.mp3', '.wav', '.aac', '.ogg', '.m4a', '.flac'):
                    tracks.append({
                        "filename": p.name,
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    })
        return {"tracks": tracks, "music_dir": str(MUSIC_DIR)}

    def get_music_url(self, filename):
        """Return a local HTTP URL for a music file so the browser can play it."""
        music_path = MUSIC_DIR / filename
        if music_path.exists():
            return {"url": f"http://127.0.0.1:{self._music_port}/{filename}"}
        return {"url": None}

    def open_music_folder(self):
        """Open the music folder in system explorer."""
        MUSIC_DIR.mkdir(exist_ok=True)
        try:
            os.startfile(str(MUSIC_DIR))
        except Exception as e:
            logger.warning("Failed to open music folder: %s", e)
        return {"ok": True}

    def get_music_waveform(self, filename):
        """Generate waveform data + duration for a music file.

        Returns {peaks: [...], duration: float} where peaks is ~200 normalized
        amplitude values (0.0-1.0) representing the waveform shape.
        """
        from subprocess_utils import run as _run
        music_path = MUSIC_DIR / filename
        if not music_path.exists():
            return {"error": "File not found", "peaks": [], "duration": 0}

        try:
            # Get duration
            dr = _run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(music_path)],
                capture_output=True, text=True, timeout=15,
            )
            duration = float(dr.stdout.strip())

            # Extract raw PCM samples at low sample rate for waveform
            # 200 peaks over the full duration → sample_rate ~ 200/duration
            num_peaks = 200
            sample_rate = max(100, int(num_peaks / max(duration, 0.1)))

            pr = _run(
                ["ffmpeg", "-y", "-i", str(music_path),
                 "-ac", "1",  # mono
                 "-ar", str(sample_rate),  # low sample rate
                 "-f", "s16le",  # raw 16-bit PCM
                 "-"],
                capture_output=True, timeout=30,
            )

            if pr.returncode != 0:
                return {"error": "Failed to read audio", "peaks": [], "duration": duration}

            import struct
            raw = pr.stdout
            # Parse 16-bit signed samples
            n_samples = len(raw) // 2
            if n_samples == 0:
                return {"peaks": [], "duration": duration}

            samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * 2])

            # Bucket into num_peaks groups and take max absolute amplitude
            bucket_size = max(1, n_samples // num_peaks)
            peaks = []
            for i in range(0, n_samples, bucket_size):
                bucket = samples[i:i + bucket_size]
                peak = max(abs(s) for s in bucket) / 32768.0
                peaks.append(round(peak, 3))

            # Trim or pad to exactly num_peaks
            peaks = peaks[:num_peaks]

            return {"peaks": peaks, "duration": round(duration, 2)}

        except Exception as e:
            return {"error": str(e), "peaks": [], "duration": 0}

    # ── Exposed: processing ──────────────────────────────────────────────

    def start_processing(self, url, settings, file_path=None, item_index=None):
        """Add a video processing task to the queue."""
        task = {
            "url": url,
            "settings": settings,
            "file_path": file_path,
            "item_index": item_index,
            "results_before": len(self._results)
        }
        self._pipeline_queue.put(task)
        
        with self._worker_lock:
            if not self._worker_thread or not self._worker_thread.is_alive():
                with self._cancel_lock:
                    self._cancel = False
                from subprocess_utils import reset_cancel
                reset_cancel()
                self._worker_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
                self._worker_thread.start()
                 
        return {"ok": True, "queued": self._pipeline_queue.qsize()}

    def _is_cancelled(self) -> bool:
        with self._cancel_lock:
            return self._cancel

    def _pipeline_worker(self):
        """Worker thread that pulls tasks from the queue and executes the pipeline."""
        while not self._pipeline_queue.empty():
            if self._is_cancelled():
                break
            
            try:
                task = self._pipeline_queue.get_nowait()
            except queue.Empty:
                break
                
            self._processing = True
            self._active_item_index = task["item_index"]
            self._results_before = task["results_before"]
            
            # Reset cancellation state for this specific task
            from subprocess_utils import reset_cancel
            reset_cancel()
            
            try:
                self._run_pipeline(task["url"], task["settings"], task["file_path"])
            except Exception as e:
                logger.exception("Critical error in pipeline")
                self._error(f"Internal error: {e}")
            finally:
                self._pipeline_queue.task_done()
                
        self._processing = False
        self._active_item_index = None

    def cancel_processing(self):
        with self._cancel_lock:
            self._cancel = True
        
        # Clear the pending queue
        while not self._pipeline_queue.empty():
            try:
                task = self._pipeline_queue.get_nowait()
                idx = task.get("item_index")
                if idx is not None:
                    self._js(f"window.onPipelineComplete(false, 0, 0, 'Cancelled', {idx})")
                self._pipeline_queue.task_done()
            except queue.Empty:
                break

        from subprocess_utils import request_cancel
        request_cancel()
        return {"ok": True}

    # ── Exposed: results ─────────────────────────────────────────────────

    def get_results(self):
        clips = []
        for i, p in enumerate(self._results): # type: ignore
            try:
                st = p.stat()
                size_mb = round(st.st_size / (1024 * 1024), 1)
                url = f"http://127.0.0.1:{self._video_port}/{p.name}"
            except OSError:
                size_mb = 0
                url = ""
            clip = {
                "path": str(p),
                "filename": p.name,
                "size_mb": size_mb,
                "url": url,
            }
            # Include source_stem for grouping renamed clips
            if i < len(self._moments) and self._moments[i].get("source_stem"):
                clip["source_stem"] = self._moments[i]["source_stem"]
            clips.append(clip)
        return {"clips": clips, "moments": self._moments}

    def open_output_folder(self):
        try:
            os.startfile(str(self._clips_dir))
        except Exception as e:
            logger.warning("Failed to open output folder: %s", e)
        return {"ok": True}

    def select_clips_folder(self):
        """Open a folder picker dialog and return the chosen path."""
        import webview
        try:
            result = self._window.create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=str(self._clips_dir),
            )
            if result and len(result) > 0:
                return {"path": result[0]}
        except Exception as e:
            logger.debug("select_clips_folder failed: %s", e)
        return {"path": None}

    def select_file(self):
        import webview

        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video files (*.mp4;*.mkv;*.mov;*.webm)", "All files (*.*)"),
        )
        if result and len(result) > 0:
            return {"path": result[0]}
        return {"path": None}

    def select_files_multiple(self):
        """Open file dialog allowing multiple file selection."""
        import webview

        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video files (*.mp4;*.mkv;*.mov;*.webm)", "All files (*.*)"),
            allow_multiple=True,
        )
        if result and len(result) > 0:
            return {"paths": list(result)}
        return {"paths": []}

    # ── Exposed: video preview ───────────────────────────────────────────

    def _cleanup_servers(self):
        """Shut down both video and music HTTP servers."""
        for server in ('_video_server', '_music_server'):
            srv = getattr(self, server, None)
            if srv:
                try:
                    srv.shutdown()
                    srv.server_close()
                except Exception as e:
                    logger.debug("Error shutting down %s: %s", server, e)

    def _restart_video_server(self, new_dir: Path):
        """Shut down the old video server and start a new one serving new_dir."""
        try:
            self._video_server.shutdown()
            self._video_server.server_close()
        except Exception as e:
            logger.debug("Error shutting down old video server: %s", e)
        new_dir.mkdir(parents=True, exist_ok=True)
        self._video_port, self._video_server = _start_video_server(new_dir)
        self._clips_dir = new_dir
        logger.info(f"Video server restarted on port {self._video_port} for {new_dir}")

    def get_video_url(self, clip_index):
        """Return a local HTTP URL for the clip so the HTML5 <video> can play it."""
        if 0 <= clip_index < len(self._results): # type: ignore
            p = self._results[clip_index]
            if p.exists():
                return {"url": f"http://127.0.0.1:{self._video_port}/{p.name}"}
        return {"url": None}

    # ── Exposed: delete clip ────────────────────────────────────────────

    def _robust_unlink(self, path: Path, retries: int = 5, delay: float = 0.3):
        """Attempt to unlink a file with retries, handling temporary locks."""
        if not path or not path.exists():
            return True
        for i in range(retries):
            try:
                path.unlink() # type: ignore
                return True
            except OSError as e:
                if i < retries - 1:
                    logger.warning(f"Retrying unlink of {path.name} (attempt {i+1}/{retries}): {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to unlink {path.name} after {retries} attempts: {e}")
                    return False
            except Exception as e:
                print(f"[cleanup] Unexpected error unlinking {path.name}: {e}")
                return False
        return False

    def delete_clip(self, clip_index):
        """Delete a clip by its index in the current results list."""
        with self._state_lock:
            if 0 <= clip_index < len(self._results): # type: ignore
                p = self._results[clip_index]
                try:
                    if p.exists():
                        self._robust_unlink(p)
                    self._results.pop(clip_index)
                    # Remove matching moments entry
                    if clip_index < len(self._moments):
                        self._moments.pop(clip_index)
                    self._save_state() # type: ignore
                    return {"ok": True}
                except Exception as e:
                    return {"error": str(e)}
        return {"error": "Invalid clip index"}

    def delete_library_file(self, filename):
        """Delete a video file from the clips folder by filename."""
        target = self._clips_dir / filename
        if target.exists() and target.parent == self._clips_dir:
            try:
                self._robust_unlink(target)
                # Also remove from results if it was there
                self._results = [p for p in self._results if p.name != filename]
                self._save_state()
                return {"ok": True} # type: ignore
            except Exception as e:
                return {"error": str(e)}
        return {"error": "File not found"}

    # ── Exposed: library (all videos) ────────────────────────────────────

    def list_all_clips(self):
        """List all video files in the clips directory."""
        clips = []
        total_size = 0
        _exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
        if self._clips_dir.exists(): # type: ignore
            # Single stat() per file — cache the result
            entries = []
            for p in self._clips_dir.iterdir():
                if p.suffix.lower() in _exts:
                    st = p.stat()
                    entries.append((p, st))
            entries.sort(key=lambda x: x[1].st_mtime, reverse=True)
            for p, st in entries:
                total_size += st.st_size
                clips.append({
                    "filename": p.name,
                    "size_mb": round(st.st_size / (1024 * 1024), 1),
                    "modified": st.st_mtime,
                    "url": f"http://127.0.0.1:{self._video_port}/{p.name}",
                })
        return {
            "clips": clips,
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "count": len(clips),
        }

    def import_folder_clips(self):
        """Scan the clips folder and add any videos not already tracked.

        This lets users drop videos into the clips/ folder and have them
        appear in the upload section alongside pipeline-generated clips.
        Returns the updated results list.
        """
        _exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
        existing = {p.resolve() for p in self._results if p.exists()}
        added = 0

        if self._clips_dir.exists(): # type: ignore
            for p in sorted(self._clips_dir.iterdir(), key=lambda x: x.stat().st_mtime):
                if p.suffix.lower() in _exts and p.resolve() not in existing:
                    self._results.append(p)
                    existing.add(p.resolve())
                    added += 1

        if added:
            # Ensure moments metadata list matches results length
            while len(self._moments) < len(self._results):
                self._moments.append({})
            self._save_state() # type: ignore
            print(f"[+] Imported {added} clip(s) from clips folder")

        return self.get_results()

    # ── Exposed: schedule management ─────────────────────────────────────

    def save_scheduled(self, scheduled_list):
        """Replace the full scheduled list (called from JS on every change)."""
        if not isinstance(scheduled_list, list):
            return {"ok": False, "error": "Expected a list"}
        # Validate each item has required fields
        for item in scheduled_list:
            if not isinstance(item, dict):
                return {"ok": False, "error": "Each scheduled item must be a dict"}
            for key in ("clipIdx", "date", "time", "title", "uploaded"):
                if key not in item:
                    return {"ok": False, "error": f"Scheduled item missing required field: {key}"}
        with self._scheduled_lock:
            self._scheduled = scheduled_list
            self._save_state() # type: ignore
        return {"ok": True}

    def get_all_scheduled(self):
        """Return the persisted scheduled list."""
        return {"scheduled": self._scheduled}

    # ── Exposed: upload ──────────────────────────────────────────────────

    def start_upload(self, clips_metadata, schedule_start, interval_hours, channel_id=None, item_index=None):
        """Upload clips with per-clip metadata.

        clips_metadata: list of {index, title, description, tags, category_id, privacy}
        channel_id: YouTube channel ID to upload to (from get_channels())
        item_index: batch queue index for progress callbacks
        """
        if self._processing:
            return {"error": "Processing in progress"}
        self._processing = True
        with self._cancel_lock:
            self._cancel = False
        self._active_item_index = item_index
        threading.Thread(
            target=self._run_upload,
            args=(clips_metadata, schedule_start, interval_hours, channel_id),
            daemon=True,
        ).start()
        return {"ok": True}

    def upload_single_clip(self, clip_index, meta, channel_id=None):
        """Upload a single clip immediately (used by background scheduler)."""
        if clip_index >= len(self._results): # type: ignore
            return {"error": "Invalid clip index"}
        video_path = self._results[clip_index]
        if not video_path.exists():
            return {"error": "Clip file not found"}
        try:
            upload_to_youtube(
                video_path,
                title=meta.get("title", f"Viral Clip #{clip_index + 1}"),
                description=meta.get("description", ""),
                tags=meta.get("tags", ["shorts", "gaming", "gameplay", "clips"]),
                category_id=str(meta.get("category_id", "20")),
                privacy=meta.get("privacy", "private"),
                channel_id=channel_id,
            )
            return {"ok": True}
        except Exception as e: # type: ignore
            return {"error": str(e)}

    # ── Exposed: background scheduler ────────────────────────────────────

    def start_scheduler(self):
        """Start the background upload scheduler thread."""
        if self._scheduler_running:
            return {"ok": True}
        self._scheduler_running = True
        threading.Thread(target=self._scheduler_loop, daemon=True).start() # type: ignore
        print("[+] Background upload scheduler started")
        return {"ok": True}

    # ── Exposed: state persistence ───────────────────────────────────────

    def clear_history(self):
        """Wipe results, moments, and scheduled tasks from the tracking state."""
        with self._state_lock:
            self._results = []
            self._moments = []
            self._scheduled = []
            self._save_state()
        logger.info("History cleared (results, moments, and schedule reset)")
        return {"ok": True}

    def load_persisted_state(self):
        """Return persisted results/moments/scheduled for frontend init."""
        clips = [] # type: ignore
        for i, p in enumerate(self._results):
            try:
                st = p.stat()
            except OSError:
                continue
            clip = {
                "path": str(p),
                "filename": p.name,
                "size_mb": round(st.st_size / (1024 * 1024), 1),
            }
            # Include source_stem so frontend can group renamed clips by source video
            if i < len(self._moments) and self._moments[i].get("source_stem"):
                clip["source_stem"] = self._moments[i]["source_stem"]
            clips.append(clip)
        return {
            "clips": clips,
            "moments": self._moments[:len(self._results)],
            "scheduled": self._scheduled,
        }

    # ── Pipeline orchestrator (background thread) ────────────────────────

    def _get_video_duration(self, path: Path) -> float:
        from subprocess_utils import run as _srun
        r = _srun(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        out = r.stdout.strip()
        return float(out) if out else 0.0

    def _run_pipeline(self, url, settings, local_file_path=None):
        video_path = None
        is_downloaded = False
        subtitle_files = []
        stem = None
        try:
            num_clips_raw = settings.get("num_clips", NUM_CLIPS)
            auto_clips = num_clips_raw == "auto"
            num_clips = NUM_CLIPS if auto_clips else int(num_clips_raw)
            print(f"[*] Pipeline settings: num_clips_raw={num_clips_raw!r}, auto_clips={auto_clips}, num_clips={num_clips}")
            clip_duration = int(settings.get("clip_duration", CLIP_DURATION))
            sentence_buffer = float(settings.get("sentence_buffer", SENTENCE_BUFFER))
            min_gap = int(settings.get("min_gap", MIN_GAP))
            style = settings.get("subtitle_style", SUBTITLE_STYLE)
            model = settings.get("whisper_model", WHISPER_MODEL)
            language = settings.get("whisper_language") or None
            preset = settings.get("ffmpeg_preset", FFMPEG_PRESET)
            crf = str(settings.get("video_crf", VIDEO_CRF))
            video_encoder = settings.get("video_encoder", VIDEO_ENCODER)
            video_decoder = settings.get("video_decoder", VIDEO_DECODER)
            yolo_device = settings.get("yolo_device", YOLO_DEVICE)
            whisper_device = settings.get("whisper_device", WHISPER_DEVICE)
            crop_vertical = bool(settings.get("crop_vertical", True))
            ai_detector = settings.get("ai_detector", AI_DETECTOR_MODE)
            effect = settings.get("video_effect", "none")
            music_file = settings.get("music_file", None)
            music_volume = float(settings.get("music_volume", 0.12))
            music_start = float(settings.get("music_start", 0))
            music_end = float(settings.get("music_end", 0))
            debug_mode = bool(settings.get("debug_logging", False))

            # ── 1. Download or load local file ───────────────────────
            if self._is_cancelled():
                return self._cancelled()

            if local_file_path:
                self._push("download", 0, "Loading video...")
                video_path = Path(local_file_path).expanduser()
                try:
                    video_path = video_path.resolve()
                except OSError:
                    return self._error("Video file not found or could not be accessed.")
                if not video_path.is_file():
                    return self._error("Video file not found.")
                ext = video_path.suffix.lower()
                stem = video_path.stem[:50]
                if ext not in ALLOWED_INPUT_VIDEO_EXT:
                    return self._error(
                        "Unsupported video format. Use MP4, MKV, MOV, or WebM."
                    )
                self._push("download", 100, f"Loaded: {video_path.name}")
            else:
                if not (url or "").strip():
                    return self._error("No video URL provided.")
                self._push("download", 0, "Downloading video...")
                video_path = self._download_with_progress(url.strip())
                is_downloaded = True
                stem = video_path.stem[:50]
                self._push("download", 100, f"Downloaded: {video_path.name}")

            # ── Get video duration (needed for auto clip count + sentence snapping) ──
            try: # type: ignore
                # Using local helper or ffprobe directly via robust logic
                vid_duration = self._get_video_duration(video_path)
            except Exception:
                vid_duration = 600  # default 10 min
            
            # ── Auto clip count ──────────────────────────────────────
            if auto_clips:
                vid_w, vid_h = get_dimensions(video_path)
                # Smart auto: scale clips based on video length
                #   < 5 min  → 2-3 clips
                #   5-15 min → 3-5 clips
                #   15-30 min → 5-8 clips
                #   30-60 min → 8-15 clips
                #   1-2 hrs  → 15-25 clips
                #   2+ hrs   → 25-40 clips
                # Formula: roughly 1 clip per 3-4 minutes, with a minimum of 2
                vid_mins = vid_duration / 60
                if vid_mins < 5:
                    num_clips = max(2, min(3, int(vid_mins / 1.5)))
                elif vid_mins < 15:
                    num_clips = max(3, int(vid_mins / 3))
                elif vid_mins < 30:
                    num_clips = max(5, int(vid_mins / 3.5))
                elif vid_mins < 60:
                    num_clips = max(8, int(vid_mins / 3.5))
                elif vid_mins < 120:
                    num_clips = max(15, min(30, int(vid_mins / 4)))
                else:
                    num_clips = max(25, min(50, int(vid_mins / 4)))
                # Also consider clip duration — shorter clips = can fit more
                if clip_duration < 20:
                    num_clips = int(num_clips * 1.3)
                elif clip_duration > 60:
                    num_clips = max(2, int(num_clips * 0.7))
                num_clips = max(2, min(50, num_clips))
                self._push("detect", 0, f"Auto: {num_clips} clips for {int(vid_mins)}min video")
                print(f"[+] Auto clip count: {num_clips} (video is {vid_duration:.0f}s / {vid_mins:.1f}min)")

            # ── 2. Detect viral moments ──────────────────────────────
            if self._is_cancelled():
                return self._cancelled()
            self._push("detect", 0, "Detecting viral moments...")

            ai_ready = False
            ai_provider = settings.get("ai_provider", AI_PROVIDER)
            gemini_key = self._decrypt(config.GEMINI_API_KEY)
            use_gemini = ai_provider == "gemini" and gemini_client.is_available(gemini_key)
            candidate_count = num_clips
            if ai_detector != "off":
                if use_gemini:
                    ai_ready = True
                else:
                    ai_ready = detector_ready(OLLAMA_DETECTOR_MODEL)
                
                if ai_ready:
                    candidate_count = max(
                        num_clips,
                        num_clips * OLLAMA_DETECTOR_CANDIDATE_MULTIPLIER,
                    )
                    self._push("detect", 5, f"AI detector ready; scanning {candidate_count} candidates...")
                elif ai_detector == "on": # type: ignore
                    print("[ai-detector] Ollama detector requested but unavailable; using heuristic detector")

            moments = find_viral_moments(
                video_path, num_clips=candidate_count, clip_duration=clip_duration, min_gap=min_gap
            )

            if not moments:
                self._push("detect", 100, "No moments found")
                return self._error("No viral moments found. Try a longer video or fewer clips.")

            if ai_ready and len(moments) > num_clips:
                self._push("detect", 35, "Transcribing AI detector candidates...")
                candidate_moments: list[dict] = []
                for cand_idx, m in enumerate(moments, 1):
                    if self._is_cancelled():
                        return self._cancelled()
                    start, end = m["start"], m["end"]
                    wav = SUBTITLES_DIR / f"{video_path.stem[:50]}_candidate{cand_idx}.wav"
                    subtitle_files.append(wav)
                    pct = 35 + int((cand_idx - 1) / max(1, len(moments)) * 35)
                    self._push("detect", pct, f"AI candidate {cand_idx}/{len(moments)}...")
                    try:
                        # Transcribe
                        if extract_audio_clip(video_path, start, end, wav): # type: ignore
                            words = transcribe_clip(
                                wav, model_size=model, language=language, device_pref=whisper_device,
                            )
                            m["_words"] = words
                            m["transcript"] = " ".join(
                                w.get("word", w.get("text", "")) for w in words
                            ).strip()
                        
                        # Visual check (person detection)
                        if crop_vertical:
                            # We use a lower sample count for candidates to keep it fast
                            detections, _, _ = detect_all_persons(
                                video_path, start, end, 1920, 1080, sample_count=20, yolo_device=yolo_device
                            )
                            if detections: # type: ignore
                                hits = sum(1 for _, persons in detections if persons)
                                m["visual_score"] = hits / len(detections)
                            else:
                                m["visual_score"] = 0.0
                        else:
                            m["visual_score"] = 1.0
                        
                        candidate_moments.append(m)
                    finally:
                        try:
                            wav.unlink(missing_ok=True) # type: ignore
                        except Exception as e:
                            logger.debug("Failed to cleanup wav %s: %s", wav, e)

                if candidate_moments:
                    self._push("detect", 72, f"Reranking candidates with {'Gemini' if use_gemini else 'Ollama'}...")

                    if use_gemini: # type: ignore
                        ranked = gemini_client.rerank_moments(candidate_moments, gemini_key, keep=num_clips)
                    else:
                        def _rank_progress(done, total_rank, score):
                            pct = 72 + int(done / max(1, total_rank) * 22)
                            label = "scored" if score else "skipped"
                            self._push("detect", pct, f"AI rerank {done}/{total_rank}: {label}")

                        ranked = rerank_moments(
                            candidate_moments,
                            clip_duration=clip_duration,
                            keep=num_clips,
                            model=OLLAMA_DETECTOR_MODEL,
                            timeout=OLLAMA_DETECTOR_TIMEOUT,
                            on_progress=_rank_progress,
                        )
                        
                    if ranked: # type: ignore
                        moments = ranked
                        print(f"[ai-detector] Selected {len(moments)} clips with AI reranking")
                    else:
                        moments = moments[:num_clips]
                        print("[ai-detector] No valid AI scores; using heuristic candidates")
                else:
                    moments = moments[:num_clips]
                    print("[ai-detector] Candidate transcription failed; using heuristic candidates")
            else:
                moments = moments[:num_clips]

            for m in moments:
                m.pop("_words", None) # type: ignore

            # Append moments (batch mode: preserve previous video's moments)
            with self._state_lock:
                self._moments.extend(moments)

            self._push("detect", 100, f"Found {len(moments)} moments")
            self._js(f"window.onMomentsDetected({json.dumps(moments)}, {self._active_item_index if self._active_item_index is not None else 'null'})")

            # ── 3. Process each clip (parallel with multi-GPU) ──────────
            done: list[Path] = []
            total = len(moments)
            results_lock = threading.Lock()
            gpu_count = get_gpu_count()

            def _process_one(idx: int, m: dict, gpu_idx: int | None = None) -> Path | None:
                """Process a single clip from audio extraction through rendering.

                *gpu_idx* pins whisper / YOLO / NVENC to a specific GPU.
                """
                if self._is_cancelled():
                    return None
                clip_num = idx + 1
                out = self._clips_dir / f"{stem}_viral{clip_num}.mp4"

                # Resume Logic: Skip if clip already exists and is valid
                if out.exists() and validate_shorts_output(out): # type: ignore
                    if m.get("transcript"):
                        print(f"    [+] Clip {clip_num} already processed, skipping.")
                        return out

                start, end = m["start"], m["end"]
                original_duration = end - start

                # ── 3b: extract audio WITH extended buffer ──
                self._clip_push(clip_num, total, "audio", 50, f"Clip {clip_num}/{total}: Extracting audio...")
                wav = SUBTITLES_DIR / f"{stem}_c{clip_num}.wav"
                extended_end = min(end + sentence_buffer, int(vid_duration))
                r = extract_audio_clip(video_path, start, extended_end, wav) # type: ignore
                if not r:
                    self._clip_push(clip_num, total, "render", 100, f"Clip {clip_num}: failed, skipping")
                    return None
                self._clip_push(clip_num, total, "audio", 100, "Audio extracted")

                # ── 3c: transcribe the extended audio ──
                if self._is_cancelled():
                    return None
                self._clip_push(clip_num, total, "transcribe", 0, f"Clip {clip_num}/{total}: Transcribing...")
                words = transcribe_clip( # type: ignore
                    wav, model_size=model, language=language, device_pref=whisper_device,
                    gpu_index=gpu_idx,
                )
                self._clip_push(clip_num, total, "transcribe", 100, f"{len(words)} words transcribed")

                # ── 3c.1: find natural sentence boundary ──
                new_duration = find_sentence_boundary(
                    words,
                    clip_duration=float(original_duration), # type: ignore
                    min_keep=0.60,
                    max_extend=float(sentence_buffer),
                )
                if new_duration is not None:
                    end = start + int(new_duration + 0.5)
                    words = [w for w in words if w["end"] <= (end - start) + 0.1]
                    self._clip_push(clip_num, total, "transcribe", 100,
                                    f"Adjusted to {end - start}s (sentence end)")
                else:
                    words = [w for w in words if w["end"] <= original_duration + 0.1]

                # Update moment info for UI (thread-safe: each thread writes to its own slot)
                m["end"] = end # type: ignore
                m["duration"] = end - start
                m["transcript"] = " ".join(w.get("word", w.get("text", "")) for w in words).strip()

                # ── 3a: compute crop params ──
                crop_params = None
                crop_w, crop_h = get_dimensions(video_path)
                if crop_vertical:
                    if self._is_cancelled():
                        return None
                    self._clip_push(clip_num, total, "audio", 0, f"Clip {clip_num}/{total}: Tracking speakers...")
                    try:
                        crop_params = get_crop_params_dynamic(
                            video_path, start, end, yolo_device=yolo_device, debug_frames=True,
                            gpu_index=gpu_idx,
                        )
                    except Exception as e:
                        print(f"[!] Crop detection failed for clip {clip_num}: {e}")
                        crop_params = None
                    if crop_params:
                        crop_w, crop_h = crop_params[0], crop_params[1]

                # ── 3d: subtitles ──
                if self._is_cancelled():
                    return None
                self._clip_push(clip_num, total, "subtitle", 0, f"Clip {clip_num}/{total}: Generating subtitles...")
                ass = SUBTITLES_DIR / f"{stem}_c{clip_num}.ass" # type: ignore
                with results_lock:
                    subtitle_files.append(ass)
                generate_subtitles(
                    words, ass,
                    video_width=1080,
                    video_height=1920,
                    style=style,
                )
                self._clip_push(clip_num, total, "subtitle", 100, "Subtitles generated")

                # ── 3e: render clip ──
                if self._is_cancelled():
                    return None
                self._clip_push(clip_num, total, "render", 0, f"Clip {clip_num}/{total}: Rendering...")

                resolved_music = None
                if music_file:
                    mp = Path(music_file)
                    if not mp.is_absolute():
                        mp = MUSIC_DIR / music_file
                    if mp.exists(): # type: ignore
                        resolved_music = mp

                clip_result = extract_clip( # type: ignore
                    video_path, start, end, out,
                    subtitle_path=ass if words else None,
                    crop_params=crop_params,
                    preset=preset, crf=crf, encoder=video_encoder, decoder=video_decoder,
                    effect=effect,
                    music_path=resolved_music,
                    music_volume=music_volume,
                    music_trim_start=music_start,
                    music_trim_end=music_end,
                    gpu_index=gpu_idx,
                )

                if clip_result and clip_result.path:
                    if not validate_shorts_output(clip_result.path): # type: ignore
                        self._clip_push(clip_num, total, "render", 100,
                                        f"Clip {clip_num} failed Shorts validation")
                        result = None
                    else:
                        done_path = clip_result.path
                        if not clip_result.subtitles_burned and clip_result.warning:
                            self._clip_push(clip_num, total, "render", 100,
                                            f"Clip {clip_num} done (WARNING: {clip_result.warning})")
                        else:
                            self._clip_push(clip_num, total, "render", 100, f"Clip {clip_num} complete!")
                        result = done_path
                elif clip_result and not clip_result.path:
                    self._clip_push(clip_num, total, "render", 100, f"Clip {clip_num} failed")
                    result = None
                else:
                    self._clip_push(clip_num, total, "render", 100, f"Clip {clip_num} failed")
                    result = None

                # cleanup temp wav
                try:
                    self._robust_unlink(wav) # type: ignore
                except Exception as e:
                    logger.debug("Failed to cleanup wav %s: %s", wav, e)

                return result

            # Run clips in parallel with VRAM-aware GPU assignment
            max_workers = max(gpu_count, 1)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, m in enumerate(moments):
                    if self._is_cancelled():
                        return self._cancelled()
                    gpu_idx = select_least_loaded_gpu(
                        list(range(gpu_count)) if gpu_count > 0 else None
                    ) if gpu_count > 0 else None
                    futures[executor.submit(_process_one, idx, m, gpu_idx)] = idx

                # Collect results in order of submission
                ordered_results = [None] * total
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        ordered_results[idx] = fut.result()
                    except Exception:
                        logger.exception(f"Clip {idx + 1} failed with exception")
                        ordered_results[idx] = None

            done = [r for r in ordered_results if r is not None]

            # Append results (batch mode: preserve previous video's clips)
            with self._state_lock:
                self._results.extend(done)
            self._save_state()
            self._js(f"window.onPipelineComplete(true, {len(done)}, {total}, null, {self._active_item_index if self._active_item_index is not None else 'null'})")

        except CancelledError:
            return self._cancelled() # type: ignore
        except Exception:
            logger.exception("Pipeline error")
            self._error("An unexpected error occurred during processing.")
        finally:
            # Windows needs a moment to release handles (FFmpeg/Clipper/Whisper)
            time.sleep(1.5)

            if debug_mode:
                print("[debug] Skipping cleanup of temporary files.")
                return
            
            if is_downloaded and video_path and video_path.exists():
                try:
                    self._robust_unlink(video_path)
                    print(f"[cleanup] Automatically cleared downloaded source: {video_path.name}")
                except Exception as e:
                    print(f"[cleanup] Could not clear {video_path.name}: {e}")

            # Thorough cleanup of subtitles folder
            for sf in subtitle_files: # type: ignore
                try:
                    self._robust_unlink(sf)
                except Exception as e:
                    logger.debug("Failed to cleanup subtitle %s: %s", sf, e)

            # Catch any orphans using the video stem
            if stem:
                for p in SUBTITLES_DIR.glob(f"{stem}*"):
                    self._robust_unlink(p)

    # ── Download with real progress ──────────────────────────────────────

    def _download_with_progress(self, url):
        """Download via yt-dlp with progress_hooks for live percent updates."""

        def hook(d):
            if self._is_cancelled():
                raise CancelledError("Download cancelled")
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                if total > 0:
                    pct = int(downloaded / total * 100)
                    self._push("download", pct, f"Downloading... {pct}%")
            elif d["status"] == "finished":
                self._push("download", 95, "Merging formats...")

        DOWNLOADS_DIR.mkdir(exist_ok=True)

        # Prefer H.264 (avc1) — universally supported by ffmpeg.
        # Fall back to any codec if avc1/mp4a not available for the video.
        # restrictfilenames removes unicode chars that break Windows paths.
        fmt = (
            "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[acodec^=mp4a]/"
            "bestvideo[vcodec^=avc1][height<=1080]+bestaudio/"
            "bestvideo[height<=1080]+bestaudio/"
            "bestvideo[height<=1080]+worstaudio/"
            "bestvideo+bestaudio/"
            "best"
        )
        ydl_opts = {
            "format": fmt,
            "outtmpl": str(DOWNLOADS_DIR / "%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "restrictfilenames": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [hook],
        }

        # If it looks like a local file path, just use it directly
        if Path(url).exists(): # type: ignore
            return Path(url)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if self._is_cancelled():
                raise CancelledError("Download cancelled after completion")
            return Path(ydl.prepare_filename(info))

    # ── Upload orchestrator (background thread) ──────────────────────────

    def _run_upload(self, clips_metadata, schedule_start_iso, interval_hours, channel_id=None):
        try:
            start_time = None
            if schedule_start_iso:
                start_time = datetime.fromisoformat(schedule_start_iso)
                start_time = start_time.astimezone()  # make timezone-aware (local time)
            total = len(clips_metadata)
            uploaded = 0

            for i, meta in enumerate(clips_metadata):
                if self._is_cancelled():
                    self._js(f"window.onUploadComplete(false, {uploaded}, 'Cancelled')")
                    return
                pct = int((i / total) * 100)
                self._push("upload", pct, f"Uploading clip {i + 1}/{total}...")

                idx = meta.get("index", i)
                if idx >= len(self._results):
                    continue # type: ignore
                video_path = self._results[idx]

                scheduled = None
                if start_time:
                    scheduled = start_time + timedelta(hours=int(interval_hours) * i)

                result = upload_to_youtube(
                    video_path,
                    title=meta.get("title", f"Viral Clip #{i + 1}"),
                    description=meta.get("description", ""),
                    tags=meta.get("tags", ["shorts", "viral", "clips"]),
                    category_id=str(meta.get("category_id", "22")),
                    privacy=meta.get("privacy", "private"),
                    scheduled_time=scheduled,
                    channel_id=channel_id,
                )

                if result is None:
                    raise RuntimeError(f"Upload failed for clip {i + 1}")

                uploaded += 1

                # Auto-delete from disk after successful upload
                if self._delete_after_upload:
                    self._delete_uploaded_clip(idx, video_path)

            self._push("upload", 100, f"All {total} clips uploaded!")
            self._js(f"window.onUploadComplete(true, {uploaded}, null)") # type: ignore

        except Exception as e:
            self._error(f"Upload failed: {e}")
        finally:
            self._processing = False
            with self._cancel_lock:
                self._cancel = False

    # ── Background upload scheduler ──────────────────────────────────────

    def _scheduler_loop(self):
        """Check every 30s for scheduled uploads whose time has arrived."""
        while self._scheduler_running:
            now = datetime.now().astimezone()
            changed = False

            with self._scheduled_lock:
                items = list(self._scheduled)

            for item in items:
                if item.get("uploaded"):
                    continue
                try:
                    sched_dt = datetime.fromisoformat(f"{item['date']}T{item['time']}")  # type: ignore
                    # Make timezone-aware (assume local time)
                    local_tz = datetime.now().astimezone().tzinfo
                    sched_dt = sched_dt.replace(tzinfo=local_tz)
                except KeyError:
                    print(f"[scheduler] Warning: skipping malformed schedule item (missing keys): {item}")
                    continue
                except ValueError:
                    print(f"[scheduler] Warning: skipping schedule item with invalid date/time: {item}")
                    continue

                if now >= sched_dt:
                    clip_idx = item.get("clipIdx", -1)
                    if clip_idx < 0 or clip_idx >= len(self._results):
                        print(f"[scheduler] Warning: clipIdx {clip_idx} out of range (results have {len(self._results)} clips), marking as uploaded")
                        item["uploaded"] = True
                        changed = True
                        continue
                    video_path = self._results[clip_idx]
                    if not video_path.exists(): # type: ignore
                        print(f"[scheduler] Warning: clip file missing: {video_path}, marking as uploaded")
                        item["uploaded"] = True
                        changed = True
                        continue

                    title = item.get("title", f"Viral Clip #{clip_idx + 1}")
                    print(f"[scheduler] Uploading Clip {clip_idx + 1}: {title}")
                    self._js(f"window.onSchedulerStatus({json.dumps(f'Uploading: {title}')})")
                    try:
                        tags = item.get("tags", "shorts, viral, clips")
                        if isinstance(tags, str):
                            tags = [t.strip() for t in tags.split(",") if t.strip()]
                        result = upload_to_youtube(
                            video_path,
                            title=title,
                            description=item.get("description", ""),
                            tags=tags,
                            category_id=str(item.get("category_id", "22")),
                            privacy=item.get("privacy", "private"),
                            channel_id=item.get("channel_id"),
                        )
                        if result is None:
                            raise RuntimeError("upload_to_youtube returned None")

                        item["uploaded"] = True
                        changed = True # type: ignore
                        print(f"[scheduler] Uploaded: {title}")
                        self._js(f"window.onScheduledUploadDone({clip_idx}, true, null)")

                        # Auto-delete from disk after successful upload
                        if self._delete_after_upload:
                            self._delete_uploaded_clip(clip_idx, video_path)

                    except Exception as e:
                        print(f"[scheduler] Upload failed: {e}")
                        self._js(f"window.onScheduledUploadDone({clip_idx}, false, {json.dumps(str(e))})")

            if changed:
                with self._scheduled_lock:
                    self._save_state() # type: ignore
                self._js("window.onScheduleUpdated()")

            time.sleep(30)

    def _delete_uploaded_clip(self, clip_idx, video_path):
        """Delete a clip file from disk after successful upload."""
        if clip_idx in self._uploading_indices:
            return  # already being handled
        self._uploading_indices.add(clip_idx)
        try:
            if self._robust_unlink(video_path):
                logger.info(f"Deleted uploaded clip: {video_path.name}")
                self._js(f"window.onClipDeleted({clip_idx}, {json.dumps(video_path.name)})")
        except Exception as e:
            print(f"[cleanup] Failed to delete {video_path.name}: {e}")
        finally:
            self._uploading_indices.discard(clip_idx)

    # ── State persistence ────────────────────────────────────────────────

    def _save_state(self):
        """Persist results, moments, schedule, and settings to JSON with thread safety and atomic write."""
        with self._state_lock:
            data = {
                "results": [str(p) for p in self._results],
                "moments": self._moments,
                "scheduled": self._scheduled,
                "delete_after_upload": self._delete_after_upload,
                "user_settings": self._user_settings,
            }
            # Windows: retry a few times if the file is locked by another process
            for i in range(5):
                try:
                    # Use a temporary file for atomic write to prevent corruption on crash
                    temp_state = STATE_FILE.with_suffix(".tmp")
                    content = json.dumps(data, indent=2, default=str)
                    temp_state.write_text(content, encoding="utf-8")
                    
                    # Atomic rename (shutil.move handles cross-device and existing files)
                    shutil.move(str(temp_state), str(STATE_FILE))
                    return
                except Exception as e:
                    if i == 4: # type: ignore
                        logger.error(f"Failed to save state after 5 retries: {e}")
                    time.sleep(0.2)

    def _load_state(self):
        """Load persisted state from previous session."""
        with self._state_lock:
            if not STATE_FILE.exists() or STATE_FILE.stat().st_size == 0:
                return
            try:
                text = STATE_FILE.read_text(encoding="utf-8").strip()
                if not text:
                    return
                data = json.loads(text)
                
                # Restore results as Path objects, keeping moments aligned
                paths = [Path(p) for p in data.get("results", [])] # type: ignore
                all_moments = data.get("moments", [])
                # Filter out missing files AND their corresponding moments
                self._results = []
                self._moments = []
                for i, p in enumerate(paths):
                    if p.exists() and p.is_file():
                        self._results.append(p)
                        self._moments.append(all_moments[i] if i < len(all_moments) else {}) # type: ignore
                self._scheduled = data.get("scheduled", [])
                self._delete_after_upload = data.get("delete_after_upload", False)
                self._user_settings = data.get("user_settings", {})
                logger.info(f"Restored state: {len(self._results)} clips, {len(self._scheduled)} scheduled")
            except Exception as e:
                logger.error(f"Failed to load state file {STATE_FILE}: {e}")

    # ── Progress push helpers ────────────────────────────────────────────

    def _push(self, stage, pct, msg):
        self._js(f"window.onPipelineProgress({json.dumps(stage)}, {pct}, {json.dumps(msg)}, {json.dumps(self._active_item_index)})")

    def _clip_push(self, num, total, substep, pct, msg):
        self._js(
            f"window.onClipProgress({num}, {total}, {json.dumps(substep)}, {pct}, {json.dumps(msg)}, {json.dumps(self._active_item_index)})"
        )

    def _error(self, msg):
        self._js(f"window.onPipelineComplete(false, 0, 0, {json.dumps(msg)}, {json.dumps(self._active_item_index)})")
        self._processing = False

    def _cancelled(self):
        self._js(f"window.onPipelineCancelled({json.dumps(self._active_item_index)})")
        self._processing = False

    def _js(self, code):
        """Execute JS in the frontend. Queues calls if window is hidden/minimized."""
        try:
            if self._window:
                self._window.evaluate_js(code)
                return
        except Exception as e:
            logger.debug("Failed to push JS to frontend: %s", e)
        # Window is hidden or unavailable — queue for when it comes back.
        # Only keep the last progress update per type (avoid flooding the queue)
        # but ALWAYS keep completion/error/cancel callbacks.
        is_progress = "onPipelineProgress" in code or "onClipProgress" in code
        is_console = "onConsoleLog" in code
        if is_progress:
            # Replace previous progress of same type
            self._pending_js = [c for c in self._pending_js
                                if ("onPipelineProgress" not in c and "onClipProgress" not in c)]
        if is_console and len([c for c in self._pending_js if "onConsoleLog" in c]) > 200:
            # Trim old console logs to avoid memory bloat
            non_console = [c for c in self._pending_js if "onConsoleLog" not in c]
            console = [c for c in self._pending_js if "onConsoleLog" in c][-100:]
            self._pending_js = non_console + console
        self._pending_js.append(code)

    def flush_pending_js(self):
        """Called from frontend when window is restored — replay any queued JS calls."""
        pending = list(self._pending_js)
        self._pending_js.clear()
        for code in pending:
            try:
                if self._window:
                    self._window.evaluate_js(code)
            except Exception as e:
                logger.debug("Failed to flush pending JS: %s", e)
        return {"flushed": len(pending)}

    @staticmethod
    def _esc(s):
        return str(s).replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$") \
                      .replace("\n", "\\n").replace("\r", "\\r") \
                      .replace("'", "\\'")
