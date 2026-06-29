import sys
import json
from pathlib import Path

# In PyInstaller frozen builds, __file__ resolves to the temp _MEIPASS dir.
# User data (downloads, clips, tokens, secrets) must live next to the .exe.
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
CLIPS_DIR = BASE_DIR / "clips"
SUBTITLES_DIR = BASE_DIR / "subtitles"
MUSIC_DIR = BASE_DIR / "music"
TOKENS_DIR = BASE_DIR / "tokens"

for d in [DOWNLOADS_DIR, CLIPS_DIR, SUBTITLES_DIR, MUSIC_DIR, TOKENS_DIR]:
    d.mkdir(exist_ok=True)

CLIENT_SECRETS_FILE = BASE_DIR / "client_secrets.json"
GEMINI_TOKEN_FILE = TOKENS_DIR / "gemini_key.json"

# The Gemini API key is loaded from tokens/gemini_key.json to keep it out of source control.
GEMINI_API_KEY = ""
if GEMINI_TOKEN_FILE.exists():
    try:
        with open(GEMINI_TOKEN_FILE, "r", encoding="utf-8") as f:
            _secrets = json.load(f)
            GEMINI_API_KEY = _secrets.get("gemini_api_key", "")
    except Exception:
        pass
elif CLIENT_SECRETS_FILE.exists():
    # Migration: check old location if new one doesn't exist
    try:
        with open(CLIENT_SECRETS_FILE, "r", encoding="utf-8") as f:
            _secrets = json.load(f)
            GEMINI_API_KEY = _secrets.get("gemini_api_key", "")
    except Exception:
        pass

# Clip detection (gaming-tuned: more clips, shorter duration, tighter gaps)
NUM_CLIPS = 8
CLIP_DURATION = 25
MIN_GAP = 10
AI_DETECTOR_MODE = "auto"  # auto | off | on
AI_PROVIDER = "gemini"     # gemini | ollama
OLLAMA_DETECTOR_MODEL = "qwen2.5:3b"
OLLAMA_DETECTOR_CANDIDATE_MULTIPLIER = 3
OLLAMA_DETECTOR_TIMEOUT = 20

# Whisper
WHISPER_MODEL = "base"
WHISPER_LANGUAGE = None

# Subtitle style
SUBTITLE_STYLE = "tiktok"

# Cropping
CROP_VERTICAL = True          # auto-crop to 9:16 for Shorts

# FFmpeg encoding
FFMPEG_PRESET = "ultrafast"
VIDEO_CRF = "23"
VIDEO_ENCODER = "nvenc"  # auto | nvenc | qsv | amf | cpu | nvenc_hevc | qsv_hevc | amf_hevc | cpu_hevc
VIDEO_DECODER = "cuda"   # auto | cuda | d3d11va | dxva2 | vaapi | v4l2m2m | cpu

# AI / detection devices
YOLO_DEVICE = "cuda"     # auto | cuda | cpu
WHISPER_DEVICE = "cuda"  # auto | cuda | cpu

# YouTube
TOKEN_FILE = BASE_DIR / "token.json"
DEFAULT_TAGS = ["shorts", "gaming", "gameplay", "clips"]
