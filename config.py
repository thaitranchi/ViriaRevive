import logging
import sys
import json
from pathlib import Path

logger = logging.getLogger(__name__)

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
OPENROUTER_TOKEN_FILE = TOKENS_DIR / "openrouter_key.json"

# The Gemini API key is loaded from tokens/gemini_key.json to keep it out of source control.
# It may be stored as a Fernet-encrypted blob (from GUI settings) or as a plaintext key.
# If encrypted, config.py stores it as-is; api_bridge.py's _decrypt() handles decryption.
# Direct consumers (e.g. gemini_client.py) must use api_bridge to get the decrypted key.
GEMINI_API_KEY = ""
GEMINI_API_KEY_ENCRYPTED = False
if GEMINI_TOKEN_FILE.exists():
    try:
        with open(GEMINI_TOKEN_FILE, "r", encoding="utf-8") as f:
            _secrets = json.load(f)
            _val = _secrets.get("gemini_api_key", "")
            # Fernet tokens start with gAAAA
            if _val.startswith("gAAAA"):
                GEMINI_API_KEY_ENCRYPTED = True
            GEMINI_API_KEY = _val
    except Exception as e:
        logger.debug("Failed to load Gemini key from %s: %s", GEMINI_TOKEN_FILE, e)
elif CLIENT_SECRETS_FILE.exists():
    # Migration: check old location if new one doesn't exist
    try:
        with open(CLIENT_SECRETS_FILE, "r", encoding="utf-8") as f:
            _secrets = json.load(f)
            _val = _secrets.get("gemini_api_key", "")
            if _val.startswith("gAAAA"):
                GEMINI_API_KEY_ENCRYPTED = True
            GEMINI_API_KEY = _val
    except Exception as e:
        logger.debug("Failed to load Gemini key from %s: %s", CLIENT_SECRETS_FILE, e)

# OpenRouter API key — loaded from tokens/openrouter_key.json (same encryption scheme)
OPENROUTER_API_KEY = ""
OPENROUTER_API_KEY_ENCRYPTED = False
if OPENROUTER_TOKEN_FILE.exists():
    try:
        with open(OPENROUTER_TOKEN_FILE, "r", encoding="utf-8") as f:
            _or_secrets = json.load(f)
            _or_val = _or_secrets.get("openrouter_api_key", "")
            if _or_val.startswith("gAAAA"):
                OPENROUTER_API_KEY_ENCRYPTED = True
            OPENROUTER_API_KEY = _or_val
    except Exception as e:
        logger.debug("Failed to load OpenRouter key from %s: %s", OPENROUTER_TOKEN_FILE, e)

OPENROUTER_MODEL = "openai/gpt-4o-mini"


# Clip detection (gaming-tuned: more clips, shorter duration, tighter gaps)
NUM_CLIPS = 8
CLIP_DURATION = 25
MIN_GAP = 10
SENTENCE_BUFFER = 5
AI_DETECTOR_MODE = "auto"  # auto | off | on
AI_PROVIDER = "gemini"     # gemini | ollama | openrouter
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

