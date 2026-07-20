"""Frame extraction + Qwen3-VL vision analysis for candidate moments.

This module adds an opt-in, multimodal signal to the highlight pipeline:
sample a few frames from each candidate clip, ask a vision model
(Qwen3-VL:4B) to score highlight-worthiness, read on-screen OCR text, and
describe the scene/action. The structured result is merged into reranking
(see ``ollama_detector.vision_score_candidate``).

All calls fail soft — when Ollama or the vision model is unavailable the
pipeline continues with the text-only path.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from ollama_client import ensure_model, generate_vision

logger = logging.getLogger(__name__)


VISION_PROMPT = """You are analyzing frames sampled evenly from one candidate clip of a gaming video, to pick highlights for Shorts/TikTok.

Tasks:
1. Estimate how highlight-worthy this clip is for a gaming audience (0.0 to 1.0).
2. Read any on-screen text (OCR): "Victory"/"Defeat", killfeed, scores, item/weapon names, ability names.
3. Describe the scene, key objects, and the player's action.
4. Detect whether this is a non-gameplay UI screen (main menu, inventory, loading, settings, pause, scoreboard) — these are NOT highlights.

Boost the score for:
- boss kills / final blows / clutches (1vX, last-second wins)
- "Victory"/"Win"/"GG" text, very low remaining HP bars, big explosion / ultimate effects
- impressive mechanics, multi-kills, comebacks, funny reactions

Set is_ui_screen=true and keep the score below 0.2 for:
- menu navigation, lobby, inventory/gear, loading screens, settings, pause menus

Return ONLY a JSON object:
{"highlight_score":0.0,"ocr_text":"","scene":"","objects":[],"is_ui_screen":false,"action":"","reason":""}
"""


def extract_frames(
    video_path: Path,
    start: float,
    end: float,
    count: int = 4,
    width: int = 640,
    tmp_dir: Path | None = None,
) -> list[Path]:
    """Extract *count* evenly-spaced JPG frames across [start, end].

    Returns a list of temp frame paths (caller is responsible for cleanup).
    Falls back to a non-hardware-accelerated FFmpeg call on failure.
    """
    from subprocess_utils import run as _run

    video_path = Path(video_path)
    if tmp_dir is None:
        try:
            from config import SUBTITLES_DIR
            tmp_dir = SUBTITLES_DIR
        except Exception:
            tmp_dir = video_path.parent
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Path] = []
    duration = max(0.1, float(end) - float(start))

    for i in range(max(1, count)):
        frac = (i + 0.5) / count
        ts = float(start) + duration * frac
        out = tmp_dir / f"_vis_{video_path.stem}_{int(ts * 1000)}_{i}.jpg"
        base_cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale={width}:-1",
            "-q:v", "4", str(out),
        ]
        try:
            r = _run(base_cmd, capture_output=True, timeout=60, errors="replace")
            if not (out.exists() and r.returncode == 0):
                # Retry without any hwaccel decode hint (keeps it portable)
                retry = [
                    "ffmpeg", "-y", "-ss", f"{ts:.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-vf", f"scale={width}:-1",
                    "-q:v", "4", str(out),
                ]
                r2 = _run(retry, capture_output=True, timeout=60, errors="replace")
                if not (out.exists() and r2.returncode == 0):
                    continue
            frames.append(out)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Frame extraction failed @%.3f: %s", ts, e)

    return frames


def _frames_to_b64(frames: list[Path]) -> list[str]:
    images: list[str] = []
    for f in frames:
        try:
            with open(f, "rb") as fh:
                images.append(base64.b64encode(fh.read()).decode("utf-8"))
        except Exception:
            continue
    return images


def analyze_moment_frames(
    frames: list[Path],
    model: str,
    timeout: int = 30,
) -> dict[str, Any] | None:
    """Run vision analysis over extracted frames and return normalized metadata."""
    images = _frames_to_b64(frames)
    if not images:
        return None

    data = generate_vision(VISION_PROMPT, images, model=model, timeout=timeout,
                           options={"temperature": 0.1, "num_predict": 300})
    if not isinstance(data, dict):
        return None

    try:
        score = float(data.get("highlight_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    is_ui = bool(data.get("is_ui_screen", False))
    if is_ui:
        score = min(score, 0.15)

    objects = data.get("objects")
    if not isinstance(objects, list):
        objects = []

    return {
        "highlight_score": score,
        "ocr_text": str(data.get("ocr_text", "")).strip()[:300],
        "scene": str(data.get("scene", "")).strip()[:300],
        "objects": objects[:20],
        "is_ui_screen": is_ui,
        "action": str(data.get("action", "")).strip()[:200],
        "reason": str(data.get("reason", "")).strip()[:240],
    }


def vision_ready(model: str) -> bool:
    """Ensure Ollama and the vision model are available (pulls if missing)."""
    return ensure_model(model)


def analyze_moment(
    video_path: Path,
    start: float,
    end: float,
    model: str,
    count: int = 4,
    width: int = 640,
    timeout: int = 30,
    tmp_dir: Path | None = None,
) -> dict[str, Any] | None:
    """End-to-end: extract frames, run vision analysis, clean up temp files."""
    frames = extract_frames(video_path, start, end, count=count, width=width, tmp_dir=tmp_dir)
    try:
        return analyze_moment_frames(frames, model=model, timeout=timeout)
    finally:
        for f in frames:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to cleanup vision frame %s", f)
