"""Pipeline artifact caching — avoids redundant re-computation on resume.

Cacheable artifacts:
  - WAV files (per clip)
  - Transcript word timestamps (per clip)
  - YOLO crop parameters (per clip time range)
  - Completed clip paths (resume detection)
  - Pipeline state checkpoint (download, detection, per-clip)
"""

import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import logging

import config

logger = logging.getLogger(__name__)


# ── Pipeline State ───────────────────────────────────────────────────────────


@dataclass
class PipelineState:
    url: str = ""
    stem: str = ""
    num_clips: int = 0
    clip_duration: int = 0
    step_downloaded: bool = False
    step_detected: bool = False
    step_reranked: bool = False
    clips_completed: list[int] = field(default_factory=list)
    moments: list[dict] = field(default_factory=list)

    @property
    def all_clips_done(self) -> bool:
        return len(self.clips_completed) >= self.num_clips if self.num_clips > 0 else False

    @property
    def resume_step(self) -> str:
        if not self.step_downloaded:
            return "download"
        if not self.step_detected:
            return "detect"
        if not self.step_reranked:
            return "rerank"
        if not self.all_clips_done:
            return "clips"
        return "done"


# ── Pipeline Cache ───────────────────────────────────────────────────────────


class PipelineCache:
    """Thread-safe cache for pipeline artifacts.

    Key design:
      - State stored as JSON in ``{CLIPS_DIR}/.pipeline_state_{stem}.json``
      - WAV files cached under ``{SUBTITLES_DIR}/{stem}_c{clip}.wav``
      - Transcripts cached in state JSON (lightweight)
      - Clip outputs live in ``{CLIPS_DIR}/{stem}_viral{clip}.mp4``
    """

    def __init__(self, stem: str):
        self.stem = stem
        self._state_path = config.CLIPS_DIR / f".pipeline_state_{stem}.json"
        self._lock = threading.Lock()

    # ── State persistence ───────────────────────────────────────────────

    def load_state(self) -> PipelineState:
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PipelineState(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return PipelineState(stem=self.stem)

    def save_state(self, state: PipelineState):
        with self._lock:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2, default=str)
            tmp.replace(self._state_path)

    def clear_state(self):
        self._state_path.unlink(missing_ok=True)

    # ── WAV caching ─────────────────────────────────────────────────────

    def _wav_path(self, clip_num: int) -> Path:
        return config.SUBTITLES_DIR / f"{self.stem}_c{clip_num}.wav"

    def _cache_key(self, prefix: str, clip_num: int) -> str:
        return f"{prefix}_{clip_num}"

    def has_wav(self, clip_num: int) -> bool:
        return self._wav_path(clip_num).exists()

    def get_wav(self, clip_num: int) -> Optional[Path]:
        p = self._wav_path(clip_num)
        return p if p.exists() else None

    # ── Transcript caching ──────────────────────────────────────────────

    def get_transcript(self, clip_num: int) -> Optional[list]:
        state = self.load_state()
        for m in state.moments:
            if m.get("_clip_num") == clip_num:
                return m.get("_words")
        return None

    def set_transcript(self, clip_num: int, words: list, transcript_text: str):
        state = self.load_state()
        for m in state.moments:
            if m.get("_clip_num") == clip_num:
                m["_words"] = words
                m["transcript"] = transcript_text
                break
        else:
            state.moments.append({"_clip_num": clip_num, "_words": words, "transcript": transcript_text})
        self.save_state(state)

    # ── Crop params caching ─────────────────────────────────────────────

    def _reconstruct_crop_params(self, raw):
        """Convert JSON-safe list back to original tuple form."""
        if raw is None:
            return None
        if isinstance(raw, list):
            if len(raw) == 4:
                return tuple(raw)
            if len(raw) == 3:
                cw, ch, kf = raw
                if isinstance(kf, list):
                    return (cw, ch, [tuple(k) for k in kf])
                return (cw, ch, kf)
        return raw

    def get_crop_params(self, clip_num: int):
        state = self.load_state()
        for m in state.moments:
            if m.get("_clip_num") == clip_num:
                raw = m.get("_crop_params")
                return self._reconstruct_crop_params(raw)
        return None

    def set_crop_params(self, clip_num: int, crop_params):
        state = self.load_state()
        for m in state.moments:
            if m.get("_clip_num") == clip_num:
                m["_crop_params"] = crop_params
                break
        else:
            state.moments.append({"_clip_num": clip_num, "_crop_params": crop_params})
        self.save_state(state)

    # ── Completed clip detection ────────────────────────────────────────

    def is_clip_done(self, clip_num: int) -> bool:
        state = self.load_state()
        return clip_num in state.clips_completed

    def mark_clip_done(self, clip_num: int):
        state = self.load_state()
        if clip_num not in state.clips_completed:
            state.clips_completed.append(clip_num)
        self.save_state(state)

    def done_clips(self) -> set[int]:
        state = self.load_state()
        return set(state.clips_completed)

    # ── Moments cache (lightweight: just start/end/duration) ────────────

    def set_moments(self, moments: list[dict]):
        state = self.load_state()
        state.moments = moments
        self.save_state(state)

    def get_moments(self) -> list[dict]:
        state = self.load_state()
        return state.moments

    # ── WAV lifecycle: keep across AI reranking + clip processing ───────

    def keep_wav(self, clip_num: int):
        """Mark a WAV file for retention (don't delete between stages)."""
        marker = config.SUBTITLES_DIR / f".keep_{self.stem}_c{clip_num}.wav"
        marker.touch()

    def should_keep_wav(self, clip_num: int) -> bool:
        marker = config.SUBTITLES_DIR / f".keep_{self.stem}_c{clip_num}.wav"
        return marker.exists()

    def release_wav(self, clip_num: int):
        marker = config.SUBTITLES_DIR / f".keep_{self.stem}_c{clip_num}.wav"
        marker.unlink(missing_ok=True)

    def cleanup_all_wavs(self):
        for p in config.SUBTITLES_DIR.glob(f"{self.stem}_c*.wav"):
            p.unlink(missing_ok=True)
        for p in config.SUBTITLES_DIR.glob(f".keep_{self.stem}_c*.wav"):
            p.unlink(missing_ok=True)
