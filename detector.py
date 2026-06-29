import re
import subprocess
import shutil
import numpy as np
from pydub import AudioSegment
import tempfile
import time
from pathlib import Path

from subprocess_utils import run as _run, is_cancelled, CancelledError
from hwaccel import input_hwaccel_args
from utils import fmt_time, wait_for_file_unlock

import logging
logger = logging.getLogger(__name__)

# Explicitly point pydub to ffmpeg/ffprobe binaries found in PATH
def _sync_pydub_paths():
    _ffmpeg_path = shutil.which("ffmpeg")
    _ffprobe_path = shutil.which("ffprobe")
    if _ffmpeg_path:
        AudioSegment.converter = _ffmpeg_path
    if _ffprobe_path:
        AudioSegment.ffprobe = _ffprobe_path

_sync_pydub_paths()


def find_viral_moments(
    video_path: Path,
    num_clips: int = 5,
    clip_duration: int = 30,
    min_gap: int = 15,
) -> list:
    """Find viral moments using audio energy + scene change analysis (no AI)."""

    logger.info("Analyzing audio energy (waiting for file access)...")
    
    # Windows can sometimes hold a lock on newly downloaded files (antivirus/indexing).
    if not wait_for_file_unlock(video_path, timeout=5.0):
        logger.warning("File still locked after 5s, proceeding anyway...")

    _sync_pydub_paths()  # Refresh paths in case they were updated during runtime
    
    total_seconds = _video_duration_seconds(video_path)
    if total_seconds < 10:
        print("[!] Video too short for analysis")
        return []

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # To handle files > 4GB and avoid MemoryErrors, we extract a downsampled 
        # mono version of the audio for analysis instead of loading the full stream.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            # Extract at 8kHz mono — plenty for energy/peak analysis
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "8000", "-acodec", "pcm_s16le",
                "-rf64", "auto", tmp_path
            ]
            _run(cmd, capture_output=True, timeout=300)
            
            audio = AudioSegment.from_file(tmp_path)
        finally:
            if tmp_path:
                # Attempt cleanup with retries for Windows locks
                for _ in range(5):
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                        break
                    except OSError:
                        time.sleep(0.2)
        
    except Exception as e:
        if isinstance(e, IndexError):
            detail = " (no audio stream found or file corrupted)"
        else:
            error_msg = str(e).lower()
            detail = " (ffprobe/ffmpeg missing or path issue)" if "ffprobe" in error_msg or "ffmpeg" in error_msg or "no such file" in error_msg else f" ({e})"
        print(f"[!] Audio analysis unavailable{detail}. Using fallback detection.")
        audio = None

    if audio is None:
        energies = np.zeros(total_seconds, dtype=float)
    else:
        # --- Audio RMS energy (1-second windows) ---
        window_ms = 1000
        energies_list = []
        for i in range(0, len(audio), window_ms):
            if is_cancelled():
                raise CancelledError("Audio analysis cancelled")
            energies_list.append(audio[i : i + window_ms].rms)
        energies = np.array(energies_list, dtype=float)

    if len(energies) == 0:
        return _fallback_moments(total_seconds, num_clips, clip_duration, min_gap)

    # Smooth
    kernel = np.ones(5) / 5
    smoothed = np.convolve(energies, kernel, mode="same")

    # --- Volume variance (dynamic = interesting) ---
    var_window = 10
    variance = np.array(
        [
            np.std(energies[max(0, i - var_window // 2) : i + var_window // 2])
            for i in range(len(energies))
        ]
    )

    # --- Scene change density ---
    logger.info("Analyzing scene changes...") # type: ignore
    if is_cancelled():
        raise CancelledError("Moment detection cancelled before scene analysis")
        
    scene_density = _scene_change_density(video_path, len(energies))

    # --- Combine (normalize each to 0-1) ---
    def norm(a):
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 1e-8 else np.zeros_like(a)

    combined = (
        0.45 * norm(smoothed)
        + 0.25 * norm(variance)
        + 0.30 * norm(scene_density[: len(smoothed)])
    )

    # --- Pick top N non-overlapping peaks ---
    half = clip_duration // 2
    clips = []
    for _ in range(num_clips):
        if combined.max() <= 0:
            break
        peak = int(np.argmax(combined))
        start = max(0, peak - half)
        end = min(len(combined), start + clip_duration)
        if end - start < clip_duration and start > 0:
            start = max(0, end - clip_duration)

        clips.append(
            {"start": start, "end": end, "duration": end - start, "score": float(combined[peak])}
        )

        # mask out neighbourhood
        lo = max(0, peak - clip_duration - min_gap)
        hi = min(len(combined), peak + clip_duration + min_gap)
        combined[lo:hi] = 0

    clips.sort(key=lambda c: c["start"])

    if not clips:
        clips = _fallback_moments(total_seconds, num_clips, clip_duration, min_gap)

    logger.info(f"Found {len(clips)} viral moments") # type: ignore
    for i, c in enumerate(clips):
        print(f"    Clip {i+1}: {fmt_time(c['start'])} - {fmt_time(c['end'])}  (score {c['score']:.2f})")
    return clips


# ── helpers ──────────────────────────────────────────────────────────────────


def _scene_change_density(video_path: Path, length: int) -> np.ndarray:
    """Count scene changes per second using ffmpeg."""
    try:
        cmd = [
            "ffmpeg",
            *input_hwaccel_args(),
            "-i", str(video_path),
            "-an", "-sn",
            "-vf", "fps=2,select='gt(scene,0.3)',showinfo",
            "-vsync", "vfr", "-f", "null", "-",
            "-threads", "4",
        ]
        r = _run(cmd, capture_output=True, text=True, timeout=600, errors="replace")
        if r.returncode != 0:
            logger.warning("Scene detection unavailable, using audio only")
            return np.zeros(length + 1)
        timestamps = []
        pts_pattern = re.compile(r"pts_time:\s*([\d.]+)")
        for line in (r.stderr or "").split("\n"):
            m = pts_pattern.search(line)
            if m:
                try:
                    timestamps.append(float(m.group(1)))
                except ValueError:
                    pass

        density = np.zeros(length + 1)
        win = 10
        for ts in timestamps:
            lo = max(0, int(ts) - win // 2)
            hi = min(length + 1, int(ts) + win // 2)
            density[lo:hi] += 1
        return density

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        logger.warning("Scene detection unavailable, using audio only")
        return np.zeros(length + 1)


def _video_duration_seconds(video_path: Path) -> int:
    try:
        r = _run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            errors="replace",
        )
        out = (r.stdout or "0").strip()
        if not out or out == "0":
            return 0
        # Handle cases where ffprobe might return multiple lines or non-numeric output
        return max(0, int(float(out.split()[0])))
    except Exception:
        return 0


def _fallback_moments(
    total_seconds: int,
    num_clips: int,
    clip_duration: int,
    min_gap: int,
) -> list[dict]:
    """Return evenly spaced clips when scoring has no usable peaks."""
    usable_duration = min(clip_duration, total_seconds)
    if usable_duration <= 0:
        return []

    if total_seconds <= usable_duration:
        starts = [0]
    else:
        max_start = total_seconds - usable_duration
        spacing = max(1, usable_duration + min_gap)
        count = max(1, min(num_clips, max_start // spacing + 1))
        if count == 1:
            starts = [max_start // 2]
        else:
            starts = [round(i * max_start / (count - 1)) for i in range(count)]

    clips = []
    for start in starts[:num_clips]:
        end = min(total_seconds, int(start) + usable_duration)
        clips.append(
            {
                "start": int(start),
                "end": int(end),
                "duration": int(end - start),
                "score": 0.0,
            }
        )

    if clips: # type: ignore
        logger.warning("Detector scores were flat; using evenly spaced fallback clips")
    return clips


_fmt = fmt_time
