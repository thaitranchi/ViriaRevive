import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from subprocess_utils import run as _run
from hwaccel import (
    input_hwaccel_args,
    video_encode_args,
    run_ffmpeg_with_encode_fallback,
)
from utils import fmt_time, wait_for_file_unlock

import logging
logger = logging.getLogger(__name__)

DEBUG = False  # Toggle full ffmpeg command logging

@dataclass
class ClipResult:
    path: Path | None
    subtitles_burned: bool = True
    warning: str | None = None


SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
ALOOP_MAX_SIZE = 2000000000  # 2e9 samples (~41h at 48kHz) for infinite audio loop


# ── Subtitle filter detection (cached) ────────────────────────────────────────

_sub_filter_cache: str | None = None


def _detect_subtitle_filter() -> str:
    """Detect the best available subtitle filter in ffmpeg.

    Prefers 'subtitles' (better font handling on Windows) over 'ass'.
    """
    global _sub_filter_cache
    if _sub_filter_cache is not None:
        return _sub_filter_cache

    try:
        r = _run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, errors="replace", timeout=10,
        )
        output = r.stdout
        for filt in ["subtitles", "ass"]:
            if re.search(rf'\b{filt}\b', output):
                _sub_filter_cache = filt
                logger.info(f"Using ffmpeg subtitle filter: {filt}")
                return filt
    except Exception:
        logger.exception("Failed to detect ffmpeg subtitle filter")

    _sub_filter_cache = ""
    print("[!] No subtitle filter available in ffmpeg (need libass)")
    return ""


def _escape_sub_path_win(path: Path) -> str:
    """Escape a subtitle file path for ffmpeg filter on Windows."""
    s = str(path).replace("\\", "/")
    s = s.replace(":", "\\:")
    return s


def _copy_fonts_to_dir(dest_dir: Path):
    """Copy common fonts to subtitle temp dir so libass can find them without fontconfig."""
    import platform
    if platform.system() != "Windows":
        return
    fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    for name in ["arial.ttf", "arialbd.ttf", "ariblk.ttf", "impact.ttf", "verdana.ttf"]:
        src = fonts_dir / name
        dst = dest_dir / name
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(str(src), str(dst))
            except OSError as e:
                logger.debug("Failed to copy font %s: %s", name, e)


def _fonts_dir_option(sub_dir: Path, use_cwd: bool) -> str:
    """Return fontsdir option for subtitle filter. Uses local dir with copied fonts."""
    import platform
    if platform.system() != "Windows":
        return ""
    if use_cwd:
        return ":fontsdir=."
    escaped = str(sub_dir).replace("\\", "/").replace(":", "\\:")
    return f":fontsdir={escaped}"


def _sanitize_filename(name: str) -> str:
    """Remove characters invalid on Windows filenames."""
    return re.sub(r'[<>:"/\\|?*\'`]', '_', name)[:80]

def _prepare_subtitle_file(subtitle_path: Path, output_stem: str) -> tuple[Path | None, Path | None]:
    """Copy subtitle file to a temp location with a safe ASCII name.

    Returns (temp_sub_path, temp_dir) or (None, None).
    """
    if not subtitle_path or not Path(subtitle_path).exists():
        return None, None
    if Path(subtitle_path).stat().st_size < 20:
        return None, None

    sub_dir = Path(tempfile.gettempdir()) / "viria_subs"
    sub_dir.mkdir(exist_ok=True)
    temp_sub = sub_dir / f"sub_{_sanitize_filename(output_stem)}.ass"

    # Plain copy — no BOM (BOM breaks libass ASS header parsing)
    shutil.copy2(str(subtitle_path), str(temp_sub))

    # Copy font files locally so libass finds them without fontconfig
    _copy_fonts_to_dir(sub_dir)

    return temp_sub, sub_dir


def _try_subtitle_burn(input_path: Path, output_path: Path, temp_sub: Path, sub_dir: Path,
                        preset: str, crf: str, copy_audio: bool = False,
                        encoder: str = "auto", decoder: str = "auto",
                        gpu_index: int | None = None) -> bool:
    """Try to burn subtitles into a video. Tries multiple approaches.

    Returns True on success.
    """
    filt = _detect_subtitle_filter()
    if not filt:
        return False

    audio_args = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-strict", "-2", "-b:a", "128k"]
    enc_args = video_encode_args(preset, crf, encoder, gpu_index=gpu_index)

    # Attempt 1: filename-only with CWD set to subtitle directory + local fontsdir
    fontsdir_cwd = _fonts_dir_option(sub_dir, use_cwd=True)
    vf = f"{filt}={temp_sub.name}{fontsdir_cwd}"
    cmd = [
        "ffmpeg", "-y",
        *input_hwaccel_args(decoder, gpu_index=gpu_index),
        "-i", str(input_path),
        "-vf", vf,
        *enc_args,
        *audio_args,
        str(output_path),
    ]
    if DEBUG:
        logger.info(f"Subs attempt 1 (cwd): {' '.join(cmd)}")
    r = run_ffmpeg_with_encode_fallback(
        cmd,
        lambda c: _run(c, capture_output=True, text=True, errors="replace", cwd=str(sub_dir)),
        preset, crf,
    )
    if r.returncode == 0:
        if r.stderr:
            stderr_lines = [l for l in r.stderr.split('\n') if 'font' in l.lower() or 'libass' in l.lower()]
            if stderr_lines:
                logger.info(f"Font info: {'; '.join(stderr_lines[:3])}")
        logger.info("Subtitles burned successfully (cwd method)")
        return True

    logger.warning(f"Attempt 1 (cwd) failed: {(r.stderr or '')[-200:]}")

    # Attempt 2: full escaped path + fontsdir, no CWD
    escaped = _escape_sub_path_win(temp_sub)
    fontsdir_full = _fonts_dir_option(sub_dir, use_cwd=False)
    vf2 = f"{filt}={escaped}{fontsdir_full}"
    cmd2 = [
        "ffmpeg", "-y",
        *input_hwaccel_args(decoder, gpu_index=gpu_index),
        "-i", str(input_path),
        "-vf", vf2,
        *enc_args,
        *audio_args,
        str(output_path),
    ]
    if DEBUG:
        logger.info(f"Subs attempt 2 (escaped path): {' '.join(cmd2)}")
    r2 = run_ffmpeg_with_encode_fallback(
        cmd2,
        lambda c: _run(c, capture_output=True, text=True, errors="replace"),
        preset, crf,
    )
    if r2.returncode == 0:
        if r2.stderr:
            stderr_lines = [l for l in r2.stderr.split('\n') if 'font' in l.lower() or 'libass' in l.lower()]
            if stderr_lines:
                logger.info(f"Font info: {'; '.join(stderr_lines[:3])}")
        logger.info("Subtitles burned successfully (escaped path method)")
        return True

    logger.warning(f"Attempt 2 (escaped path) failed: {(r2.stderr or '')[-200:]}")

    # Attempt 3: try the other filter if available
    other = "ass" if filt == "subtitles" else "subtitles"
    try:
        r_check = _run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10)
        if re.search(rf'\b{other}\b', r_check.stdout):
            vf3 = f"{other}={temp_sub.name}{fontsdir_cwd}"
            cmd3 = [
                "ffmpeg", "-y",
                *input_hwaccel_args(decoder, gpu_index=gpu_index),
                "-i", str(input_path),
                "-vf", vf3,
                *enc_args,
                *audio_args,
                str(output_path),
            ]
            if DEBUG:
                logger.info(f"Subs attempt 3 ({other} filter): {' '.join(cmd3)}")
            r3 = run_ffmpeg_with_encode_fallback(
                cmd3,
                lambda c: _run(c, capture_output=True, text=True, errors="replace", cwd=str(sub_dir)),
                preset, crf,
            )
            if r3.returncode == 0:
                if r3.stderr:
                    stderr_lines = [l for l in r3.stderr.split('\n') if 'font' in l.lower() or 'libass' in l.lower()]
                    if stderr_lines:
                        logger.info(f"Font info: {'; '.join(stderr_lines[:3])}")
                logger.info(f"Subtitles burned successfully ({other} filter)")
                return True
            logger.warning(f"Attempt 3 ({other}) failed: {(r3.stderr or '')[-200:]}")
    except Exception as e:
        logger.debug("Alternate subtitle filter failed: %s", e)

    return False


# ── Crop filter expression builder ───────────────────────────────────────────


def _build_crop_vf(crop_params: tuple, duration: float) -> str:
    """Build the -vf crop filter string. Handles static and dynamic crop.

    For dynamic crop, builds a piecewise-linear time expression for the
    x/y offset — no external files needed, works on all ffmpeg versions.
    """
    if len(crop_params) == 4:
        # Static crop: (cw, ch, cx, cy)
        cw, ch, cx, cy = crop_params
        cw, ch = (cw // 2) * 2, (ch // 2) * 2  # Ensure even for HW encoders
        return f"crop={cw}:{ch}:{cx}:{cy}"

    if len(crop_params) == 3 and isinstance(crop_params[2], list):
        # Dynamic crop: (cw, ch, [(t, x, y), ...])
        cw, ch, keyframes = crop_params
        if not keyframes:
            cw, ch = (cw // 2) * 2, (ch // 2) * 2
            return f"crop={cw}:{ch}:0:0"

        # Downsample keyframes to max 15 to keep expression manageable.
        # IMPORTANT: Always keep keyframes where position changes (transitions).
        # Only drop keyframes that repeat the same position as their predecessor.
        if len(keyframes) > 15:
            # First pass: mark all transition keyframes (position changes)
            must_keep = {0, len(keyframes) - 1}  # always keep first and last
            for i in range(1, len(keyframes)):
                prev_x, prev_y = keyframes[i - 1][1], keyframes[i - 1][2]
                cur_x, cur_y = keyframes[i][1], keyframes[i][2]
                if cur_x != prev_x or cur_y != prev_y:
                    must_keep.add(i)
                    if i > 0:
                        must_keep.add(i - 1)  # keep the frame before transition too

            if len(must_keep) <= 15:
                # We can fit all transitions — fill remaining slots evenly
                remaining = 15 - len(must_keep)
                optional = [i for i in range(len(keyframes)) if i not in must_keep]
                if optional and remaining > 0:
                    step = max(1, len(optional) / remaining)
                    extras = {optional[int(j * step)] for j in range(min(remaining, len(optional)))}
                    must_keep |= extras
                keyframes = [keyframes[i] for i in sorted(must_keep)]
            else:
                # More than 15 transitions — keep them all, they're all important
                keyframes = [keyframes[i] for i in sorted(must_keep)]

        # Build step-function x and y expressions
        x_expr = _build_lerp_expr([t for t, x, y in keyframes], [x for t, x, y in keyframes])
        y_expr = _build_lerp_expr([t for t, x, y in keyframes], [y for t, x, y in keyframes])

        cw, ch = (cw // 2) * 2, (ch // 2) * 2
        return f"crop={cw}:{ch}:{x_expr}:{y_expr}"

    # Fallback — shouldn't happen
    cw, ch = crop_params[0], crop_params[1]
    return f"crop={cw}:{ch}:0:0"


def _shorts_vf() -> str:
    """Normalize any video stream to exact 1080x1920 Shorts output by cropping."""
    return (
        f"scale={SHORTS_WIDTH}:{SHORTS_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={SHORTS_WIDTH}:{SHORTS_HEIGHT},setsar=1"
    )


def _blur_pad_vf() -> str:
    """Reformat to 1080x1920 using a blurred background instead of cropping."""
    # 1. Scale background to fill 1080x1920 (increase) and blur it
    # 2. Scale foreground to fit 1080x1920 (decrease)
    # 3. Overlay foreground on blurred background
    return (
        f"split[v1][v2];"
        f"[v1]scale={SHORTS_WIDTH}:{SHORTS_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={SHORTS_WIDTH}:{SHORTS_HEIGHT},boxblur=20:10[bg];"
        f"[v2]scale={SHORTS_WIDTH}:{SHORTS_HEIGHT}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def _chain_vf(*filters: str | None) -> str:
    return ",".join(f for f in filters if f)


def _probe_dimensions(path: Path) -> tuple[int, int]:
    try:
        r = _run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        parts = [p for p in r.stdout.strip().split(",") if p.strip()]
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _probe_duration(path: Path) -> float:
    try:
        r = _run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return max(0.0, float((r.stdout or "").strip()))
    except Exception:
        return 0.0


def validate_shorts_output(path: Path) -> bool:
    """Return True only for exact 1080x1920 output."""
    width, height = _probe_dimensions(Path(path))
    ok = width == SHORTS_WIDTH and height == SHORTS_HEIGHT
    if ok:
        logger.info(f"Shorts output validated: {width}x{height}")
    else:
        print(f"[!] Shorts output validation failed: {width}x{height}, expected {SHORTS_WIDTH}x{SHORTS_HEIGHT}")
    return ok


def _validated_result(path: Path, subtitles_burned: bool = True, warning: str | None = None) -> ClipResult:
    if path and validate_shorts_output(path):
        return ClipResult(path=path, subtitles_burned=subtitles_burned, warning=warning)
    return ClipResult(path=None, subtitles_burned=subtitles_burned, warning="Output was not 1080x1920")


def _build_lerp_expr(times: list, values: list) -> str:
    """Build an ffmpeg step-function expression from keyframes (instant cuts).

    For 3 keyframes at t=0,4,8 with values 100,200,150:
    → if(lt(t,4),100,if(lt(t,8),200,150))
    """
    if not times or not values:
        return "0"
    if len(set(values)) == 1:
        return str(int(values[0]))
    if len(times) == 1:
        return str(int(values[0]))
    return _step_recursive(times, values, 0)


def _step_recursive(times: list[float], values: list[float], idx: int) -> str:
    """Recursively build nested if() for step function (instant cuts)."""
    if idx >= len(times) - 1:
        return str(int(values[-1]))

    t1 = times[idx + 1]
    v0 = int(values[idx])
    rest = _step_recursive(times, values, idx + 1)

    if v0 == int(values[idx + 1]) and idx + 2 >= len(times):
        return str(v0)

    return f"if(lt(t\\,{t1:.3f})\\,{v0}\\,{rest})"


def _fallback_shorts_encode(video_path: Path, start: float, duration: float, 
                             output_path: Path, preset: str, crf: str, 
                             encoder: str, decoder: str = "auto") -> Path | None:
    """Last-resort encode using pure software (libx264) and blurry padding."""
    logger.warning("Running software fallback encode...")
    enc_args = video_encode_args(preset, crf, force_cpu=True)
    cmd = [ # type: ignore
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-filter_complex", _blur_pad_vf(),
        *enc_args,
        "-c:a", "aac", "-b:a", "128k",
        str(output_path),
    ]
    r = _run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode == 0 and output_path.exists():
        return output_path
    return None


# ── Helpers for merged FFmpeg pass ───────────────────────────────────────────


def _build_music_extra_inputs(music_path: Path | None) -> list[str]:
    """Return extra -i args if a music file is provided and exists."""
    if not music_path or not Path(music_path).exists():
        return []
    return ["-i", str(music_path)]


def _build_music_af(clip_duration: float, music_path: Path | None,
                     volume: float, trim_start: float, trim_end: float) -> str | None:
    """Build the audio filter string for background music amix, or None."""
    if not music_path or not Path(music_path).exists():
        return None
    has_trim = trim_end > trim_start and trim_end > 0
    if has_trim:
        trim_dur = trim_end - trim_start
        music_part = (
            f"[1:a]atrim=start={trim_start:.3f}:end={trim_end:.3f},asetpts=PTS-STARTPTS,"
            f"aloop=loop=-1:size={int(trim_dur * 48000)},"
            f"atrim=duration={clip_duration:.3f},volume={volume:.2f}[bg]"
        )
    else:
        music_part = (
            f"[1:a]aloop=loop=-1:size={ALOOP_MAX_SIZE},"
            f"atrim=duration={clip_duration:.3f},volume={volume:.2f}[bg]"
        )
    return f"{music_part};[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"


def _build_sub_filter_str(subtitle_path: Path | None, output_stem: str) -> tuple[str | None, Path | None, Path | None]:
    """Prepare subtitle filter string and return (filter_str_with_stream_label, temp_sub, sub_dir)."""
    temp_sub, sub_dir = _prepare_subtitle_file(subtitle_path, output_stem)
    if not temp_sub:
        return None, None, None
    
    filt = _detect_subtitle_filter()
    if not filt:
        return None, None, None
    
    fontsdir_cwd = _fonts_dir_option(sub_dir, use_cwd=True)
    sub_str = f"{filt}={temp_sub.name}{fontsdir_cwd}"
    # Add stream label for filter_complex
    return sub_str, temp_sub, sub_dir


def _run_merged_ffmpeg(
    video_path: Path, start: float, duration: float,
    output_path: Path,
    vf_chain: str | None,
    af_chain: str | None,
    sub_filter_cwd: str | None,
    extra_inputs: list[str],
    preset: str, crf: str, encoder: str, decoder: str,
    gpu_index: int | None = None,
) -> subprocess.CompletedProcess | None:
    """Run a single merged ffmpeg call with optional video/audio filter chains."""
    filter_parts = []
    map_args = []

    if vf_chain:
        filter_parts.append(f"[0:v]{vf_chain}[vout]")
        map_args.extend(["-map", "[vout]"])
    else:
        map_args.extend(["-map", "0:v"])

    if af_chain:
        filter_parts.append(af_chain)
        map_args.extend(["-map", "[aout]"])
        audio_enc = ["-c:a", "aac", "-b:a", "192k"]
    else:
        map_args.extend(["-map", "0:a"])
        audio_enc = ["-c:a", "aac", "-b:a", "192k"]

    filter_complex = ";".join(filter_parts) if filter_parts else None

    cmd = [
        "ffmpeg", "-y", "-ss", str(start),
        *input_hwaccel_args(decoder, gpu_index=gpu_index),
        "-i", str(video_path), "-t", str(duration),
        *extra_inputs,
    ]
    if filter_complex:
        cmd.extend(["-filter_complex", filter_complex])
    cmd.extend([
        *map_args,
        *video_encode_args(preset, crf, encoder, gpu_index=gpu_index),
        *audio_enc,
        str(output_path),
    ])

    kwargs = {}
    if sub_filter_cwd:
        kwargs["cwd"] = sub_filter_cwd

    return run_ffmpeg_with_encode_fallback(
        cmd,
        lambda c: _run(c, capture_output=True, text=True, errors="replace", **kwargs),
        preset, crf,
    )


def _try_sub_filter_approaches(
    video_path: Path, start: float, duration: float,
    output_path: Path, temp_sub: Path, sub_dir: Path,
    sub_filter_str: str, vf_base: str, af_chain: str | None,
    extra_inputs: list[str],
    preset: str, crf: str, encoder: str, decoder: str,
    gpu_index: int | None = None,
) -> bool:
    """Try subtitle burn with multiple approaches in a merged ffmpeg call.
    
    Returns True on success.
    """
    filt = _detect_subtitle_filter()
    other = "ass" if filt == "subtitles" else "subtitles"
    fontsdir_cwd = _fonts_dir_option(sub_dir, use_cwd=True)
    fontsdir_full = _fonts_dir_option(sub_dir, use_cwd=False)

    # Approach 1: filename-only with CWD
    sub_vf = f"{filt}={temp_sub.name}{fontsdir_cwd}"
    vf_chain = f"{vf_base},{sub_vf}" if vf_base else sub_vf
    r = _run_merged_ffmpeg(
        video_path, start, duration, output_path,
        vf_chain, af_chain, str(sub_dir), extra_inputs,
        preset, crf, encoder, decoder, gpu_index=gpu_index,
    )
    if r and r.returncode == 0 and output_path.exists():
        return True

    # Approach 2: full escaped path, no CWD
    escaped = _escape_sub_path_win(temp_sub)
    sub_vf = f"{filt}={escaped}{fontsdir_full}"
    vf_chain = f"{vf_base},{sub_vf}" if vf_base else sub_vf
    r = _run_merged_ffmpeg(
        video_path, start, duration, output_path,
        vf_chain, af_chain, None, extra_inputs,
        preset, crf, encoder, decoder, gpu_index=gpu_index,
    )
    if r and r.returncode == 0 and output_path.exists():
        return True

    # Approach 3: try the other filter
    try:
        r_check = _run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10)
        if re.search(rf'\b{other}\b', r_check.stdout):
            sub_vf = f"{other}={temp_sub.name}{fontsdir_cwd}"
            vf_chain = f"{vf_base},{sub_vf}" if vf_base else sub_vf
            r = _run_merged_ffmpeg(
                video_path, start, duration, output_path,
                vf_chain, af_chain, str(sub_dir), extra_inputs,
                preset, crf, encoder, decoder, gpu_index=gpu_index,
            )
            if r and r.returncode == 0 and output_path.exists():
                return True
    except Exception as e:
        logger.debug("Fallback subtitle filter failed: %s", e)

    return False


# ── Main extract function ────────────────────────────────────────────────────


def extract_clip(
    video_path: Path,
    start: int,
    end: int,
    output_path: Path,
    subtitle_path: Path | None = None,
    crop_params: tuple | None = None,
    preset: str = "ultrafast",
    crf: str = "23",
    encoder: str = "auto",
    decoder: str = "auto",
    shorts_format: str = "crop",  # "crop" | "blur_pad" | "none"
    effect: str | None = None,
    music_path: Path | None = None,
    music_volume: float = 0.12,
    music_trim_start: float = 0,
    music_trim_end: float = 0,
    gpu_index: int | None = None,
) -> ClipResult:
    """Extract a clip, always exporting exact 1080x1920 Shorts video.

    Merges crop, subtitle burn, video effect, and background music into a single
    FFmpeg call for maximum performance. Falls back to multi-pass if needed.

    crop_params can be:
      - (cw, ch, cx, cy)         → static crop (4-tuple)
      - (cw, ch, keyframes_list) → dynamic crop (3-tuple)
    """

    duration = end - start

    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_duration = _probe_duration(video_path)
    if source_duration:
        if start < 0:
            start = 0
        if end > source_duration:
            logger.warning(
                "Clipping end %.3fs exceeds source duration %.3fs; clamping.",
                end, source_duration,
            )
            end = source_duration
        duration = end - start

    if duration <= 0.25:
        warning = f"Invalid clip range: start={start:.3f}s end={end:.3f}s"
        logger.error(warning)
        return ClipResult(path=None, subtitles_burned=False, warning=warning)

    # Ensure input file is not locked (common issue on Windows after download)
    if not wait_for_file_unlock(video_path, timeout=5.0):
        logger.warning("Input file still locked after 5s, proceeding anyway...")

    # Prepare subtitle temp copy
    temp_sub, sub_dir = _prepare_subtitle_file(subtitle_path, output_path.stem)

    logger.info(f"Clipping {fmt_time(start)} -> {fmt_time(end)}  ({duration}s)")

    # Build common filter components
    music_af = _build_music_af(duration, music_path, music_volume, music_trim_start, music_trim_end)
    extra_inputs = _build_music_extra_inputs(music_path)

    # Build video base filter (crop + shorts format + effect)
    effect_vf = None
    if effect and effect in EFFECTS_PRESETS:
        effect_vf = EFFECTS_PRESETS[effect]["vf"]

    if crop_params:
        if shorts_format == "none":
            vf_base = effect_vf
        elif shorts_format == "blur_pad":
            vf_base = _blur_pad_vf()
            if effect_vf:
                vf_base = f"{vf_base},{effect_vf}"
        else:
            vf_base = _chain_vf(_build_crop_vf(crop_params, duration), _shorts_vf(), effect_vf)
    else:
        if shorts_format == "none":
            vf_base = effect_vf
        elif shorts_format == "blur_pad":
            vf_base = _blur_pad_vf()
            if effect_vf:
                vf_base = f"{vf_base},{effect_vf}"
        else:
            vf_base = _chain_vf(_shorts_vf(), effect_vf)

    # ── CASE A: crop + subtitles → try merged single-pass first ──────────
    if crop_params and temp_sub:
        filt = _detect_subtitle_filter()
        if filt:
            sub_str, _, _ = _build_sub_filter_str(subtitle_path, output_path.stem)
            if sub_str:
                merged_ok = _try_sub_filter_approaches(
                    video_path, start, duration, output_path,
                    temp_sub, sub_dir, sub_str, vf_base or "",
                    music_af, extra_inputs,
                    preset, crf, encoder, decoder, gpu_index=gpu_index,
                )
                if merged_ok:
                    _cleanup(temp_sub)
                    logger.info(f"Merged (crop+sub+effect+music) -> {output_path.name}")
                    return _validated_result(output_path)

        # Fallback to 2-pass: crop with effect+music → temp, then burn subs
        temp_cropped = output_path.with_name(output_path.stem + "_tmp_crop.mp4")
        r = _run_merged_ffmpeg(
            video_path, start, duration, temp_cropped,
            vf_base, music_af, None, extra_inputs,
            preset, "18", encoder, decoder, gpu_index=gpu_index,
        )
        if r and r.returncode == 0 and temp_cropped.exists():
            # Pass 2: burn subtitles on cropped file (with copy_audio for speed)
            sub_ok = _try_subtitle_burn(
                temp_cropped, output_path, temp_sub, sub_dir,
                preset, crf, copy_audio=True, encoder=encoder, decoder=decoder,
                gpu_index=gpu_index,
            )
            if sub_ok:
                _cleanup(temp_cropped)
                _cleanup(temp_sub)
                logger.info(f"Saved {output_path.name}")
                return _validated_result(output_path)
            else:
                _rename_safe(temp_cropped, output_path)
                _cleanup(temp_sub)
                print(f"[!] Saved (crop only, no subs): {output_path.name}")
                return _validated_result(
                    output_path,
                    subtitles_burned=False,
                    warning="Subtitle burn failed — ffmpeg may lack libass",
                )
        else:
            _cleanup(temp_cropped)
            _cleanup(temp_sub)
            result = _fallback_shorts_encode(video_path, start, duration, output_path,
                                             preset, crf, encoder, decoder)
            if result:
                return _validated_result(result, subtitles_burned=False, warning="Crop failed")
            return ClipResult(path=None, subtitles_burned=False, warning="Crop failed")

    # ── CASE B: crop only (with optional effect + music) ─────────────────
    elif crop_params:
        r = _run_merged_ffmpeg(
            video_path, start, duration, output_path,
            vf_base, music_af, None, extra_inputs,
            preset, crf, encoder, decoder, gpu_index=gpu_index,
        )
        if r and r.returncode == 0:
            # Apply effect via merged pass already included above
            print(f"[+] Saved {output_path.name}")
            return _validated_result(output_path)
        logger.error(f"Crop failed:\n{(r.stderr[-400:] if r else 'N/A')}")
        result = _fallback_shorts_encode(video_path, start, duration, output_path,
                                         preset, crf, encoder, decoder)
        if result:
            return _validated_result(result)
        return ClipResult(path=None)

    # ── CASE C: subtitles only → try merged single-pass first ────────────
    elif temp_sub:
        filt = _detect_subtitle_filter()
        if filt:
            sub_str, _, _ = _build_sub_filter_str(subtitle_path, output_path.stem)
            if sub_str:
                merged_ok = _try_sub_filter_approaches(
                    video_path, start, duration, output_path,
                    temp_sub, sub_dir, sub_str, vf_base or "",
                    music_af, extra_inputs,
                    preset, crf, encoder, decoder, gpu_index=gpu_index,
                )
                if merged_ok:
                    _cleanup(temp_sub)
                    print(f"[+] Saved {output_path.name}")
                    return _validated_result(output_path)

        # Fallback to 2-pass: extract with effect+music → temp, then burn subs
        temp_input = output_path.with_name(output_path.stem + "_tmp_nosub.mp4")
        r = _run_merged_ffmpeg(
            video_path, start, duration, temp_input,
            vf_base, music_af, None, extra_inputs,
            preset, "18", encoder, decoder, gpu_index=gpu_index,
        )
        if r and r.returncode == 0 and temp_input.exists():
            sub_ok = _try_subtitle_burn(
                temp_input, output_path, temp_sub, sub_dir,
                preset, crf, copy_audio=True, encoder=encoder, decoder=decoder,
                gpu_index=gpu_index,
            )
            _cleanup(temp_input)
            _cleanup(temp_sub)
            if sub_ok:
                print(f"[+] Saved {output_path.name}")
                return _validated_result(output_path)
            else:
                result = _fallback_shorts_encode(video_path, start, duration, output_path,
                                                 preset, crf, encoder, decoder)
                if result:
                    return _validated_result(
                        result,
                        subtitles_burned=False,
                        warning="Subtitle filter failed — check ffmpeg libass support",
                    )
                return ClipResult(
                    path=None,
                    subtitles_burned=False,
                    warning="Subtitle filter failed — check ffmpeg libass support",
                )
        else:
            _cleanup(temp_input)
            _cleanup(temp_sub)
            result = _fallback_shorts_encode(video_path, start, duration, output_path,
                                             preset, crf, encoder, decoder)
            if result:
                return _validated_result(result, subtitles_burned=False, warning="Extract failed")
            return ClipResult(path=None, subtitles_burned=False, warning="Extract failed")

    # ── CASE D: no crop/subtitle filters → single pass with effect+music ─
    r = _run_merged_ffmpeg(
        video_path, start, duration, output_path,
        vf_base, music_af, None, extra_inputs,
        preset, crf, encoder, decoder, gpu_index=gpu_index,
    )
    if r and r.returncode == 0:
        print(f"[+] Saved {output_path.name}")
        return _validated_result(output_path)
    logger.error(f"Shorts encode failed:\n{(r.stderr[-400:] if r else 'N/A')}")
    result = _fallback_shorts_encode(video_path, start, duration, output_path,
                                     preset, crf, encoder, decoder)
    if result:
        return _validated_result(result)
    return ClipResult(path=None)


def extract_audio_clip(video_path: Path, start: float, end: float, output_path: Path) -> Path | None:
    """Extract mono 16 kHz WAV audio for whisper transcription."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(video_path), "-t", str(end - start),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-f", "wav", "-rf64", "auto",
        str(output_path),
    ]
    r = _run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"[!] Audio extraction error:\n{r.stderr[-400:]}")
        return None
    return output_path


# ── Utility helpers ──────────────────────────────────────────────────────────


def _rename_safe(src: Path, dst: Path):
    import time
    # Try a few times to handle Windows file locks (e.g. from the UI preview or OS indexer)
    for i in range(10):
        try:
            if dst.exists():
                dst.unlink()
            src.rename(dst)
            return
        except OSError:
            if i == 9:  # Final attempt
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    logger.debug("shutil.move fallback failed for %s: %s", src, e)
            time.sleep(0.3)


def _robust_unlink(path: Path, retries: int = 5, delay: float = 0.3):
    """Attempt to unlink a file with retries, handling temporary locks."""
    if not path or not path.exists():
        return True
    for i in range(retries):
        try:
            path.unlink()
            return True
        except OSError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                return False
    return False


def _cleanup(path):
    """Wrapper for robust unlink."""
    _robust_unlink(path)





# ── Post-processing: background music ───────────────────────────────────────


def add_background_music(
    clip_path: Path,
    music_path: Path,
    volume: float = 0.12,
    trim_start: float = 0,
    trim_end: float = 0,
) -> bool:
    """Mix background music into a clip at the given volume level.

    - music_path: path to an audio file (mp3/wav/aac)
    - volume: 0.0-1.0, default 0.12 (12% = subtle background)
    - trim_start/trim_end: use only this portion of the music file (seconds).
      If both are 0 or trim_end <= trim_start, uses the full track.
    - The trimmed selection is looped if shorter than the clip
    - The original audio is kept at full volume
    - Overwrites the clip in-place

    Returns True on success.
    """
    clip_path = Path(clip_path).resolve()
    music_path = Path(music_path).resolve()

    if not clip_path.exists() or not music_path.exists():
        logger.warning(f"Music mix: missing file (clip={clip_path.exists()}, music={music_path.exists()})")
        return False

    temp_out = clip_path.with_name(clip_path.stem + "_music_tmp.mp4")

    # Get clip duration
    try:
        r = _run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(clip_path)],
            capture_output=True, text=True, timeout=10,
        )
        clip_dur = float(r.stdout.strip())
    except Exception:
        clip_dur = 60

    # Build audio filter for music input:
    # 1. If trimming, first seek + trim to the selected portion
    # 2. Loop the (trimmed) audio to fill the clip duration
    # 3. Apply volume
    has_trim = trim_end > trim_start and trim_end > 0
    music_filter_parts = []

    if has_trim:
        trim_duration = trim_end - trim_start
        # atrim to extract the selected portion, then asetpts to reset timestamps
        music_filter_parts.append(
            f"[1:a]atrim=start={trim_start:.3f}:end={trim_end:.3f},asetpts=PTS-STARTPTS"
        )
        # Loop the trimmed portion to fill clip duration
        music_filter_parts.append(
            f"aloop=loop=-1:size={int(trim_duration * 48000)},"
            f"atrim=duration={clip_dur:.3f},volume={volume:.2f}[bg]"
        )
        af_music = ",".join(music_filter_parts)
    else:
        # No trim — loop the full track
        af_music = (
            f"[1:a]aloop=loop=-1:size={ALOOP_MAX_SIZE},"
            f"atrim=duration={clip_dur:.3f},volume={volume:.2f}[bg]"
        )

    af = f"{af_music};[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-i", str(music_path),
        "-filter_complex", af,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(temp_out),
    ]

    trim_info = f", trim {trim_start:.1f}-{trim_end:.1f}s" if has_trim else ""
    logger.info(f"Mixing background music ({volume:.0%} vol{trim_info})...")
    r = _run(cmd, capture_output=True, text=True, errors="replace")

    if r.returncode == 0 and temp_out.exists():
        _rename_safe(temp_out, clip_path)
        logger.info(f"Background music added to {clip_path.name}")
        return True
    else:
        print(f"[!] Music mix failed:\n{r.stderr[-400:]}")
        _cleanup(temp_out)
        return False


# ── Post-processing: video effects ──────────────────────────────────────────

# Available effects presets
EFFECTS_PRESETS = {
    "none": {
        "label": "No Effects",
        "desc": "Clean original look",
        "vf": None,
    },
    "cinematic": {
        "label": "Cinematic",
        "desc": "Slight contrast boost + warm tones",
        "vf": "eq=contrast=1.08:brightness=0.02:saturation=1.15",
    },
    "vibrant": {
        "label": "Vibrant",
        "desc": "Vivid colors + sharpness",
        "vf": "eq=saturation=1.35:contrast=1.05,unsharp=3:3:1.0",
    },
    "moody": {
        "label": "Moody",
        "desc": "Dark cinematic with crushed blacks",
        "vf": "eq=contrast=1.2:brightness=-0.03:saturation=0.85,curves=m='0/0.05 0.5/0.45 1/0.95'",
    },
    "vintage": {
        "label": "Vintage",
        "desc": "Warm retro film look",
        "vf": "eq=saturation=0.75:contrast=1.1:brightness=0.03,colorbalance=rs=0.08:gs=0.02:bs=-0.06",
    },
    "bright": {
        "label": "Bright & Clean",
        "desc": "Boosted brightness + light feel",
        "vf": "eq=brightness=0.06:contrast=1.05:saturation=1.1",
    },
    "bw": {
        "label": "Black & White",
        "desc": "Classic monochrome with contrast",
        "vf": "eq=saturation=0:contrast=1.15",
    },
    "streamer": {
        "label": "Streamer",
        "desc": "Punchy vibrant look — boosted saturation and sharpness for game feeds",
        "vf": "eq=saturation=1.30:contrast=1.10:brightness=0.02,unsharp=5:5:0.8",
    },
    "hdr": {
        "label": "Game HDR",
        "desc": "Expanded contrast range with vibrance for gameplay footage",
        "vf": "eq=saturation=1.15:contrast=1.20:brightness=0.01,curves=m='0/0.02 0.3/0.35 0.7/0.7 1/1.0'",
    },
}


def apply_video_effect(
    clip_path: Path,
    effect: str = "none",
    preset: str = "ultrafast",
    crf: str = "23",
    encoder: str = "auto",
    decoder: str = "auto",
    gpu_index: int | None = None,
) -> bool:
    """Apply a video effect preset to a clip (in-place).

    effect: key from EFFECTS_PRESETS ('cinematic', 'vibrant', etc.)
    Returns True on success.
    """
    if effect == "none" or effect not in EFFECTS_PRESETS:
        return True

    vf = EFFECTS_PRESETS[effect]["vf"]
    if not vf:
        return True

    clip_path = Path(clip_path).resolve()
    if not clip_path.exists():
        return False

    temp_out = clip_path.with_name(clip_path.stem + "_fx_tmp.mp4")

    cmd = [
        "ffmpeg", "-y",
        *input_hwaccel_args(decoder, gpu_index=gpu_index),
        "-i", str(clip_path),
        "-vf", vf,
        *video_encode_args(preset, crf, encoder, gpu_index=gpu_index),
        "-c:a", "copy",
        str(temp_out),
    ]

    logger.info(f"Applying '{effect}' effect...")
    r = run_ffmpeg_with_encode_fallback(
        cmd,
        lambda c: _run(c, capture_output=True, text=True, errors="replace"),
        preset, crf,
    )

    if r.returncode == 0 and temp_out.exists():
        _rename_safe(temp_out, clip_path)
        logger.info(f"Effect '{effect}' applied to {clip_path.name}")
        return True
    else:
        print(f"[!] Effect failed:\n{r.stderr[-400:]}")
        _cleanup(temp_out)
        return False


def get_effects_list() -> list[dict]:
    """Return list of available effects for the UI."""
    return [
        {"id": k, "label": v["label"], "desc": v["desc"]}
        for k, v in EFFECTS_PRESETS.items()
    ]
