#!/usr/bin/env python3
"""
ViriaRevive  –  Viral Clip Generator

  Downloads a YouTube video, finds the most engaging moments (no AI –
  pure audio-energy + scene-change analysis), adds TikTok-style
  word-by-word subtitles, and optionally schedules uploads to YouTube.

Usage:
  python main.py "https://youtube.com/watch?v=VIDEO_ID"
  python main.py "URL" --clips 3 --duration 45 --style bold
  python main.py "URL" --upload --schedule 12
  python main.py "URL" --auto-clips --effect streamer --music bg.mp3
"""

import argparse
import logging
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from config import (
    AI_DETECTOR_MODE,
    CLIPS_DIR,
    CLIP_DURATION,
    CROP_VERTICAL,
    FFMPEG_PRESET,
    MIN_GAP,
    NUM_CLIPS,
    OLLAMA_DETECTOR_CANDIDATE_MULTIPLIER,
    OLLAMA_DETECTOR_MODEL,
    OLLAMA_DETECTOR_TIMEOUT,
    SENTENCE_BUFFER,
    SUBTITLE_STYLE,
    SUBTITLES_DIR,
    TRANSLATE_TARGET,
    TRANSLATE_MODEL,
    VIDEO_CRF,
    VIDEO_ENCODER,
    VIDEO_DECODER,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    YOLO_DEVICE,
)
from hwaccel import log_hardware_startup, get_gpu_count, select_least_loaded_gpu
from pipeline_cache import PipelineCache
from utils import auto_clip_count
from subprocess_utils import run as _run, is_cancelled, CancelledError, reset_cancel


# ── Progress callback type ──────────────────────────────────────────────────

ProgressCB = Optional[Callable[[str, int, str], None]]
"""on_progress(step: str, pct: int, msg: str)"""


def _default_progress(step: str, pct: int, msg: str):
    """Print progress to stdout (used when no callback is provided)."""
    if pct == 0:
        print(f"\n══ {step} · {msg} ══")
    elif pct < 100:
        print(f"  [{step}] {msg}")
    else:
        print(f"[+] {msg}")


def _check_deps():
    if not shutil.which("ffmpeg"):
        print("[!] ffmpeg not found – install from https://ffmpeg.org/download.html")
        sys.exit(1)


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        r = _run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        return float((r.stdout or "0").strip().split()[0])
    except Exception as e:
        logger.warning("Failed to probe video duration: %s — defaulting to 600s", e)
        return 600.0


def process(
    url: str,
    num_clips: int = NUM_CLIPS,
    clip_duration: int = CLIP_DURATION,
    style: str = SUBTITLE_STYLE,
    model: str = WHISPER_MODEL,
    language: str = WHISPER_LANGUAGE,
    title_language: str = None,
    upload: bool = False,
    schedule_hours: int = 24,
    crop: bool = CROP_VERTICAL,
    ai_detector: str = AI_DETECTOR_MODE,
    auto_clips: bool = False,
    effect: str = None,
    music_path: str = None,
    music_volume: float = 0.12,
    music_start: float = 0.0,
    music_end: float = 0.0,
    preset: str = FFMPEG_PRESET,
    crf: str = VIDEO_CRF,
    encoder: str = VIDEO_ENCODER,
    decoder: str = VIDEO_DECODER,
    sentence_buffer: float = SENTENCE_BUFFER,
    ollama_model: str = OLLAMA_DETECTOR_MODEL,
    translate_to: str | None = None,
    translate_model: str = TRANSLATE_MODEL,
    resume: bool = True,
    on_progress: ProgressCB = None,
) -> list[Path]:
    """Full pipeline: download → detect → clip → subtitle → titles → upload.

    Args:
        url: YouTube URL or local file path.
        num_clips: Number of clips to generate.
        clip_duration: Duration of each clip in seconds.
        style: Subtitle style ('tiktok', 'clean', 'bold', 'game').
        model: Whisper model size.
        language: Language code for transcription.
        title_language: Language for AI title generation.
        upload: Whether to upload to YouTube.
        schedule_hours: Hours between scheduled uploads.
        crop: Whether to apply YOLO person-detection cropping.
        ai_detector: AI detector mode ('auto', 'off', 'on').
        auto_clips: Auto-compute clip count from video duration.
        effect: Video effect preset name (e.g. 'streamer', 'hdr').
        music_path: Path to background music file.
        music_volume: Background music volume (0.0-1.0).
        music_start: Trim start for music (seconds).
        music_end: Trim end for music (seconds).
        preset: FFmpeg encoding preset.
        crf: FFmpeg CRF value.
        encoder: Video encoder ('nvenc', 'qsv', 'amf', 'cpu').
        decoder: Video decoder ('cuda', 'd3d11va', etc.).
        sentence_buffer: Extra seconds to extend audio for sentence boundary detection.
        ollama_model: Ollama model name for AI reranking & title gen.
        resume: Whether to attempt resume from previous pipeline state.
        on_progress: Callback(step, pct, msg) for progress updates.

    Returns:
        List of paths to completed clip files.
    """
    _check_deps()
    reset_cancel()
    log_hardware_startup(encoder, YOLO_DEVICE, WHISPER_DEVICE)

    # Lazy imports (heavy deps: numpy, yt-dlp, ultralytics, etc.)
    from detector import find_viral_moments
    from ollama_detector import detector_ready, rerank_moments
    from transcriber import transcribe_clip, find_sentence_boundary
    from subtitler import generate_subtitles
    from title_generator import generate_titles_batch
    from translator import translate_words
    from clipper import extract_clip, extract_audio_clip, validate_shorts_output
    from cropper import get_crop_params_dynamic, detect_all_persons
    from uploader import upload_to_youtube, build_schedule

    if on_progress is None:
        on_progress = _default_progress

    if not crop:
        print("[shorts] Output will preserve original aspect ratio")

    # ── 1. Download ──────────────────────────────────────────────────────
    on_progress("download", 0, "Downloading video...")
    from downloader import download_video
    video_path = download_video(url)
    if video_path is None:
        print("[!] Download failed — aborting")
        return []
    print(f"[+] {video_path}")
    stem = video_path.stem[:50]

    cache = PipelineCache(stem)

    on_progress("download", 100, f"Downloaded: {video_path.name}")

    # ── Video duration + auto clips ─────────────────────────────────────
    vid_duration = _get_video_duration(video_path)
    if auto_clips:
        old_n = num_clips
        num_clips = auto_clip_count(vid_duration, clip_duration)
        print(f"[auto-clips] {old_n} → {num_clips} clips (video: {vid_duration:.0f}s)")

    # Save download state (after auto_clips may have updated num_clips)
    if resume:
        state = cache.load_state()
        state.url = url
        state.stem = stem
        state.step_downloaded = True
        state.num_clips = num_clips
        state.clip_duration = clip_duration
        cache.save_state(state)

    # ── 2. Detect viral moments ──────────────────────────────────────────
    on_progress("detect", 0, "Finding viral moments...")
    ai_ready = False
    candidate_count = num_clips
    if ai_detector != "off":
        ai_ready = detector_ready(ollama_model)
        if ai_ready:
            candidate_count = num_clips * OLLAMA_DETECTOR_CANDIDATE_MULTIPLIER
            print(f"[*] AI detector ready; scanning {candidate_count} candidates")
        elif ai_detector == "on":
            print("[ai-detector] Ollama detector requested but unavailable; using heuristic detector")

    moments = find_viral_moments(
        video_path, num_clips=candidate_count, clip_duration=clip_duration, min_gap=MIN_GAP
    )
    if not moments:
        print("[!] Nothing found – try a longer video or lower --clips")
        return []

    on_progress("detect", 30, f"Found {len(moments)} candidate moments")

    if ai_ready and len(moments) > num_clips:
        on_progress("detect", 35, f"Processing {len(moments)} AI candidates (parallel)...")

        # ── Parallel candidate processing ───────────────────────────────
        candidate_lock = threading.Lock()
        candidates: list[dict] = []

        _candidate_wavs: list[Path] = []

        def _process_candidate(args):
            idx, m = args
            if is_cancelled():
                raise CancelledError("Candidate processing cancelled")
            wav = SUBTITLES_DIR / f"{stem}_candidate{idx}.wav"
            try:
                if extract_audio_clip(video_path, m["start"], m["end"], wav):
                    words = transcribe_clip(
                        wav, model_size=model, language=language, device_pref=WHISPER_DEVICE,
                    )
                    m["transcript"] = " ".join(
                        w.get("word", w.get("text", "")) for w in words
                    ).strip()

                if crop:
                    # Lower sample count for candidate visual check (binary person detection)
                    detections, _, _ = detect_all_persons(
                        video_path, m["start"], m["end"], 1920, 1080,
                        sample_count=10, yolo_device=YOLO_DEVICE,
                    )
                    if detections:
                        hits = sum(1 for _, persons in detections if persons)
                        m["visual_score"] = hits / len(detections)
                    else:
                        m["visual_score"] = 0.0
                else:
                    m["visual_score"] = 1.0

                with candidate_lock:
                    candidates.append(m)
                    _candidate_wavs.append(wav)
                print(f"  [candidate {idx}/{len(moments)}] done")
                on_progress("detect", 35 + int((idx - 1) / max(1, len(moments)) * 35),
                            f"AI candidate {idx}/{len(moments)}")
            except Exception:
                wav.unlink(missing_ok=True)
                raise

        worker_count = max(1, get_gpu_count() or 1)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futs = {pool.submit(_process_candidate, (i + 1, m)): i for i, m in enumerate(moments)}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except CancelledError:
                    print("[!] Candidate processing cancelled")
                    return []
                except Exception:
                    logger.exception("Candidate processing failed")

        # ── Rerank with AI ─────────────────────────────────────────────
        if candidates:
            on_progress("detect", 72, "Reranking candidates with AI...")
            ranked = rerank_moments(
                candidates,
                clip_duration=clip_duration,
                keep=num_clips,
                model=ollama_model,
                timeout=OLLAMA_DETECTOR_TIMEOUT,
                on_progress=lambda done, total, score: print(
                    f"[ai-detector] {done}/{total}: {'scored' if score else 'skipped'}"
                ),
            )
            if ranked is not None:
                moments = ranked
                print(f"[ai-detector] Selected {len(moments)} clips with Ollama reranking")
            else:
                moments = moments[:num_clips]
                print("[ai-detector] No valid AI scores; using heuristic candidates")
        else:
            moments = moments[:num_clips]
            print("[ai-detector] No successful candidate transcriptions; using heuristic")
    else:
        moments = moments[:num_clips]

    on_progress("detect", 100, f"Found {len(moments)} viral moments")

    # Save moments to cache for resume
    if resume:
        state = cache.load_state()
        state.step_detected = True
        state.step_reranked = ai_ready
        cache.set_moments(moments)
        cache.save_state(state)

    # ── 3. Clip + subtitle each moment (parallel multi-GPU) ────────────
    on_progress("clips", 0, f"Processing {len(moments)} clips...")
    gpu_count = get_gpu_count()
    if gpu_count > 0:
        print(f"[+] {gpu_count} GPU(s) detected — processing clips in parallel")
    else:
        print("[+] CPU mode — processing clips sequentially")

    def _process_one_clip(idx: int, m: dict) -> Path | None:
        """Process a single clip, assigned to a specific GPU."""
        if is_cancelled():
            return None
        is_multi = gpu_count > 0
        gpu_idx = select_least_loaded_gpu(list(range(gpu_count))) if is_multi else None
        clip_num = idx + 1

        # Resume: skip if clip already has valid output
        out = CLIPS_DIR / f"{stem}_viral{clip_num}.mp4"
        if resume and out.exists() and validate_shorts_output(out):
            print(f"  [+] Clip {clip_num}/{len(moments)} already done, skipping")
            on_progress("clips", int((idx + 1) / len(moments) * 100),
                        f"Clip {clip_num}/{len(moments)}: already done")
            # Backfill transcript for title generation on resume path
            wav = SUBTITLES_DIR / f"{stem}_c{clip_num}.wav"
            if wav.exists():
                words = transcribe_clip(
                    wav, model_size=model, language=language,
                    device_pref=WHISPER_DEVICE,
                )
                m["transcript"] = " ".join(w.get("word", w.get("text", "")) for w in words).strip()
            return out

        print(f"\n── clip {clip_num}/{len(moments)} {'GPU ' + str(gpu_idx) if gpu_idx is not None else ''}──")
        start, end = m["start"], m["end"]

        # 3a. compute dynamic crop params for 9:16
        crop_params = None
        if crop:
            on_progress("clips", 0, f"Clip {clip_num}/{len(moments)}: Tracking speakers...")
            try:
                crop_params = get_crop_params_dynamic(
                    video_path, start, end, yolo_device=YOLO_DEVICE, debug_frames=True,
                    gpu_index=gpu_idx,
                )
                if crop_params:
                    pass
            except Exception as e:
                print(f"[!] Crop detection failed for clip {clip_num}: {e}")
                crop_params = None

        # 3b. extract wav for whisper (try candidate cache first, then cached clip wav)
        wav = SUBTITLES_DIR / f"{stem}_c{clip_num}.wav"
        extended_end = min(end + sentence_buffer, int(vid_duration))
        _wav_from_cache = False
        if not wav.exists():
            # Check if any candidate WAV covers this range
            for cw in _candidate_wavs:
                cw_stem = cw.stem
                cw_idx_str = cw_stem.replace(f"{stem}_candidate", "")
                try:
                    cw_idx = int(cw_idx_str) - 1
                    if 0 <= cw_idx < len(moments):
                        cm = moments[cw_idx]
                        if abs(cm["start"] - start) < 1.0 and cm["end"] >= extended_end - 1.0:
                            shutil.copy2(cw, wav)
                            _wav_from_cache = True
                            print(f"  [cache] Reusing candidate WAV for clip {clip_num}")
                            break
                except (ValueError, IndexError):
                    pass
            if not _wav_from_cache:
                if not extract_audio_clip(video_path, start, extended_end, wav):
                    return None
        else:
            print(f"  [cache] Using cached audio for clip {clip_num}")

        # 3c. transcribe → word timestamps
        on_progress("clips", 0, f"Clip {clip_num}/{len(moments)}: Transcribing...")
        words = transcribe_clip(
            wav, model_size=model, language=language,
            device_pref=WHISPER_DEVICE, gpu_index=gpu_idx,
        )

        # 3c.1. find natural sentence boundary
        original_duration = end - start
        new_end = end
        if sentence_buffer > 0:
            new_duration = find_sentence_boundary(
                words,
                clip_duration=float(original_duration),
                min_keep=0.60,
                max_extend=float(sentence_buffer),
            )
            if new_duration is not None:
                new_end = start + int(new_duration + 0.5)
                words = [w for w in words if w["end"] <= (new_end - start) + 0.1]
                print(f"    [sentence] Snapped clip to {new_end - start}s (was {original_duration}s)")
            else:
                words = [w for w in words if w["end"] <= original_duration + 0.1]
        else:
            words = [w for w in words if w["end"] <= original_duration + 0.1]

        # 3c.2. translate words to target language (if requested)
        if translate_to and words:
            on_progress("clips", 0, f"Clip {clip_num}/{len(moments)}: Translating...")
            print(f"  [translate] Translating {len(words)} words to {translate_to}")
            words = translate_words(words, translate_to, model=translate_model)

        m["end"] = new_end
        m["duration"] = new_end - start
        m["transcript"] = " ".join(w.get("word", w.get("text", "")) for w in words).strip()

        # 3d. build ASS subtitles (sized for cropped resolution)
        on_progress("clips", 0, f"Clip {clip_num}/{len(moments)}: Generating subtitles...")
        ass = SUBTITLES_DIR / f"{stem}_c{clip_num}.ass"
        generate_subtitles(words, ass, video_width=1080, video_height=1920, style=style)

        # 3e. extract clip + crop + burn subs + effects + music (single ffmpeg pass)
        on_progress("clips", 0, f"Clip {clip_num}/{len(moments)}: Rendering...")
        resolved_music = None
        if music_path:
            mp = Path(music_path)
            if mp.exists():
                resolved_music = mp

        result = extract_clip(
            video_path, start, new_end, out,
            subtitle_path=ass if words else None,
            crop_params=crop_params,
            preset=preset,
            crf=crf,
            encoder=encoder,
            decoder=decoder,
            effect=effect,
            music_path=resolved_music,
            music_volume=music_volume,
            music_trim_start=music_start,
            music_trim_end=music_end,
            gpu_index=gpu_idx,
        )

        # cleanup temp wav (only if not needed for future)
        wav.unlink(missing_ok=True)

        if result and result.path:
            # Mark clip as done in cache
            if resume:
                cache.mark_clip_done(clip_num)
            on_progress("clips", int((idx + 1) / len(moments) * 100),
                        f"Clip {clip_num}/{len(moments)} complete")
            return result.path

        on_progress("clips", int((idx + 1) / len(moments) * 100),
                    f"Clip {clip_num}/{len(moments)} failed")
        return None

    done: list[Path] = []
    if gpu_count > 0:
        futures = {}
        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            for idx, m in enumerate(moments):
                if is_cancelled():
                    print("[!] Pipeline cancelled")
                    return done
                futures[executor.submit(_process_one_clip, idx, m)] = idx
            ordered = [None] * len(moments)
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    ordered[i] = fut.result()
                except CancelledError:
                    print("[!] Pipeline cancelled")
                    return done
                except Exception:
                    logger.exception("Clip %d failed", i + 1)
                    ordered[i] = None
        done = [p for p in ordered if p is not None]
    else:
        for idx, m in enumerate(moments):
            if is_cancelled():
                print("[!] Pipeline cancelled")
                return done
            p = _process_one_clip(idx, m)
            if p:
                done.append(p)

    # Clean up candidate WAVs now that all clips are processed
    for cw in _candidate_wavs:
        cw.unlink(missing_ok=True)

    on_progress("clips", 100, f"{len(done)} clips created")

    # ── 4. Generate AI Titles ───────────────────────────────────────────
    all_titles = []
    if done:
        on_progress("titles", 0, "Generating AI titles...")
        # Only collect transcripts for moments that actually completed
        _path_to_idx = {}
        _prefix = f"{stem}_viral"
        for p in done:
            name = p.name
            if name.endswith(".mp4") and name.startswith(_prefix):
                num_str = name[len(_prefix):-4]
                try:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(moments):
                        _path_to_idx[p] = idx
                except ValueError:
                    pass
        transcripts = [moments[_path_to_idx[p]].get("transcript", "") for p in done if p in _path_to_idx]
        if any(transcripts):
            all_titles = generate_titles_batch(
                transcripts, model=ollama_model, language=title_language or language
            )
        else:
            print("[title-gen] No transcripts found; using default titles")
        on_progress("titles", 100, f"{len(all_titles)} titles generated")

    print(f"\n══ Done! {len(done)} clips ══")
    for p in done:
        print(f"  → {p}")

    # ── 5. Upload / schedule ─────────────────────────────────────────────
    if upload and done:
        on_progress("upload", 0, f"Uploading {len(done)} clips...")
        sched = build_schedule(
            done,
            start_time=datetime.now().astimezone() + timedelta(hours=1),
            interval_hours=schedule_hours,
        )
        for i, item in enumerate(sched):
            try:
                idx = done.index(item["path"]) + 1
            except ValueError:
                idx = 0
            pct = int((i + 1) / len(sched) * 100)
            on_progress("upload", pct, f"Uploading clip {i + 1}/{len(sched)}...")
            upload_to_youtube(
                item["path"],
                title=all_titles[idx - 1] if (all_titles and 0 <= idx - 1 < len(all_titles)) else f"{stem} – Viral Clip #{idx}",
                description=f"Viral clip from {stem}\n\n#shorts #viral",
                scheduled_time=item["scheduled_time"],
            )
        on_progress("upload", 100, "Upload complete")

    return done


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="ViriaRevive – viral clip generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s \"https://youtube.com/watch?v=ID\"\n"
            "  %(prog)s \"URL\" --clips 3 --duration 45 --style bold\n"
            "  %(prog)s \"URL\" --auto-clips --effect streamer --music bg.mp3\n"
            "  %(prog)s \"URL\" --upload --schedule 12\n"
        ),
    )
    p.add_argument("url", help="YouTube video URL or local file path")
    p.add_argument("-n", "--clips",    type=int, default=None,    help="number of clips")
    p.add_argument("-d", "--duration", type=int, default=CLIP_DURATION, help=f"clip length in seconds  (default {CLIP_DURATION})")
    p.add_argument("-s", "--style",    choices=["tiktok", "clean", "bold", "game"], default=SUBTITLE_STYLE, help="subtitle style")
    p.add_argument("-m", "--model",    choices=["tiny", "base", "small", "medium", "large-v3"], default=WHISPER_MODEL, help="whisper model size")
    p.add_argument("-l", "--language", default=WHISPER_LANGUAGE, help="force language (en, es, fr …)")
    p.add_argument("--title-language", default=None, help="force AI title language (Spanish, French …)")
    p.add_argument("-u", "--upload",   action="store_true", help="upload clips to YouTube")
    p.add_argument("--schedule",       type=int, default=24, help="hours between scheduled uploads")
    p.add_argument("--no-crop",        action="store_true", help="skip YOLO person-detection cropping (uses center crop)")
    p.add_argument("--ai-detector",    choices=["auto", "off", "on"], default=AI_DETECTOR_MODE, help="local Ollama AI detector mode")
    p.add_argument("--ollama-model",   default=OLLAMA_DETECTOR_MODEL, help=f"Ollama model name  (default {OLLAMA_DETECTOR_MODEL})")
    p.add_argument("--translate",      default=None, help="translate subtitles to language (es, fr, de, ja, ...)")

    # ── New options ────────────────────────────────────────────────────
    p.add_argument("--auto-clips",     action="store_true", help="auto-compute clip count from video duration")
    p.add_argument("--effect",         default=None, help="video effect preset (cinematic, vibrant, moody, streamer, hdr, …)")
    p.add_argument("--music",          default=None, help="path to background music file")
    p.add_argument("--music-volume",   type=float, default=0.12, help="background music volume 0-1  (default 0.12)")
    p.add_argument("--music-start",    type=float, default=0.0, help="trim start for music (seconds)")
    p.add_argument("--music-end",      type=float, default=0.0, help="trim end for music (seconds)")
    p.add_argument("--preset",         default=FFMPEG_PRESET, help=f"ffmpeg preset  (default {FFMPEG_PRESET})")
    p.add_argument("--crf",            default=VIDEO_CRF, help=f"ffmpeg CRF  (default {VIDEO_CRF})")
    p.add_argument("--encoder",        default=VIDEO_ENCODER, help=f"video encoder  (default {VIDEO_ENCODER})")
    p.add_argument("--decoder",        default=VIDEO_DECODER, help=f"video decoder  (default {VIDEO_DECODER})")
    p.add_argument("--sentence-buffer", type=float, default=SENTENCE_BUFFER,
                    help=f"extra seconds for sentence-boundary detection  (default {SENTENCE_BUFFER})")
    p.add_argument("--no-resume",      action="store_true", help="disable pipeline resume (always re-process)")

    a = p.parse_args()

    # Resolve clip count: auto overrides explicit
    explicit_clips = a.clips if a.clips is not None else NUM_CLIPS

    process(
        url=a.url,
        num_clips=explicit_clips,
        clip_duration=a.duration,
        style=a.style,
        model=a.model,
        language=a.language,
        title_language=a.title_language,
        upload=a.upload,
        schedule_hours=a.schedule,
        crop=not a.no_crop,
        ai_detector=a.ai_detector,
        ollama_model=a.ollama_model,
        translate_to=a.translate or TRANSLATE_TARGET,
        auto_clips=a.auto_clips,
        effect=a.effect,
        music_path=a.music,
        music_volume=a.music_volume,
        music_start=a.music_start,
        music_end=a.music_end,
        preset=a.preset,
        crf=a.crf,
        encoder=a.encoder,
        decoder=a.decoder,
        sentence_buffer=a.sentence_buffer,
        resume=not a.no_resume,
    )


if __name__ == "__main__":
    main()

