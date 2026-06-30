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
"""

import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

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
from hwaccel import log_hardware_startup, get_gpu_count, select_least_loaded_gpu
from downloader import download_video
from detector import find_viral_moments
from ollama_detector import detector_ready, rerank_moments
from transcriber import transcribe_clip
from subtitler import generate_subtitles
from title_generator import generate_titles_batch
from clipper import extract_clip, extract_audio_clip
from cropper import get_crop_params, get_dimensions, detect_all_persons
from uploader import upload_to_youtube, build_schedule


def _check_deps():
    if not shutil.which("ffmpeg"):
        print("[!] ffmpeg not found – install from https://ffmpeg.org/download.html")
        sys.exit(1)


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
):
    _check_deps()
    log_hardware_startup(VIDEO_ENCODER, YOLO_DEVICE, WHISPER_DEVICE)
    if not crop:
        print("[shorts] Output will preserve original aspect ratio")

    # ── 1. Download ──────────────────────────────────────────────────────
    print("\n══ 1 · Downloading video ══")
    video_path = download_video(url)
    if video_path is None:
        print("[!] Download failed — aborting")
        return []
    print(f"[+] {video_path}")
    stem = video_path.stem[:50]

    # ── 2. Detect viral moments ──────────────────────────────────────────
    print("\n══ 2 · Finding viral moments ══")
    ai_ready = False
    candidate_count = num_clips
    if ai_detector != "off":
        ai_ready = detector_ready(OLLAMA_DETECTOR_MODEL)
        if ai_ready:
            candidate_count = max(
                num_clips,
                num_clips * OLLAMA_DETECTOR_CANDIDATE_MULTIPLIER,
            )
            print(f"[*] AI detector ready; scanning {candidate_count} candidates")
        elif ai_detector == "on":
            print("[ai-detector] Ollama detector requested but unavailable; using heuristic detector")

    moments = find_viral_moments(
        video_path, num_clips=candidate_count, clip_duration=clip_duration, min_gap=MIN_GAP
    )
    if not moments:
        print("[!] Nothing found – try a longer video or lower --clips")
        return []

    if ai_ready and len(moments) > num_clips:
        print("[*] Transcribing AI detector candidates...")
        candidates = []
        for idx, m in enumerate(moments, 1):
            wav = SUBTITLES_DIR / f"{stem}_candidate{idx}.wav"
            try:
                # Transcribe
                if extract_audio_clip(video_path, m["start"], m["end"], wav):
                    words = transcribe_clip(
                        wav, model_size=model, language=language, device_pref=WHISPER_DEVICE,
                    )
                    m["transcript"] = " ".join(
                        w.get("word", w.get("text", "")) for w in words
                    ).strip()
                
                # Visual check (person detection)
                if crop:
                    print(f"[*] Visual scanning candidate {idx}/{len(moments)}...")
                    detections, _, _ = detect_all_persons(
                        video_path, m["start"], m["end"], 1920, 1080, sample_count=30, yolo_device=YOLO_DEVICE
                    )
                    # Score = fraction of frames where at least one person was detected
                    # detections is list of (time, [(hx, hy, area, conf, h), ...])
                    if detections:
                        hits = sum(1 for _, persons in detections if persons)
                        m["visual_score"] = hits / len(detections)
                    else:
                        m["visual_score"] = 0.0
                else:
                    m["visual_score"] = 1.0 # default to perfect if crop is off
                
                candidates.append(m)
            finally:
                wav.unlink(missing_ok=True)

        ranked = rerank_moments(
            candidates,
            clip_duration=clip_duration,
            keep=num_clips,
            model=OLLAMA_DETECTOR_MODEL,
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

    # ── 3. Clip + subtitle each moment (parallel multi-GPU) ────────────
    print("\n══ 3 · Creating clips with subtitles ══")
    gpu_count = get_gpu_count()
    if gpu_count > 0:
        print(f"[+] {gpu_count} GPU(s) detected — processing clips in parallel")
    else:
        print("[+] CPU mode — processing clips sequentially")

    def _process_one_clip(idx: int, m: dict) -> Path | None:
        """Process a single clip, assigned to a specific GPU."""
        is_multi = gpu_count > 0
        if is_multi:
            gpu_idx = select_least_loaded_gpu(list(range(gpu_count)))
        else:
            gpu_idx = None
        clip_num = idx + 1

        print(f"\n── clip {clip_num}/{len(moments)} {'GPU ' + str(gpu_idx) if gpu_idx is not None else ''}──")
        start, end = m["start"], m["end"]

        # 3a. compute crop params for 9:16
        crop_params = None
        vid_w, vid_h = get_dimensions(video_path)
        if crop:
            crop_params = get_crop_params(video_path, start, end)
            if crop_params:
                vid_w, vid_h = crop_params[0], crop_params[1]

        # 3b. extract wav for whisper
        wav = SUBTITLES_DIR / f"{stem}_c{clip_num}.wav"
        if not extract_audio_clip(video_path, start, end, wav):
            return None

        # 3c. transcribe → word timestamps (pinned to assigned GPU)
        words = transcribe_clip(
            wav, model_size=model, language=language,
            device_pref=WHISPER_DEVICE, gpu_index=gpu_idx,
        )

        # 3d. build ASS subtitles (sized for cropped resolution)
        ass = SUBTITLES_DIR / f"{stem}_c{clip_num}.ass"
        generate_subtitles(words, ass, video_width=1080, video_height=1920, style=style)

        # 3e. extract clip + crop + burn subs (single ffmpeg pass, pinned to GPU)
        out = CLIPS_DIR / f"{stem}_viral{clip_num}.mp4"
        result = extract_clip(
            video_path, start, end, out,
            subtitle_path=ass if words else None,
            crop_params=crop_params,
            preset=FFMPEG_PRESET,
            crf=VIDEO_CRF,
            encoder=VIDEO_ENCODER,
            decoder=VIDEO_DECODER,
            gpu_index=gpu_idx,
        )

        # cleanup temp wav
        wav.unlink(missing_ok=True)

        if result and result.path:
            return result.path
        return None

    done: list[Path] = []
    if gpu_count > 0:
        # Parallel dispatch across GPUs
        futures = {}
        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            for idx, m in enumerate(moments):
                futures[executor.submit(_process_one_clip, idx, m)] = idx
            ordered = [None] * len(moments)
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    ordered[i] = fut.result()
                except Exception:
                    logger.exception(f"Clip {i + 1} failed")
        done = [p for p in ordered if p is not None]
    else:
        # Sequential (single GPU or CPU) — preserves existing behaviour
        for idx, m in enumerate(moments):
            p = _process_one_clip(idx, m)
            if p:
                done.append(p)

    # ── 4. Generate AI Titles ───────────────────────────────────────────
    all_titles = []
    if done:
        print("\n══ 4 · Generating AI titles ══")
        # Extract transcripts from moments (if we have them)
        transcripts = [m.get("transcript", "") for m in moments]
        if any(transcripts):
            all_titles = generate_titles_batch(
                transcripts, model=OLLAMA_DETECTOR_MODEL, language=title_language or language
            )
        else:
            print("[title-gen] No transcripts found; using default titles")

    print(f"\n══ Done! {len(done)} clips ══")
    for p in done:
        print(f"  → {p}")

    # ── 5. Upload / schedule ─────────────────────────────────────────────
    if upload and done:
        print("\n══ 4 · Uploading to YouTube ══")
        sched = build_schedule(
            done,
            start_time=datetime.utcnow() + timedelta(hours=1),
            interval_hours=schedule_hours,
        )
        for item in sched:
            idx = done.index(item["path"]) + 1
            upload_to_youtube(
                item["path"],
                title=all_titles[idx-1] if (all_titles and idx-1 < len(all_titles)) else f"{stem} – Viral Clip #{idx}",
                description=f"Viral clip from {stem}\n\n#shorts #viral",
                scheduled_time=item["scheduled_time"],
            )

    return done


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="ViriaRevive – viral clip generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("-n", "--clips",    type=int, default=NUM_CLIPS,    help=f"number of clips  (default {NUM_CLIPS})")
    p.add_argument("-d", "--duration", type=int, default=CLIP_DURATION, help=f"clip length in seconds  (default {CLIP_DURATION})")
    p.add_argument("-s", "--style",    choices=["tiktok", "clean", "bold"], default=SUBTITLE_STYLE, help="subtitle style")
    p.add_argument("-m", "--model",    choices=["tiny", "base", "small", "medium", "large-v3"], default=WHISPER_MODEL, help="whisper model size")
    p.add_argument("-l", "--language", default=WHISPER_LANGUAGE, help="force language (en, es, fr …)")
    p.add_argument("--title-language", default=None, help="force AI title language (Spanish, French …)")
    p.add_argument("-u", "--upload",   action="store_true", help="upload clips to YouTube")
    p.add_argument("--schedule",       type=int, default=24, help="hours between scheduled uploads")
    p.add_argument("--no-crop",        action="store_true", help="ignored; Shorts-ready output is always 1080x1920")
    p.add_argument("--ai-detector", choices=["auto", "off", "on"], default=AI_DETECTOR_MODE, help="local Ollama AI detector mode")

    a = p.parse_args()
    process(
        url=a.url,
        num_clips=a.clips,
        clip_duration=a.duration,
        style=a.style,
        model=a.model,
        language=a.language,
        title_language=a.title_language,
        upload=a.upload,
        schedule_hours=a.schedule,
        crop=not a.no_crop,
        ai_detector=a.ai_detector,
    )


if __name__ == "__main__":
    main()
