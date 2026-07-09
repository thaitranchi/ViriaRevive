import threading
from pathlib import Path

from hwaccel import resolve_whisper_device

# Per-GPU model cache: key = (model_size, device, compute, gpu_index)
_model_cache = {}
_cache_lock = threading.Lock()


def _get_device(device_pref: str = "auto",
                gpu_index: int | None = None) -> tuple[str, str, int]:
    """Resolve (device, compute_type, device_index) for faster-whisper.

    When *gpu_index* is provided, whisper is pinned to that specific GPU.
    """
    return resolve_whisper_device(device_pref, gpu_index=gpu_index)


def _load_whisper_model(model_size: str, device: str, compute: str,
                        device_index: int = 0):
    from faster_whisper import WhisperModel

    cache_key = (model_size, device, compute, device_index)
    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
    print(f"[*] Loading Whisper {model_size} ({device}/{compute},"
          f" device_index={device_index})...")
    try:
        model = WhisperModel(
            model_size, device=device, device_index=device_index,
            compute_type=compute,
        )
    except Exception as e:
        if device != "cpu":
            print(f"[!] Whisper GPU init failed ({e}), falling back to CPU...")
            return _load_whisper_model(model_size, "cpu", "int8", 0)
        print(f"[!] Whisper CPU init also failed ({e})")
        raise
    with _cache_lock:
        _model_cache[cache_key] = model
    return model


def transcribe_clip(
    audio_path: Path,
    model_size: str = "base",
    language: str = None,
    device_pref: str = "auto",
    gpu_index: int | None = None,
) -> list:
    """Transcribe audio and return word-level timestamps.

    Returns list of dicts: [{'text': str, 'start': float, 'end': float}, ...]

    If *gpu_index* is provided the whisper model is pinned to that GPU.
    """
    device, compute, device_index = _get_device(device_pref, gpu_index=gpu_index)
    model = _load_whisper_model(model_size, device, compute, device_index)

    print(f"[*] Transcribing {audio_path.name} (GPU {device_index})...")
    try:
        segments, info = model.transcribe(
            str(audio_path), word_timestamps=True, language=language
        )
    except Exception as e:
        if device != "cpu":
            print(f"[!] Whisper GPU transcribe failed ({e}), retrying on CPU...")
            model = _load_whisper_model(model_size, "cpu", "int8", 0)
            segments, info = model.transcribe(
                str(audio_path), word_timestamps=True, language=language
            )
        else:
            print(f"[!] Whisper CPU transcribe also failed ({e})")
            raise

    from subprocess_utils import is_cancelled, CancelledError

    words = []
    for seg in segments:
        if is_cancelled():
            raise CancelledError("Transcription cancelled")
        if seg.words:
            for w in seg.words:
                words.append({"text": w.word.strip(), "start": w.start, "end": w.end})

    print(f"[+] Transcribed {len(words)} words  (lang: {info.language})")
    return words


# ── Sentence-boundary detection ───────────────────────────────────────────────

# Punctuation that marks a natural sentence ending
_SENTENCE_ENDERS = {'.', '!', '?', '…'}
# Words/phrases that feel like natural conclusions even without strong punctuation
_SOFT_ENDERS = {',', ':', ';', '—', '-'}

# Minimum silence gap (seconds) between words to count as a natural pause
_PAUSE_THRESHOLD = 0.50


def find_sentence_boundary(words: list, clip_duration: float,
                           min_keep: float = 0.60,
                           max_extend: float = 5.0) -> float | None:
    """Find the best sentence-ending near the clip boundary.

    Scans the transcribed words and returns a new clip duration (in seconds)
    that ends on a natural sentence boundary — so the speaker finishes their
    thought instead of being cut off mid-sentence.

    Strategy: score each candidate boundary within the search range, picking
    the highest-scoring one. Boundaries closer to the original end are
    preferred (distance-based decay), with punctuation taking priority
    over pauses.

    Args:
        words: list of {'text': str, 'start': float, 'end': float}
        clip_duration: original clip duration in seconds
        min_keep: minimum fraction of clip to keep (default 60%)
        max_extend: max seconds to extend beyond original end (default 5s)

    Returns:
        New clip duration (float) or None if no good boundary found.
    """
    if not words or len(words) < 3:
        return None
    from subprocess_utils import is_cancelled
    if is_cancelled():
        return None

    min_time = clip_duration * min_keep    # don't cut before this
    max_time = clip_duration + max_extend  # don't extend past this

    # ── Single pass: score every candidate boundary ──
    best_score = 0.0
    best_end = None
    best_type = ""

    for i in range(len(words) - 1, -1, -1):
        if is_cancelled():
            return None
        w = words[i]
        if w["end"] < min_time:
            break
        if w["end"] > max_time:
            continue

        # Distance from original clip end: 1.0 at clip_duration, 0.5 at boundaries
        dist = abs(w["end"] - clip_duration)
        dist_range = max(clip_duration - min_time, max_time - clip_duration)
        dist_weight = 1.0 - 0.5 * (dist / max(1.0, dist_range))

        text = w["text"].rstrip()

        if text and text[-1] in _SENTENCE_ENDERS:
            score = 1.0 * dist_weight
            if score > best_score:
                best_score = score
                best_end = w["end"] + 0.3
                best_type = "sentence end"

        elif i < len(words) - 1:
            gap = words[i + 1]["start"] - w["end"]
            if gap >= _PAUSE_THRESHOLD:
                score = 0.8 * dist_weight
                if score > best_score:
                    best_score = score
                    best_end = w["end"] + 0.2
                    best_type = "natural pause"

        if text and text[-1] in _SOFT_ENDERS:
            score = 0.6 * dist_weight
            if score > best_score:
                best_score = score
                best_end = w["end"] + 0.25
                best_type = "soft break"

    if best_end is not None:
        print(f"    [sentence] Snapped to {best_type} at {best_end:.1f}s "
              f"(was {clip_duration}s, score={best_score:.2f})")
        return best_end

    print(f"    [sentence] No natural boundary found near {clip_duration}s, keeping as-is")
    return None
