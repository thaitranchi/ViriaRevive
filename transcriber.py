from pathlib import Path

from hwaccel import resolve_whisper_device

# Per-GPU model cache: key = (model_size, device, compute, gpu_index)
_model_cache = {}


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
    if cache_key not in _model_cache:
        print(f"[*] Loading Whisper {model_size} ({device}/{compute},"
              f" device_index={device_index})...")
        try:
            _model_cache[cache_key] = WhisperModel(
                model_size, device=device, device_index=device_index,
                compute_type=compute,
            )
        except Exception as e:
            if device != "cpu":
                print(f"[!] Whisper GPU init failed ({e}), falling back to CPU...")
                return _load_whisper_model(model_size, "cpu", "int8", 0)
            print(f"[!] Whisper CPU init also failed ({e})")
            raise
    return _model_cache[cache_key]


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

    Strategy (in priority order):
      1. Look for sentence-ending punctuation (.!?) near the end of the clip
      2. Look for a long natural pause (>0.5s gap between words)
      3. Look for soft punctuation (comma, colon, semicolon)
      4. If nothing found, return None (keep original duration)

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

    min_time = clip_duration * min_keep    # don't cut before this
    max_time = clip_duration + max_extend  # don't extend past this

    # ── Pass 1: sentence-ending punctuation (.!?) ──
    # Search backward from the end — prefer the latest sentence end
    best_sentence_end = None
    for w in reversed(words):
        if w["end"] < min_time:
            break
        if w["end"] > max_time:
            continue
        text = w["text"].rstrip()
        if text and text[-1] in _SENTENCE_ENDERS:
            best_sentence_end = w["end"]
            break  # take the latest one within range

    if best_sentence_end is not None:
        # Add a small pad (0.3s) after the last word for natural breathing room
        result = best_sentence_end + 0.3
        print(f"    [sentence] Snapped to sentence end at {result:.1f}s "
              f"(was {clip_duration}s)")
        return result

    # ── Pass 2: long natural pause between words ──
    best_pause_end = None
    for i in range(len(words) - 1, 0, -1):
        word_end = words[i - 1]["end"]
        next_start = words[i]["start"]
        if word_end < min_time:
            break
        if word_end > max_time:
            continue
        gap = next_start - word_end
        if gap >= _PAUSE_THRESHOLD:
            best_pause_end = word_end
            break

    if best_pause_end is not None:
        result = best_pause_end + 0.2
        print(f"    [sentence] Snapped to natural pause at {result:.1f}s "
              f"(was {clip_duration}s, gap={best_pause_end:.2f}s)")
        return result

    # ── Pass 3: soft punctuation (comma, colon, etc.) ──
    best_soft_end = None
    for w in reversed(words):
        if w["end"] < min_time:
            break
        if w["end"] > max_time:
            continue
        text = w["text"].rstrip()
        if text and text[-1] in _SOFT_ENDERS:
            best_soft_end = w["end"]
            break

    if best_soft_end is not None:
        result = best_soft_end + 0.25
        print(f"    [sentence] Snapped to soft break at {result:.1f}s "
              f"(was {clip_duration}s)")
        return result

    # ── No good boundary found ──
    print(f"    [sentence] No natural boundary found near {clip_duration}s, keeping as-is")
    return None
