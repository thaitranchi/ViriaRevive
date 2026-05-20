"""Lightweight title generator using a local Ollama model.

Falls back to a simple extraction heuristic if Ollama is unavailable,
so this never blocks the pipeline.
"""

from ollama_client import (
    DEFAULT_MODEL,
    ensure_model,
    generate,
    list_models,
    model_exists,
    ollama_available,
    pull_model,
)

TIMEOUT = 30  # seconds per title request


def _ollama_available() -> bool:
    """Quick check if Ollama is running."""
    return ollama_available()


def _model_exists(model: str = DEFAULT_MODEL) -> bool:
    """Check if a specific model is already downloaded in Ollama."""
    return model_exists(model)


def _pull_model(model: str = DEFAULT_MODEL) -> bool:
    """Pull (download) a model via Ollama. Blocks until complete."""
    return pull_model(model)


def _ask_ollama(transcript: str, model: str = DEFAULT_MODEL, language: str = None) -> str | None:
    """Ask Ollama for a catchy short YouTube Shorts title."""
    lang_instruction = f" and reply in {language}" if language else ""
    prompt = (
        "You are a viral YouTube Shorts title expert. "
        f"Given a transcript, create ONE clickbait title that makes people NEED to watch{lang_instruction}.\n\n"
        "RULES:\n"
        "- Max 50 characters\n"
        "- No quotes, no hashtags, no emojis\n"
        "- Use curiosity gaps, shock value, or bold claims\n"
        "- NEVER just copy words from the transcript\n"
        "- Good examples: 'Nobody Expected This to Happen', 'He Instantly Regretted It', "
        "'This Changes Everything'\n\n"
        f'Transcript: "{transcript[:500]}"\n\n'
        "Reply with ONLY the title. Nothing else."
    )
    response = generate(
        prompt,
        model=model,
        timeout=TIMEOUT,
        options={"temperature": 0.7, "num_predict": 40},
    )
    try:
        if response:
            title = response.strip().strip('"').strip("'")
            # Clean up: remove hashtags, limit length
            title = title.split("\n")[0].strip()
            # Remove common LLM artifacts
            for prefix in ["Title:", "title:", "Here's", "Here is"]:
                if title.startswith(prefix):
                    title = title[len(prefix):].strip().strip('"').strip("'").strip()
            if title and len(title) >= 3:
                # Truncate at word boundary to keep titles clean for Shorts
                if len(title) > 60:
                    words = title.split()
                    title = ""
                    for w in words:
                        candidate = f"{title} {w}".strip() if title else w
                        if len(candidate) > 60:
                            break
                        title = candidate
                return title
    except Exception as e:
        print(f"[title-gen] Ollama cleanup error: {e}")
    return None


def _heuristic_title(transcript: str) -> str:
    """Fallback: generate a clickbait-style title from transcript keywords."""
    import random
    if not transcript:
        return ""

    words = transcript.lower().split()

    # Extract a short key phrase (2-4 words) from the middle of the transcript
    # Middle tends to have the core topic, not filler intro/outro
    mid = len(words) // 2
    start = max(0, mid - 2)
    key_phrase = " ".join(words[start:start + 3]).strip(".,!?;:'\"")

    # Clickbait templates — {topic} gets replaced with the key phrase
    templates = [
        "Nobody Expected {topic} to Go Like This",
        "Wait Until You See {topic}",
        "This Is Why {topic} Went Viral",
        "{topic} Will Blow Your Mind",
        "You Won't Believe What Happened With {topic}",
        "Everyone Is Talking About {topic}",
        "The Truth About {topic} Is Shocking",
        "{topic} Changed Everything",
        "I Can't Believe {topic} Actually Worked",
        "Watch What Happens With {topic}",
    ]

    topic = key_phrase.title()
    title = random.choice(templates).format(topic=topic)

    # If title is too long, use shorter templates
    if len(title) > 55:
        short_templates = [
            "{topic} Goes Wrong",
            "{topic} Went Viral",
            "Wait for {topic}",
            "{topic} Was Insane",
            "This {topic} Though",
        ]
        title = random.choice(short_templates).format(topic=topic)

    # Final safety: truncate at word boundary
    if len(title) > 60:
        parts = title.split()
        title = ""
        for w in parts:
            candidate = f"{title} {w}".strip() if title else w
            if len(candidate) > 55:
                break
            title = candidate

    return title


def generate_title(transcript: str, model: str = DEFAULT_MODEL, language: str = None) -> str:
    """Generate a title for a clip. Uses Ollama if available, else heuristic."""
    if not transcript:
        print("[title-gen] Skipped — empty transcript")
        return ""
    
    # Clean up transcript from Whisper metadata if present (e.g. [laughter], (silence))
    import re
    clean_transcript = re.sub(r'\[.*?\]|\(.*?\)', '', transcript).strip()

    # Try Ollama first — auto-pull model if needed
    if ensure_model(model):
        result = _ask_ollama(clean_transcript, model, language)
        if result:
            print(f"[title-gen] LLM: {result}")
            return result
        print(f"[title-gen] LLM returned empty/short, trying heuristic...")

    # Fallback to heuristic
    result = _heuristic_title(transcript)
    if result:
        print(f"[title-gen] Heuristic: {result}")
    else:
        print(f"[title-gen] Both LLM and heuristic failed for transcript: {transcript[:60]}...")
    return result


def generate_titles_batch(
    transcripts: list[str],
    model: str = DEFAULT_MODEL,
    language: str = None,
    on_progress=None,
) -> list[str]:
    """Generate titles for multiple clips. Uses concurrent requests for speed.

    on_progress(done, total, title) is called after each title is generated.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(transcripts)
    if not total:
        return []

    # Check model availability ONCE, not per-title
    model_ready = ensure_model(model)

    results = [""] * total
    done_count = 0

    def _gen_one(idx_transcript):
        idx, transcript = idx_transcript
        if not transcript:
            return idx, ""
        if model_ready:
            title = _ask_ollama(transcript, model, language)
            if title:
                return idx, title
        return idx, _heuristic_title(transcript)

    # Run up to 3 concurrent Ollama requests (Ollama handles queuing internally)
    workers = min(3, total) if model_ready else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_gen_one, (i, t)): i for i, t in enumerate(transcripts)}
        for future in as_completed(futures):
            try:
                idx, title = future.result()
                results[idx] = title
                done_count += 1
                if on_progress:
                    on_progress(done_count, total, title)
                print(f"[title-gen] {done_count}/{total}: {title or '(empty)'}")
            except Exception as e:
                done_count += 1
                print(f"[title-gen] Error: {e}")

    return results


def list_ollama_models() -> list[str]:
    """Return available Ollama models, or empty list if unavailable."""
    return list_models()
