"""LLM-based transcript translation using Ollama.

Translates transcribed words or plain text to a target language.
Word-level translation preserves timing for subtitle alignment.
Falls back to original text on any failure — never blocks the pipeline.
"""

import logging
import re

from ollama_client import DEFAULT_MODEL, generate as _ollama_generate

logger = logging.getLogger(__name__)

_TIMEOUT = 30


def translate_text(text: str, target_language: str,
                   model: str = DEFAULT_MODEL) -> str | None:
    """Translate plain transcript text to *target_language*.

    Returns translated text or *None* on failure.
    """
    if not text or not target_language:
        return None
    prompt = (
        f"Translate the following text to {target_language}. "
        "Reply with ONLY the translation, no explanations, no quotes.\n\n"
        f"Text: {text[:1200]}"
    )
    try:
        response = _ollama_generate(prompt, model=model, timeout=_TIMEOUT,
                                    options={"temperature": 0.1, "num_predict": 120})
        if response:
            translation = response.strip().strip('"').strip("'")
            if translation and len(translation) >= 3:
                return translation
    except Exception as e:
        logger.debug("translate_text failed: %s", e)
    return None


def translate_words(words: list, target_language: str,
                    model: str = DEFAULT_MODEL,
                    batch_size: int = 6) -> list:
    """Translate word text preserving timing.

    Groups words into small batches, translates each batch's text,
    splits the translation back into words, and assigns original timing.
    Falls back to original words on failure.

    Returns a new list of *{'text', 'start', 'end'}* dicts.
    """
    if not words or not target_language:
        return words

    from subtitler import _group_phrases
    phrases = _group_phrases(words, max_words=batch_size, max_dur=4.0, max_gap=1.0)

    result = []
    for phrase in phrases:
        pw = phrase["words"]
        original_text = " ".join(w["text"] for w in pw)

        translated = translate_text(original_text, target_language, model=model)
        if not translated:
            # Fall back to original words for this phrase
            result.extend(pw)
            continue

        # Split translated text back into word-level tokens
        t_words = translated.split()
        if not t_words:
            result.extend(pw)
            continue

        # Distribute original timing across translated words
        phrase_start = pw[0]["start"]
        phrase_end = pw[-1]["end"]
        phrase_dur = max(0.1, phrase_end - phrase_start)
        n_orig = len(pw)
        n_trans = len(t_words)

        if n_trans == n_orig:
            # 1:1 word mapping — assign each original timing directly
            for i, tw in enumerate(t_words):
                result.append({"text": tw, "start": pw[i]["start"], "end": pw[i]["end"]})
        else:
            # Different word count — distribute timing proportionally
            per_word = phrase_dur / max(1, n_trans)
            for i, tw in enumerate(t_words):
                s = phrase_start + i * per_word
                e = s + per_word
                result.append({"text": tw, "start": s, "end": e})

    if not result:
        return words
    return result
