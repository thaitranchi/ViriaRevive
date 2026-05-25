"""Gemini API HTTP helpers for cloud-based AI tasks."""

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

MODEL = "gemini-1.5-flash"

def is_available(api_key: Optional[str]) -> bool:
    """Check if the Gemini API key is configured."""
    return bool(api_key and not api_key.startswith("YOUR_"))

def _post(api_key: str, payload: dict, timeout: int) -> Optional[dict]:
    """Internal helper to send a POST request to Gemini."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[gemini] Request failed: {e}")
        return None

def _extract_text(data: Optional[dict]) -> Optional[str]:
    """Navigate through Gemini response structure to extract generated text."""
    if not data: return None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return None

def generate(prompt: str, api_key: str, timeout: int = 30) -> Optional[str]:
    if not api_key: return None
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
    }
    return _extract_text(_post(api_key, payload, timeout))

def generate_json(prompt: str, api_key: str, timeout: int = 30) -> Optional[Any]:
    if not api_key: return None
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    text = _extract_text(_post(api_key, payload, timeout))
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None

def rerank_moments(moments: List[dict], api_key: str, keep: int = 5) -> List[dict]:
    """Score and rerank viral candidates using Gemini."""
    if not moments or not api_key:
        return []
        
    prompt = "Analyze these video transcript segments and score their viral potential (0.0 to 1.0). " \
             "Focus on hooks, humor, and self-contained stories. Return a JSON object with a 'scores' " \
             "list of floats corresponding to the input order.\n\nSegments:\n"
             
    for i, m in enumerate(moments):
        prompt += f"{i}. {m.get('transcript', '')[:300]}\n"
        
    result = generate_json(prompt, api_key)
    if result and "scores" in result:
        scores = result["scores"]
        for i, m in enumerate(moments):
            if i < len(scores):
                m["ai_score"] = float(scores[i])
            else:
                m["ai_score"] = 0.0
                
        # Sort by score and take top N
        sorted_moments = sorted(moments, key=lambda x: x.get("ai_score", 0), reverse=True)
        return sorted_moments[:keep]
    return moments[:keep]

def generate_titles_batch(
    transcripts: List[str],
    api_key: str,
    language: str = None,
    on_progress=None,
    max_workers: int = 5
) -> List[str]:
    """Generate titles for multiple clips in parallel using a thread pool."""
    total = len(transcripts)
    if not total or not api_key:
        return [""] * total

    results = [""] * total
    done_count = 0
    lang_str = language or "English"

    def _gen_one(idx: int, text: str) -> tuple[int, str]:
        if not text:
            return idx, ""
        prompt = f"Create a viral YouTube Short title in {lang_str} for: {text}"
        res = generate(prompt, api_key)
        return idx, res or ""

    # Using a thread pool to handle concurrent HTTP requests to Gemini.
    # max_workers is capped to 5 to avoid hitting standard rate limits too quickly.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_gen_one, i, t): i 
            for i, t in enumerate(transcripts) 
            if t # Only submit non-empty transcripts
        }
        
        for future in as_completed(futures):
            try:
                idx, title = future.result()
                results[idx] = title
                done_count += 1
                if on_progress:
                    # Report progress back to the UI
                    on_progress(done_count, total, title)
            except Exception as e:
                done_count += 1
                print(f"[gemini-batch] Error generating title {idx}: {e}")

    return results