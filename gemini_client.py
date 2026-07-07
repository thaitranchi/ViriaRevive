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


def _build_url(api_key: str) -> str:
    """Build Gemini API URL (key goes in header, not URL, to avoid exposure in logs)."""
    return f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def _build_headers(api_key: str) -> dict:
    """Build request headers with API key in x-goog-api-key header."""
    return {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }


def test_connection(api_key: str, timeout: int = 15) -> dict:
    """Verify a Gemini API key with a minimal generateContent request."""
    if not is_available(api_key):
        return {"ok": False, "error": "Invalid or missing API key"}
    url = _build_url(api_key)
    payload = {
        "contents": [{"parts": [{"text": "Reply with OK only."}]}],
        "generationConfig": {"maxOutputTokens": 8},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_headers(api_key),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        if _extract_text(data):
            return {"ok": True}
        return {"ok": False, "error": "Unexpected API response"}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            msg = json.loads(body).get("error", {}).get("message", body[:200])
        except json.JSONDecodeError:
            msg = body[:200] or f"HTTP {e.code}"
        return {"ok": False, "error": msg}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _post(api_key: str, payload: dict, timeout: int) -> Optional[dict]:
    """Internal helper to send a POST request to Gemini."""
    url = _build_url(api_key)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_headers(api_key),
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
        
    prompt = "Analyze these gaming video transcript segments and score their viral potential (0.0 to 1.0). " \
             "Focus on action moments: kills, clutches, wins, fails, reactions, team fights, and shoutcaster hype. " \
             "Reward segments with high energy, crowd reactions, or dramatic gameplay. " \
             "Return a JSON object with a 'scores' list of floats corresponding to the input order.\n\nSegments:\n"
             
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
        prompt = (
            f"Create a viral gaming YouTube Short title in {lang_str} for this gameplay clip. "
            f"Make it hype and gamer-friendly. Use gaming slang if appropriate "
            f"(clutch, OP, insane, wipeout, GG). Keep it under 50 chars.\n\n"
            f"Transcript: {text}"
        )
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
