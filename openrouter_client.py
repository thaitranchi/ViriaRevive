"""OpenRouter API HTTP helpers for cloud-based AI tasks.

OpenRouter provides a unified API to hundreds of models via an
OpenAI-compatible chat completions endpoint.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

BASE_URL = "https://openrouter.ai/api/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_WORKERS = 5


def is_available(api_key: Optional[str]) -> bool:
    return bool(api_key and not api_key.startswith("YOUR_"))


def test_connection(api_key: str, timeout: int = 15) -> dict:
    if not is_available(api_key):
        return {"ok": False, "error": "Invalid or missing API key"}
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "max_tokens": 8,
    }
    data = _post(api_key, payload, timeout)
    if data is None:
        return {"ok": False, "error": "Request failed — check key and network"}
    text = _extract_text(data)
    if text:
        return {"ok": True}
    error = _extract_error(data)
    return {"ok": False, "error": error or "Unexpected API response"}


def _build_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _post(
    api_key: str,
    payload: dict,
    timeout: int = 30,
) -> Optional[dict]:
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_headers(api_key),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            msg = json.loads(body).get("error", {}).get("message", body[:200])
        except json.JSONDecodeError:
            msg = body[:200] or f"HTTP {e.code}"
        print(f"[openrouter] HTTP {e.code}: {msg}")
    except Exception as e:
        print(f"[openrouter] Request failed: {e}")
    return None


def _extract_text(data: Optional[dict]) -> Optional[str]:
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return None


def _extract_error(data: dict) -> Optional[str]:
    try:
        err = data.get("error", {})
        return err.get("message") or str(err)
    except Exception:
        return None


def generate(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> Optional[str]:
    if not api_key:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    return _extract_text(_post(api_key, payload, timeout))


def generate_json(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> Optional[Any]:
    if not api_key:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    text = _extract_text(_post(api_key, payload, timeout))
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def rerank_moments(
    moments: List[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
    keep: int = 5,
) -> List[dict]:
    if not moments or not api_key:
        return []

    prompt = (
        "Analyze these gaming video transcript segments and score their viral "
        "potential (0.0 to 1.0). Focus on action moments: kills, clutches, wins, "
        "fails, reactions, team fights, and shoutcaster hype. Reward segments with "
        "high energy, crowd reactions, or dramatic gameplay. "
        "Return a JSON object with a 'scores' list of floats corresponding to the input order."
        "\n\nSegments:\n"
    )
    for i, m in enumerate(moments):
        prompt += f"{i}. {m.get('transcript', '')[:300]}\n"

    result = generate_json(prompt, api_key, model)
    if result and "scores" in result:
        scores = result["scores"]
        for i, m in enumerate(moments):
            m["ai_score"] = float(scores[i]) if i < len(scores) else 0.0
        sorted_moments = sorted(
            moments, key=lambda x: x.get("ai_score", 0), reverse=True
        )
        return sorted_moments[:keep]
    return moments[:keep]


def generate_titles_batch(
    transcripts: List[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    language: str = None,
    on_progress=None,
    max_workers: int = MAX_WORKERS,
) -> List[str]:
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
            f"Create a viral gaming YouTube Short title in {lang_str} for this "
            f"gameplay clip. Make it hype and gamer-friendly. Use gaming slang if "
            f"appropriate (clutch, OP, insane, wipeout, GG). Keep it under 50 chars."
            f"\n\nTranscript: {text}"
        )
        res = generate(prompt, api_key, model)
        return idx, res or ""

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_gen_one, i, t): i
            for i, t in enumerate(transcripts)
            if t
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                idx_res, title = future.result()
                results[idx_res] = title
                done_count += 1
                if on_progress:
                    on_progress(done_count, total, title)
            except Exception as e:
                done_count += 1
                if on_progress:
                    on_progress(done_count, total, "")
                print(f"[openrouter-batch] Error generating title {idx}: {e}")

    # Report remaining progress for empty transcripts (not submitted)
    if on_progress and done_count < total:
        on_progress(total, total, "")

    return results
