"""Shared local Ollama HTTP helpers.

All calls stay on localhost and return conservative fallbacks instead of
raising into the pipeline.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


BASE_URL = "http://127.0.0.1:11434"
GENERATE_URL = f"{BASE_URL}/api/generate"
TAGS_URL = f"{BASE_URL}/api/tags"
PULL_URL = f"{BASE_URL}/api/pull"
DEFAULT_MODEL = "qwen2.5:3b"


def ollama_available(timeout: int | float = 3) -> bool:
    """Return True when the local Ollama server responds."""
    try:
        req = urllib.request.Request(TAGS_URL)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_models(timeout: int | float = 3) -> list[str]:
    """Return local model names, or an empty list if Ollama is unavailable."""
    try:
        req = urllib.request.Request(TAGS_URL)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def model_exists(model: str = DEFAULT_MODEL, timeout: int | float = 3) -> bool:
    """Check whether a model is already downloaded."""
    names = list_models(timeout=timeout)
    return model in names or f"{model}:latest" in names


def pull_model(model: str = DEFAULT_MODEL, timeout: int | float = 300) -> bool:
    """Pull a model through Ollama. Returns False on any failure."""
    print(f"[ollama] Model '{model}' not found; pulling from Ollama...")
    body = json.dumps({"name": model, "stream": False}).encode()
    req = urllib.request.Request(
        PULL_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            status = str(data.get("status", ""))
            if status:
                print(f"[ollama] Pull status: {status}")
            return resp.status == 200
    except Exception as e:
        print(f"[ollama] Failed to pull model '{model}': {e}")
        return False


def ensure_model(model: str = DEFAULT_MODEL) -> bool:
    """Ensure Ollama and the requested model are ready."""
    if not ollama_available():
        return False
    if model_exists(model):
        return True
    return pull_model(model)


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int | float = 30,
    options: dict[str, Any] | None = None,
    format: str | None = None,
) -> str | None:
    """Generate a non-streaming response. Returns None on failure."""
    body_data: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        body_data["options"] = options
    if format:
        body_data["format"] = format

    req = urllib.request.Request(
        GENERATE_URL,
        data=json.dumps(body_data).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            response = data.get("response", "")
            return response.strip() if isinstance(response, str) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[ollama] Generate failed: {e}")
    except Exception as e:
        print(f"[ollama] Unexpected generate error: {e}")
    return None


def generate_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int | float = 30,
    options: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Generate and parse a JSON object. Returns None on invalid output."""
    response = generate(
        prompt,
        model=model,
        timeout=timeout,
        options=options,
        format="json",
    )
    if not response:
        return None

    try:
        data = json.loads(response)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(response[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
    print(f"[ollama] Invalid JSON response: {response[:120]}")
    return None
