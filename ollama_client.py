"""Shared local Ollama HTTP helpers.

All calls stay on localhost and return conservative fallbacks instead of
raising into the pipeline.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


BASE_URL = "http://127.0.0.1:11434"
GENERATE_URL = f"{BASE_URL}/api/generate"
CHAT_URL = f"{BASE_URL}/api/chat"
TAGS_URL = f"{BASE_URL}/api/tags"
PULL_URL = f"{BASE_URL}/api/pull"
DEFAULT_MODEL = "qwen2.5:3b"


def ollama_available(timeout: int | float = 3) -> bool:
    """Return True when the local Ollama server responds."""
    try:
        req = urllib.request.Request(TAGS_URL)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception as e:
        logger.debug("Ollama not available: %s", e)
        return False


def list_models(timeout: int | float = 3) -> list[str]:
    """Return local model names, or an empty list if Ollama is unavailable."""
    try:
        req = urllib.request.Request(TAGS_URL)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        logger.debug("Failed to list Ollama models: %s", e)
        return []


def model_exists(model: str = DEFAULT_MODEL, timeout: int | float = 3) -> bool:
    """Check whether a model is already downloaded.

    Handles namespaced tags (e.g. ``qcwind/qwen3-8b-instruct-Q4-K-M:latest``)
    by matching on the bare model name as well as the fully-qualified tag.
    """
    names = list_models(timeout=timeout)
    target_base = model.split(":", 1)[0]
    for n in names:
        if n == model or n.split(":", 1)[0] == target_base:
            return True
    return False


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
    response_format: str | None = None,
    context: list[int] | None = None,
    return_context: bool = False,
) -> str | tuple[str, list[int]] | None:
    """Generate a response. If return_context is True, returns (response, context)."""
    body_data: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 1024,  # Default to a higher token limit for longer responses
            "temperature": 0.7,   # Default temperature for general generation
        }
    }
    if context:
        body_data["context"] = context
    if options:
        body_data["options"].update(options)  # Merge provided options, allowing overrides
    if response_format:
        body_data["format"] = response_format

    req = urllib.request.Request(
        GENERATE_URL,
        data=json.dumps(body_data).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            response = data.get("response", "")
            text = response.strip() if isinstance(response, str) else None
            if return_context and text is not None:
                return text, data.get("context", [])
            return text
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[ollama] Generate failed: {e}")
    except Exception as e:
        print(f"[ollama] Unexpected generate error: {e}")
    return None


def resume(
    context: list[int],
    prompt: str = "",
    model: str = DEFAULT_MODEL,
    timeout: int | float = 30,
    options: dict[str, Any] | None = None,
) -> tuple[str, list[int]] | None:
    """Continue a previous generation using the provided context."""
    result = generate(
        prompt, model, timeout, options, context=context, return_context=True
    )
    if isinstance(result, tuple):
        return result
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
        response_format="json",
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


def chat(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int | float = 30,
    options: dict[str, Any] | None = None,
    response_format: str | None = None,
) -> str | None:
    """Send a multi-turn chat request (used for multimodal/vision calls).

    *messages* is a list of ``{"role", "content", "images"?}`` dicts.
    Returns the assistant message content, or None on failure.
    """
    body_data: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 1024,
            "temperature": 0.7,
        },
    }
    if options:
        body_data["options"].update(options)
    if response_format:
        body_data["format"] = response_format

    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(body_data).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            content = data.get("message", {}).get("content", "")
            return content.strip() if isinstance(content, str) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[ollama] Chat failed: {e}")
    except Exception as e:
        print(f"[ollama] Unexpected chat error: {e}")
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from an LLM response string."""
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def chat_json(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int | float = 30,
    options: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Chat and parse a JSON object from the response. Returns None on failure."""
    response = chat(messages, model=model, timeout=timeout, options=options, response_format="json")
    if not response:
        return None
    data = _extract_json(response)
    if data is None:
        print(f"[ollama] Invalid JSON chat response: {response[:120]}")
    return data


def generate_vision(
    prompt: str,
    images: list[str],
    model: str = DEFAULT_MODEL,
    timeout: int | float = 30,
    options: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run a vision model over one or more base64-encoded images.

    *images* must be base64 strings (raw bytes, no data-URI prefix).

    NOTE: vision models (e.g. Qwen3-VL) reject ``format:"json"`` when images
    are attached, so the JSON request is expressed in the prompt and the reply
    is parsed from the message content instead.
    """
    if not images:
        return None
    messages = [{"role": "user", "content": prompt, "images": images}]
    response = chat(messages, model=model, timeout=timeout, options=options)
    if not response:
        return None
    data = _extract_json(response)
    if data is None:
        print(f"[ollama] Invalid JSON vision response: {response[:120]}")
    return data
