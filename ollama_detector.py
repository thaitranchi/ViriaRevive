"""Ollama-backed reranking for candidate viral moments."""

from __future__ import annotations

from typing import Any

from ollama_client import DEFAULT_MODEL, ensure_model, generate_json


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(low, min(high, number))


def _safe_offset(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, number))


def detector_ready(model: str = DEFAULT_MODEL) -> bool:
    """Return True if the model is ready, pulling it if needed."""
    return ensure_model(model)


def score_candidate(
    transcript: str,
    moment: dict,
    clip_duration: int,
    model: str = DEFAULT_MODEL,
    timeout: int = 20,
) -> dict | None:
    """Return a normalized Ollama score object, or None on invalid output."""
    transcript = (transcript or "").strip()
    if not transcript:
        return None

    start = int(moment.get("start", 0))
    end = int(moment.get("end", start + clip_duration))
    heuristic_score = float(moment.get("score", 0.0) or 0.0)

    prompt = f"""You are selecting viral gaming moments for Shorts/TikTok.
Score this transcript from 0 to 1.

Prefer (gaming content):
- high-kill plays / multikills / aces
- clutch moments (1vX, last-second wins)
- funny fails or reactions
- impressive mechanics / outplays
- shoutcaster hype or crowd reactions
- game-winning / round-winning moments
- unexpected outcomes or comebacks
- toxic or funny voice comms

Reject (gaming content):
- menu navigation or lobby screens
- inventory management or gear selection
- quiet walking / dead air
- loading screens
- slow explanations or tutorials

Candidate metadata:
- source_start_seconds: {start}
- source_end_seconds: {end}
- heuristic_score: {heuristic_score:.4f}
- target_clip_duration_seconds: {clip_duration}
- person_detection_confidence: {moment.get("visual_score", 0.0):.2f} (0=none, 1=always in frame)

Transcript:
{transcript[:1800]}

Return only JSON:
{{"viral_score":0.0,"reason":"short reason","better_start_offset":0,"better_end_offset":{clip_duration}}}
"""
    data = generate_json(
        prompt,
        model=model,
        timeout=timeout,
        options={"temperature": 0.1, "num_predict": 120},
    )
    if not data:
        return None

    viral_score = _clamp(data.get("viral_score"))
    if viral_score is None:
        return None

    duration = max(1, end - start)
    better_start = _safe_offset(data.get("better_start_offset"), 0, 0, duration - 1)
    max_end = min(duration, max(clip_duration, better_start + 1))
    better_end = _safe_offset(
        data.get("better_end_offset"),
        min(duration, clip_duration),
        better_start + 1,
        max_end,
    )
    reason = str(data.get("reason", "")).strip()[:240]

    return {
        "viral_score": viral_score,
        "reason": reason,
        "better_start_offset": better_start,
        "better_end_offset": better_end,
    }


def vision_score_candidate(
    transcript: str,
    moment: dict,
    clip_duration: int,
    vision_meta: dict,
    model: str = DEFAULT_MODEL,
    timeout: int = 20,
) -> dict | None:
    """Score a candidate using transcript text PLUS vision metadata.

    Vision metadata (from ``vision_analyzer``) carries ``highlight_score``,
    ``is_ui_screen``, ``ocr_text``, ``scene``, ``action`` and ``reason``.
    UI screens (menu/loading/inventory) are hard-rejected without calling the
    LLM, since they are never highlights for gaming Shorts.
    """
    transcript = (transcript or "").strip()
    vmeta = vision_meta or {}
    is_ui = bool(vmeta.get("is_ui_screen", False))

    if is_ui:
        reason = f"UI screen (menu/loading) — {vmeta.get('reason', '')}"[:240]
        return {
            "viral_score": 0.1,
            "reason": reason,
            "better_start_offset": 0,
            "better_end_offset": min(clip_duration, int(moment.get("end", 0) - moment.get("start", 0))),
        }

    start = int(moment.get("start", 0))
    end = int(moment.get("end", start + clip_duration))
    heuristic_score = float(moment.get("score", 0.0) or 0.0)
    vision_score = float(vmeta.get("highlight_score", 0.0) or 0.0)

    prompt = f"""You are selecting viral gaming moments for Shorts/TikTok.
Score this candidate from 0 to 1.

Prefer (gaming content):
- high-kill plays / multikills / aces
- clutch moments (1vX, last-second wins)
- funny fails or reactions
- impressive mechanics / outplays
- game-winning / round-winning moments
- unexpected outcomes or comebacks
- toxic or funny voice comms

Reject (gaming content):
- menu navigation or lobby screens
- inventory management or gear selection
- quiet walking / dead air
- loading screens
- slow explanations or tutorials

Candidate metadata:
- source_start_seconds: {start}
- source_end_seconds: {end}
- heuristic_score: {heuristic_score:.4f}
- target_clip_duration_seconds: {clip_duration}
- person_detection_confidence: {moment.get("visual_score", 0.0):.2f} (0=none, 1=always in frame)

Vision analysis (Qwen3-VL on sampled frames):
- vision_highlight_score: {vision_score:.2f}
- is_ui_screen: {is_ui}
- ocr_text: {vmeta.get("ocr_text", "")}
- scene: {vmeta.get("scene", "")}
- action: {vmeta.get("action", "")}
- vision_reason: {vmeta.get("reason", "")}

Transcript:
{transcript[:1800]}

Return only JSON:
{{"viral_score":0.0,"reason":"short reason","better_start_offset":0,"better_end_offset":{clip_duration}}}
"""
    data = generate_json(
        prompt,
        model=model,
        timeout=timeout,
        options={"temperature": 0.1, "num_predict": 120},
    )
    if not data:
        return None

    viral_score = _clamp(data.get("viral_score"))
    if viral_score is None:
        return None

    duration = max(1, end - start)
    better_start = _safe_offset(data.get("better_start_offset"), 0, 0, duration - 1)
    max_end = min(duration, max(clip_duration, better_start + 1))
    better_end = _safe_offset(
        data.get("better_end_offset"),
        min(duration, clip_duration),
        better_start + 1,
        max_end,
    )
    reason = str(data.get("reason", "")).strip()[:240]

    return {
        "viral_score": viral_score,
        "reason": reason,
        "better_start_offset": better_start,
        "better_end_offset": better_end,
    }


def rerank_moments(
    moments: list[dict],
    clip_duration: int,
    keep: int,
    model: str = DEFAULT_MODEL,
    timeout: int = 20,
    on_progress=None,
) -> list[dict] | None:
    """Score candidates and return the best moments, or None if scoring fails.

    When a moment carries ``vision_meta`` (from ``vision_analyzer``), the
    candidate is scored with the multimodal ``vision_score_candidate`` and the
    final combined score blends text-AI, vision and person-presence:

        score = ai_score*0.65 + vision_score*0.25 + visual_score*0.10
    """
    scored: list[dict] = []
    total = len(moments)

    for idx, moment in enumerate(moments, 1):
        transcript = moment.get("transcript", "")
        vmeta = moment.get("vision_meta")
        if vmeta:
            score = vision_score_candidate(
                transcript, moment, clip_duration=clip_duration,
                vision_meta=vmeta, model=model, timeout=timeout,
            )
        else:
            score = score_candidate(
                transcript,
                moment,
                clip_duration=clip_duration,
                model=model,
                timeout=timeout,
            )
        if on_progress:
            on_progress(idx, total, score)
        if not score:
            continue

        enriched = dict(moment)
        enriched["ai_score"] = float(score["viral_score"] or 0.0)
        enriched["ai_reason"] = score["reason"]
        vision_score = float((vmeta or {}).get("highlight_score", 0.0) or 0.0)
        enriched["vision_score"] = vision_score
        presence = float(enriched.get("visual_score", 1.0) or 1.0)
        # 65% text AI + 25% vision + 10% person-presence (weak for gaming footage)
        enriched["score"] = (
            enriched["ai_score"] * 0.65
            + vision_score * 0.25
            + presence * 0.10
        )
        enriched["ai_better_start_offset"] = score["better_start_offset"]
        enriched["ai_better_end_offset"] = score["better_end_offset"]
        scored.append(enriched)

    if not scored:
        return None

    scored.sort(
        key=lambda m: float(m.get("score", 0.0) or 0.0),
        reverse=True,
    )
    selected = scored[:keep]
    selected.sort(key=lambda m: int(m.get("start", 0)))
    return selected
