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

    prompt = f"""You are selecting short-form viral moments.
Score this transcript from 0 to 1 for Shorts/TikTok potential.

Prefer:
- conflict
- surprise
- emotion
- clear setup/payoff
- quotable moment
- fast context

Reject:
- filler
- slow explanation
- unclear topic
- no payoff

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


def rerank_moments(
    moments: list[dict],
    clip_duration: int,
    keep: int,
    model: str = DEFAULT_MODEL,
    timeout: int = 20,
    on_progress=None,
) -> list[dict] | None:
    """Score candidates and return the best moments, or None if scoring fails."""
    scored: list[dict] = []
    total = len(moments)

    for idx, moment in enumerate(moments, 1):
        transcript = moment.get("transcript", "")
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
        # Combined score: 70% AI analysis + 30% Visual presence
        enriched["score"] = enriched["ai_score"] * 0.7 + float(enriched.get("visual_score", 1.0) or 1.0) * 0.3
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
