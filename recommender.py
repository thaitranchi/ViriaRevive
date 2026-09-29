"""AI recommendation engine using vector search + historical performance data.

Similarity lookups go through the pgvector HNSW index (``repository.find_similar_clips``)
so candidate selection stays index-backed as the clip archive grows. The scoring
helpers are pure functions so they can be tested without a database.
"""

import logging

logger = logging.getLogger(__name__)

# How many nearest clips to pull from the index before filtering.
CANDIDATE_POOL = 50

# How many of those actually drive the recommendation.
TOP_K = 5

# A view count this low means the upload was brand new when we last polled, so it
# should not count as evidence of what works.
MIN_VIEWS_FOR_SIGNAL = 100

# Cosine similarity below this is noise rather than a genuine match.
MIN_SIMILARITY = 0.25


def _load_similar_uploaded_clips(transcript: str, limit: int = CANDIDATE_POOL) -> list[dict]:
    """Nearest uploaded clips to ``transcript``, via the vector index.

    Returns dicts with ``clip_id``, ``duration``, ``subtitle_style``,
    ``crop_params``, ``title``, ``views``, ``likes``, ``comments``, ``similarity``.
    Returns an empty list when the embedding model or database is unavailable.
    """
    from embedder import generate_text_embedding

    query_vec = generate_text_embedding(transcript) if transcript and transcript.strip() else None
    if query_vec is None:
        logger.debug("No query embedding — skipping similarity search")
        return []

    try:
        from database import get_session
        from database.repository import find_similar_clips
    except Exception as e:
        logger.debug("Recommender DB import failed: %s", e)
        return []

    try:
        with get_session() as session:
            hits = find_similar_clips(
                session, query_vec, limit=limit, with_stats=True,
                min_similarity=MIN_SIMILARITY,
            )

            results = []
            for row, similarity in hits:
                clip = row.clip
                upload = clip.upload if clip else None
                if upload is None or not upload.title:
                    continue
                if (upload.view_count or 0) < MIN_VIEWS_FOR_SIGNAL:
                    continue
                results.append({
                    "clip_id": row.clip_id,
                    "duration": clip.duration,
                    "subtitle_style": clip.subtitle_style,
                    "crop_params": clip.crop_params,
                    "title": upload.title,
                    "views": upload.view_count or 0,
                    "likes": upload.like_count or 0,
                    "comments": upload.comment_count or 0,
                    "similarity": similarity,
                })
            return results
    except Exception as e:
        logger.debug("Vector similarity lookup failed: %s", e)
        return []


def _engagement_rate(row: dict) -> float:
    """Interactions per view, clamped to [0, 1].

    Views dominate the denominator, so a heavily-watched clip with few
    interactions scores near zero — which is the behaviour we want when ranking
    clip parameters.
    """
    views = row.get("views") or 0
    if views <= 0:
        return 0.0
    interactions = (row.get("likes") or 0) + (row.get("comments") or 0)
    return min(1.0, interactions / views)


def _score_candidates(candidates: list[dict]) -> list[dict]:
    """Blend similarity with engagement and sort best-first.

    Similarity dominates; engagement breaks ties among similarly relevant clips.
    """
    for row in candidates:
        row.setdefault("similarity", 0.0)
        row["engagement"] = _engagement_rate(row)
        row["score"] = row["similarity"] * (1.0 + row["engagement"])
    return sorted(candidates, key=lambda r: r["score"], reverse=True)


def recommend_clip_params(transcript: str) -> dict:
    """Recommend clip duration / subtitle style / crop from past performance.

    Args:
        transcript: Transcript text of the clip being planned.

    Returns:
        Dict with keys: clip_duration, subtitle_style, person_crop,
        source_clips, and a confidence score.
    """
    top_k = _score_candidates(_load_similar_uploaded_clips(transcript))[:TOP_K]
    if not top_k:
        return _default_params()

    durations = [c["duration"] for c in top_k if c["duration"]]
    avg_duration = sum(durations) / len(durations) if durations else 25

    style_counts: dict[str, int] = {}
    for c in top_k:
        style = c["subtitle_style"]
        if style:
            style_counts[style] = style_counts.get(style, 0) + 1
    best_style = max(style_counts, key=style_counts.get) if style_counts else "tiktok"

    crop_vals = [
        1 if (c["crop_params"] or {}).get("enabled") else 0
        for c in top_k
    ]
    use_crop = sum(crop_vals) > len(crop_vals) / 2

    return {
        "clip_duration": round(avg_duration),
        "subtitle_style": best_style,
        "person_crop": use_crop,
        "source_clips": len(top_k),
        "confidence": round(min(1.0, len(top_k) / TOP_K), 2),
    }


def recommend_title_suggestions(
    transcript: str,
    limit: int = 5,
) -> list[dict]:
    """Find past successful clips with similar transcripts and return their titles.

    Args:
        transcript: Transcript text to search with.
        limit: Max suggestions.

    Returns:
        List of dicts: {title, views, likes, similarity_score, clip_duration}.
    """
    candidates = _load_similar_uploaded_clips(transcript, limit=limit)
    results = []
    for c in _score_candidates(candidates)[:limit]:
        results.append({
            "title": c["title"],
            "views": c["views"],
            "likes": c["likes"],
            "clip_duration": c["duration"] or 0,
            "similarity_score": round(c["similarity"], 3),
        })
    return results


def recommend_content_for_channel(
    channel_youtube_id: str,
    limit: int = 5,
) -> list[dict]:
    """Recommend videos to process next based on past success patterns.

    Candidates are the channel's own unprocessed uploads, scored against a TF-IDF
    profile of the transcripts behind its best-performing clips. Videos have no
    transcript until downloaded, so this is lexical rather than vector-based.

    Args:
        channel_youtube_id: YouTube channel ID.
        limit: Max recommendations.

    Returns:
        List of dicts: {video_id, title, reason, expected_score}.
    """
    try:
        from database import get_session
        from database.models import Channel, Video, Clip, ClipUpload
        from sqlalchemy import select
    except Exception as e:
        logger.debug("Recommender DB import failed: %s", e)
        return []

    try:
        with get_session() as session:
            channel = session.execute(
                select(Channel).where(Channel.channel_id == channel_youtube_id)
            ).scalar_one_or_none()
            if channel is None:
                return []

            top_clips = list(
                session.execute(
                    select(Clip)
                    .join(ClipUpload)
                    .where(ClipUpload.view_count >= MIN_VIEWS_FOR_SIGNAL)
                    .order_by(ClipUpload.view_count.desc())
                    .limit(10)
                ).scalars().all()
            )
            if not top_clips:
                return []

            from database.repository import get_transcripts_by_clip

            success_texts = []
            for c in top_clips:
                segs = get_transcripts_by_clip(session, c.id)
                text = " ".join(s.text for s in segs).strip()
                if text:
                    success_texts.append(text)
            if not success_texts:
                return []

            processed = set(
                session.execute(select(Video.video_id).join(Video.pipeline_runs)).scalars().all()
            )
            unprocessed = [
                v for v in session.execute(
                    select(Video).where(Video.channel_id == channel.id)
                ).scalars().all()
                if v.video_id not in processed
            ]
            if not unprocessed:
                return []

            return _score_unprocessed_videos(success_texts, unprocessed, limit)
    except Exception as e:
        logger.debug("Content recommendation failed: %s", e)
        return []


def _score_unprocessed_videos(
    success_texts: list[str],
    unprocessed: list,
    limit: int,
) -> list[dict]:
    """Rank videos by TF-IDF similarity to the success transcript profile."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        logger.debug("sklearn unavailable (%s) — skipping content recs", e)
        return []

    unproc_texts = [(v, f"{v.title} {v.description or ''}") for v in unprocessed]
    all_texts = success_texts + [t for _, t in unproc_texts]
    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=500).fit_transform(all_texts)
    except ValueError as e:
        logger.debug("TF-IDF vectorization failed: %s", e)
        return []

    success_profile = tfidf[:len(success_texts)].mean(axis=0)
    scores = cosine_similarity(tfidf[len(success_texts):], success_profile).flatten()

    ranked = sorted(zip(unproc_texts, scores), key=lambda x: x[1], reverse=True)
    return [
        {
            "video_id": v.video_id,
            "title": (v.title or "")[:100],
            "reason": "similar content to top-performing clips",
            "expected_score": round(float(score), 3),
        }
        for (v, _), score in ranked[:limit]
    ]


def _default_params() -> dict:
    return {
        "clip_duration": 25,
        "subtitle_style": "tiktok",
        "person_crop": True,
        "source_clips": 0,
        "confidence": 0.0,
    }
