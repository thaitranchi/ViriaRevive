"""AI recommendation engine using vector search + historical performance data."""

import logging
from typing import Optional

from embedder import generate_text_embedding

logger = logging.getLogger(__name__)


def recommend_clip_params(transcript: str) -> dict:
    """Recommend optimal clip parameters based on past successful clips.

    Args:
        transcript: Transcript text of the clip being planned.

    Returns:
        Dict with keys: clip_duration, subtitle_style, person_crop,
        and a confidence score.
    """
    try:
        from database import get_session
        from database.models import Clip, ClipEmbedding, ClipUpload
        from sqlalchemy import select

        query_vec = generate_text_embedding(transcript)
        if query_vec is None:
            return _default_params()

        with get_session() as session:
            # Find clips with uploads that performed above average
            from database.repository import _l2_distance

            all_embeddings = list(
                session.execute(
                    select(ClipEmbedding).join(Clip).join(ClipUpload)
                ).scalars().all()
            )

            if not all_embeddings:
                return _default_params()

            # Score each: similarity + normalize by engagement
            scored = []
            for emb in all_embeddings:
                if emb.clip is None or emb.clip.upload is None:
                    continue
                sim = 1.0 / (1.0 + _l2_distance(emb.embedding, query_vec))
                upload = emb.clip.upload
                views = upload.view_count or 0
                scored.append((sim * (1 + views / max(1, views + 100)), emb.clip))

            if not scored:
                return _default_params()

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = scored[:5]

            durations = [c.duration for _, c in top_k if c.duration]
            avg_duration = sum(durations) / len(durations) if durations else 25

            styles = [c.subtitle_style for _, c in top_k if c.subtitle_style]
            style_counts = {}
            for s in styles:
                style_counts[s] = style_counts.get(s, 0) + 1
            best_style = max(style_counts, key=style_counts.get) if style_counts else "tiktok"

            crop_vals = []
            for _, c in top_k:
                params = c.crop_params or {}
                crop_vals.append(1 if params.get("enabled") else 0)
            use_crop = sum(crop_vals) > len(crop_vals) / 2

            confidence = len(scored) / max(1, len(all_embeddings))

            return {
                "clip_duration": round(avg_duration),
                "subtitle_style": best_style,
                "person_crop": use_crop,
                "source_clips": len(top_k),
                "confidence": round(confidence, 2),
            }
    except Exception as e:
        logger.debug("Param recommendation failed: %s", e)
        return _default_params()


def recommend_title_suggestions(
    transcript: str,
    limit: int = 5,
) -> list[dict]:
    """Find past successful clips with similar transcripts and return their titles.

    Args:
        transcript: Transcript text to search with.
        limit: Max suggestions.

    Returns:
        List of dicts: {title, views, similarity_score, clip_duration}.
    """
    try:
        from database import get_session
        from database.models import Clip, ClipEmbedding, ClipUpload
        from sqlalchemy import select
        from database.repository import _l2_distance

        query_vec = generate_text_embedding(transcript)
        if query_vec is None:
            return []

        with get_session() as session:
            all_embeddings = list(
                session.execute(
                    select(ClipEmbedding).join(Clip).join(ClipUpload)
                ).scalars().all()
            )

            scored = []
            for emb in all_embeddings:
                if emb.clip is None or emb.clip.upload is None:
                    continue
                if not emb.clip.upload.title:
                    continue
                dist = _l2_distance(emb.embedding, query_vec)
                sim = 1.0 / (1.0 + dist)
                upload = emb.clip.upload
                scored.append({
                    "title": upload.title,
                    "views": upload.view_count or 0,
                    "likes": upload.like_count or 0,
                    "clip_duration": emb.clip.duration or 0,
                    "similarity_score": round(sim, 3),
                })

            scored.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored[:limit]
    except Exception as e:
        logger.debug("Title recommendation failed: %s", e)
        return []


def recommend_content_for_channel(
    channel_youtube_id: str,
    limit: int = 5,
) -> list[dict]:
    """Recommend videos to process next based on past success patterns.

    Args:
        channel_youtube_id: YouTube channel ID.
        limit: Max recommendations.

    Returns:
        List of dicts: {video_id, title, reason, expected_score}.
    """
    try:
        from database import get_session
        from database.models import Channel, Video, Clip, ClipEmbedding, ClipUpload
        from sqlalchemy import select

        with get_session() as session:
            channel = session.execute(
                select(Channel).where(Channel.channel_id == channel_youtube_id)
            ).scalar_one_or_none()
            if channel is None:
                return []

            # Get top-performing clips for this channel
            top_clips = list(
                session.execute(
                    select(Clip)
                    .join(Clip.pipeline_run)
                    .join(ClipUpload)
                    .where(ClipUpload.view_count > 0)
                    .order_by(ClipUpload.view_count.desc())
                    .limit(10)
                ).scalars().all()
            )

            if not top_clips:
                return []

            # Get their transcripts
            top_transcripts = []
            for c in top_clips:
                from database.repository import get_transcripts_by_clip
                segs = get_transcripts_by_clip(session, c.id)
                text = " ".join(s.text for s in segs).strip()
                if text:
                    top_transcripts.append(text)

            if not top_transcripts:
                return []

            # Embed them and find average vector
            from embedder import _load_model
            model = _load_model()
            if model is None:
                return []
            import numpy as np
            vecs = model.encode(top_transcripts, normalize_embeddings=True)
            avg_vec = np.mean(vecs, axis=0).tolist()

            # Find other videos from same channel not yet processed
            processed_video_ids = set(
                session.execute(
                    select(Video.video_id).join(Video.pipeline_runs)
                ).scalars().all()
            )

            all_channel_videos = list(
                session.execute(
                    select(Video).where(
                        Video.channel_id == channel.id,
                    )
                ).scalars().all()
            )

            unprocessed = [
                v for v in all_channel_videos
                if v.video_id not in processed_video_ids
            ]

            if not unprocessed:
                return []

            # Score unprocessed by how similar they are to successful clips
            unproc_texts = [
                (v, f"{v.title} {v.description or ''}") for v in unprocessed
            ]
            # Simple title-keyword overlap scoring (no full transcript available)
            from sklearn.feature_extraction.text import TfidfVectorizer
            all_texts = top_transcripts + [t for _, t in unproc_texts]
            vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
            tfidf = vectorizer.fit_transform(all_texts)
            success_profile = tfidf[:len(top_transcripts)].mean(axis=0)
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(
                tfidf[len(top_transcripts):], success_profile
            ).flatten()

            results = []
            for (v, _), score in sorted(
                zip(unproc_texts, scores), key=lambda x: x[1], reverse=True
            )[:limit]:
                results.append({
                    "video_id": v.video_id,
                    "title": v.title[:100],
                    "reason": "similar content to top-performing clips",
                    "expected_score": round(float(score), 3),
                })
            return results
    except Exception as e:
        logger.debug("Content recommendation failed: %s", e)
        return []


def _default_params() -> dict:
    return {
        "clip_duration": 25,
        "subtitle_style": "tiktok",
        "person_crop": True,
        "source_clips": 0,
        "confidence": 0.0,
    }
