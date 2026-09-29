from datetime import datetime
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session, joinedload

from config import EMBEDDING_MODEL as DEFAULT_EMBEDDING_MODEL

from database import vector
from database.models import (
    Channel,
    Video,
    VideoStatsHistory,
    Comment,
    PipelineRun,
    Clip,
    ClipTranscript,
    ClipUpload,
    ClipEmbedding,
    UserPreference,
)


# ── Channel CRUD ──────────────────────────────────────────────────────────

def upsert_channel(
    session: Session,
    channel_id: str,
    title: str,
    subscriber_count: Optional[int] = None,
    total_views: Optional[int] = None,
    channel_thumbnail: Optional[str] = None,
) -> Channel:
    stmt = select(Channel).where(Channel.channel_id == channel_id)
    channel = session.execute(stmt).scalar_one_or_none()
    if channel is None:
        channel = Channel(channel_id=channel_id)
        session.add(channel)
    channel.title = title
    channel.subscriber_count = subscriber_count
    channel.total_views = total_views
    channel.channel_thumbnail = channel_thumbnail
    channel.last_fetched = datetime.now()
    session.flush()
    return channel


def get_channel_by_youtube_id(session: Session, channel_id: str) -> Optional[Channel]:
    return session.execute(
        select(Channel).where(Channel.channel_id == channel_id)
    ).scalar_one_or_none()


# ── Video CRUD ────────────────────────────────────────────────────────────

def upsert_video(
    session: Session,
    video_id: str,
    channel_id: int,
    title: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    category: Optional[str] = None,
    duration_sec: Optional[int] = None,
    view_count: Optional[int] = None,
    like_count: Optional[int] = None,
    comment_count: Optional[int] = None,
    publish_date: Optional[datetime] = None,
) -> Video:
    stmt = select(Video).where(Video.video_id == video_id)
    video = session.execute(stmt).scalar_one_or_none()
    if video is None:
        video = Video(video_id=video_id, channel_id=channel_id)
        session.add(video)
    video.title = title
    video.description = description
    video.tags = tags
    video.category = category
    video.duration_sec = duration_sec
    video.view_count = view_count
    video.like_count = like_count
    video.comment_count = comment_count
    video.publish_date = publish_date
    video.last_stat_fetch = datetime.now()
    session.flush()
    return video


def get_video_by_youtube_id(session: Session, video_id: str) -> Optional[Video]:
    return session.execute(
        select(Video).where(Video.video_id == video_id)
    ).scalar_one_or_none()


# ── Video Stats History ──────────────────────────────────────────────────

def insert_video_stats(
    session: Session,
    video_id: int,
    view_count: int,
    like_count: Optional[int] = None,
    comment_count: Optional[int] = None,
) -> VideoStatsHistory:
    stat = VideoStatsHistory(
        video_id=video_id,
        view_count=view_count,
        like_count=like_count,
        comment_count=comment_count,
    )
    session.add(stat)
    session.flush()
    return stat


def get_video_stats_history(session: Session, video_id: int) -> list[VideoStatsHistory]:
    stmt = (
        select(VideoStatsHistory)
        .where(VideoStatsHistory.video_id == video_id)
        .order_by(VideoStatsHistory.fetched_at)
    )
    return list(session.execute(stmt).scalars().all())


# ── Comment CRUD ──────────────────────────────────────────────────────────

def upsert_comment(
    session: Session,
    comment_id: str,
    video_id: int,
    author: str,
    text: str,
    like_count: int = 0,
    published_at: Optional[datetime] = None,
    sentiment_score: Optional[float] = None,
) -> Comment:
    stmt = select(Comment).where(Comment.comment_id == comment_id)
    comment = session.execute(stmt).scalar_one_or_none()
    if comment is None:
        comment = Comment(comment_id=comment_id, video_id=video_id)
        session.add(comment)
    comment.author = author
    comment.text = text
    comment.like_count = like_count
    comment.published_at = published_at
    comment.sentiment_score = sentiment_score
    session.flush()
    return comment


def get_comments_by_video(session: Session, video_id: int) -> list[Comment]:
    stmt = (
        select(Comment)
        .where(Comment.video_id == video_id)
        .order_by(Comment.like_count.desc())
    )
    return list(session.execute(stmt).scalars().all())


# ── Pipeline Run CRUD ─────────────────────────────────────────────────────

def create_pipeline_run(
    session: Session,
    video_id: int,
    config_snapshot: Optional[dict] = None,
) -> PipelineRun:
    run = PipelineRun(video_id=video_id, config_snapshot=config_snapshot)
    session.add(run)
    session.flush()
    return run


def complete_pipeline_run(
    session: Session,
    run_id: int,
    status: str = "completed",
    error_message: Optional[str] = None,
) -> Optional[PipelineRun]:
    run = session.get(PipelineRun, run_id)
    if run is None:
        return None
    run.status = status
    run.completed_at = datetime.now()
    run.error_message = error_message
    return run


# ── Clip CRUD ─────────────────────────────────────────────────────────────

def create_clip(
    session: Session,
    pipeline_run_id: int,
    clip_index: int,
    start_offset: float,
    end_offset: float,
    duration: float,
    heuristic_score: Optional[float] = None,
    ai_score: Optional[float] = None,
    final_score: Optional[float] = None,
    vision_score: Optional[float] = None,
    person_presence: Optional[float] = None,
    subtitle_style: Optional[str] = None,
    crop_params: Optional[dict] = None,
) -> Clip:
    clip = Clip(
        pipeline_run_id=pipeline_run_id,
        clip_index=clip_index,
        start_offset=start_offset,
        end_offset=end_offset,
        duration=duration,
        heuristic_score=heuristic_score,
        ai_score=ai_score,
        final_score=final_score,
        vision_score=vision_score,
        person_presence=person_presence,
        subtitle_style=subtitle_style,
        crop_params=crop_params,
    )
    session.add(clip)
    session.flush()
    return clip


def get_clips_by_run(session: Session, pipeline_run_id: int) -> list[Clip]:
    stmt = (
        select(Clip)
        .where(Clip.pipeline_run_id == pipeline_run_id)
        .order_by(Clip.clip_index)
    )
    return list(session.execute(stmt).scalars().all())


# ── Clip Transcript ───────────────────────────────────────────────────────

def create_clip_transcript(
    session: Session,
    clip_id: int,
    start_offset: float,
    end_offset: float,
    text: str,
) -> ClipTranscript:
    seg = ClipTranscript(
        clip_id=clip_id,
        start_offset=start_offset,
        end_offset=end_offset,
        text=text,
    )
    session.add(seg)
    session.flush()
    return seg


def get_transcripts_by_clip(session: Session, clip_id: int) -> list[ClipTranscript]:
    stmt = (
        select(ClipTranscript)
        .where(ClipTranscript.clip_id == clip_id)
        .order_by(ClipTranscript.start_offset)
    )
    return list(session.execute(stmt).scalars().all())


# ── Clip Upload ───────────────────────────────────────────────────────────

def upsert_clip_upload(
    session: Session,
    clip_id: int,
    youtube_video_id: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    privacy_status: Optional[str] = None,
    scheduled_at: Optional[datetime] = None,
    uploaded_at: Optional[datetime] = None,
    view_count: Optional[int] = None,
    like_count: Optional[int] = None,
    comment_count: Optional[int] = None,
) -> ClipUpload:
    stmt = select(ClipUpload).where(ClipUpload.clip_id == clip_id)
    upload = session.execute(stmt).scalar_one_or_none()
    if upload is None:
        upload = ClipUpload(clip_id=clip_id)
        session.add(upload)
    if youtube_video_id is not None:
        upload.youtube_video_id = youtube_video_id
    if title is not None:
        upload.title = title
    if description is not None:
        upload.description = description
    if tags is not None:
        upload.tags = tags
    if privacy_status is not None:
        upload.privacy_status = privacy_status
    if scheduled_at is not None:
        upload.scheduled_at = scheduled_at
    if uploaded_at is not None:
        upload.uploaded_at = uploaded_at
    if view_count is not None:
        upload.view_count = view_count
    if like_count is not None:
        upload.like_count = like_count
    if comment_count is not None:
        upload.comment_count = comment_count
    session.flush()
    return upload


def update_upload_stats(
    session: Session,
    clip_upload_id: int,
    view_count: int,
    like_count: Optional[int] = None,
    comment_count: Optional[int] = None,
) -> Optional[ClipUpload]:
    upload = session.get(ClipUpload, clip_upload_id)
    if upload is None:
        return None
    upload.view_count = view_count
    upload.like_count = like_count
    upload.comment_count = comment_count
    return upload


# ── Clip Embedding ────────────────────────────────────────────────────────

def create_clip_embedding(
    session: Session,
    clip_id: int,
    embedding: list[float],
    model_name: Optional[str] = None,
) -> ClipEmbedding:
    emb = ClipEmbedding(
        clip_id=clip_id,
        embedding=embedding,
        model_name=model_name or DEFAULT_EMBEDDING_MODEL,
    )
    session.add(emb)
    session.flush()
    return emb


def _cosine_distance_sql(query_vec: list[float]):
    """``embedding <=> :query`` cosine distance, or None on the float[] path.

    Ordering by this expression lets Postgres satisfy the query from the HNSW
    index instead of scanning every embedding.
    """
    return vector.cosine_distance_expr(ClipEmbedding.embedding, query_vec)


def find_similar_clips(
    session: Session,
    embedding: list[float],
    limit: int = 10,
    with_stats: bool = False,
    min_similarity: Optional[float] = None,
) -> list[tuple[ClipEmbedding, float]]:
    """Find clips whose transcript embedding is closest to ``embedding``.

    Uses pgvector's cosine distance (HNSW-indexed) when available, otherwise
    falls back to a Python scan. The fallback is exact but reads the whole
    table, so it is only appropriate for the thousands-of-rows scale.

    Args:
        embedding: Query vector, already normalized by the embedder.
        limit: Max rows to return.
        with_stats: Eager-load the clip and its upload stats.
        min_similarity: Drop hits below this cosine similarity (0..1).

    Returns:
        List of ``(ClipEmbedding, similarity)`` pairs, most similar first.
    """
    if not embedding:
        return []

    distance = _cosine_distance_sql(embedding)
    opts = [joinedload(ClipEmbedding.clip).joinedload(Clip.upload)] if with_stats else []

    if distance is None:
        # float[] fallback: exact but a full scan. Filter and cap in SQL so we
        # still avoid materialising every 1024-float array into Python.
        # Over-fetch, then apply the threshold in Python: an approximate vector
        # bound would need the raw norms, which are not stored.
        rows = list(session.execute(select(ClipEmbedding).options(*opts)).scalars().all())
        scored = [
            (row, vector.cosine_similarity(row.embedding, embedding)) for row in rows
        ]
        if min_similarity is not None:
            scored = [s for s in scored if s[1] >= min_similarity]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    stmt = select(ClipEmbedding, distance.label("distance")).options(*opts)
    if min_similarity is not None:
        # similarity = 1 - distance
        stmt = stmt.where(distance <= 1.0 - min_similarity)
    stmt = stmt.order_by(distance).limit(limit)

    return [
        (row, 1.0 - float(dist))
        for row, dist in session.execute(stmt).all()
    ]


def count_embeddings(session: Session) -> int:
    """Number of embedded clips. Drives the GUI's vector-store status card."""
    return session.execute(select(func.count(ClipEmbedding.id))).scalar() or 0


# ── User Preference ───────────────────────────────────────────────────────

def set_preference(session: Session, key: str, value: dict) -> UserPreference:
    stmt = select(UserPreference).where(UserPreference.key == key)
    pref = session.execute(stmt).scalar_one_or_none()
    if pref is None:
        pref = UserPreference(key=key)
        session.add(pref)
    pref.value = value
    pref.updated_at = datetime.now()
    session.flush()
    return pref


def get_preference(session: Session, key: str) -> Optional[dict]:
    stmt = select(UserPreference).where(UserPreference.key == key)
    pref = session.execute(stmt).scalar_one_or_none()
    return pref.value if pref else None
