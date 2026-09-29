from datetime import datetime
from typing import Optional

from sqlalchemy import (
    ARRAY,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base
from database.vector import embedding_column_type
from config import EMBEDDING_MODEL as DEFAULT_EMBEDDING_MODEL


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    subscriber_count: Mapped[Optional[int]]
    total_views: Mapped[Optional[int]]
    channel_thumbnail: Mapped[Optional[str]] = mapped_column(String(512))
    last_fetched: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    videos: Mapped[list["Video"]] = relationship(back_populates="channel")

    def __repr__(self) -> str:
        return f"<Channel {self.channel_id} '{self.title}'>"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"))
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON)
    category: Mapped[Optional[str]] = mapped_column(String(64))
    duration_sec: Mapped[Optional[int]]
    view_count: Mapped[Optional[int]]
    like_count: Mapped[Optional[int]]
    comment_count: Mapped[Optional[int]]
    publish_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    download_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_stat_fetch: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    channel: Mapped["Channel"] = relationship(back_populates="videos")
    stats_history: Mapped[list["VideoStatsHistory"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    comments: Mapped[list["Comment"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    pipeline_runs: Mapped[list["PipelineRun"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Video {self.video_id} '{self.title[:50]}'>"


class VideoStatsHistory(Base):
    __tablename__ = "video_stats_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    view_count: Mapped[int]
    like_count: Mapped[Optional[int]]
    comment_count: Mapped[Optional[int]]
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    video: Mapped["Video"] = relationship(back_populates="stats_history")

    __table_args__ = (
        Index("ix_video_stats_history_video_fetched", "video_id", "fetched_at"),
    )


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    comment_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    author: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    like_count: Mapped[int] = mapped_column(default=0)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    sentiment_score: Mapped[Optional[float]]

    video: Mapped["Video"] = relationship(back_populates="comments")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="started"
    )  # started | completed | failed
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    config_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    video: Mapped["Video"] = relationship(back_populates="pipeline_runs")
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="pipeline_run", cascade="all, delete-orphan"
    )


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[int] = mapped_column(primary_key=True)
    pipeline_run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_runs.id"), nullable=False
    )
    clip_index: Mapped[int]
    start_offset: Mapped[float]
    end_offset: Mapped[float]
    duration: Mapped[float]
    heuristic_score: Mapped[Optional[float]]
    ai_score: Mapped[Optional[float]]
    final_score: Mapped[Optional[float]]
    vision_score: Mapped[Optional[float]]
    person_presence: Mapped[Optional[float]]
    subtitle_style: Mapped[Optional[str]] = mapped_column(String(32))
    crop_params: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    pipeline_run: Mapped["PipelineRun"] = relationship(back_populates="clips")
    transcripts: Mapped[list["ClipTranscript"]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )
    upload: Mapped[Optional["ClipUpload"]] = relationship(
        back_populates="clip", uselist=False, cascade="all, delete-orphan"
    )
    embedding: Mapped[Optional["ClipEmbedding"]] = relationship(
        back_populates="clip", uselist=False, cascade="all, delete-orphan"
    )


class ClipTranscript(Base):
    __tablename__ = "clip_transcripts"

    id: Mapped[int] = mapped_column(primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id"), nullable=False)
    start_offset: Mapped[float]
    end_offset: Mapped[float]
    text: Mapped[str] = mapped_column(Text)

    clip: Mapped["Clip"] = relationship(back_populates="transcripts")


class ClipUpload(Base):
    __tablename__ = "clip_uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id"), unique=True, nullable=False)
    youtube_video_id: Mapped[Optional[str]] = mapped_column(String(64))
    title: Mapped[Optional[str]] = mapped_column(String(512))
    description: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON)
    privacy_status: Mapped[Optional[str]] = mapped_column(String(16))
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    uploaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    view_count: Mapped[Optional[int]]
    like_count: Mapped[Optional[int]]
    comment_count: Mapped[Optional[int]]

    clip: Mapped["Clip"] = relationship(back_populates="upload")


class ClipEmbedding(Base):
    __tablename__ = "clip_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id"), unique=True, nullable=False)
    # vector(1024) when pgvector is installed, float[] otherwise.
    embedding: Mapped[list[float]] = mapped_column(embedding_column_type())
    model_name: Mapped[str] = mapped_column(String(64), default=DEFAULT_EMBEDDING_MODEL)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    clip: Mapped["Clip"] = relationship(back_populates="embedding")

    # The HNSW index is created by database.init_db() (fresh installs) and by
    # Alembic migration b7e2f1a4c9d0 (existing ones). Declaring it here would
    # bake in the column type at import time, before we know whether the server
    # has pgvector — and Postgres rejects hnsw indexes on float[].


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
