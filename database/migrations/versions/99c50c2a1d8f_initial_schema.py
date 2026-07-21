"""initial schema

Revision ID: 99c50c2a1d8f
Revises:
Create Date: 2026-07-21 18:16:34.006568

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '99c50c2a1d8f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "channels",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("channel_id", sa.String(64), unique=True, nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("subscriber_count", sa.Integer()),
        sa.Column("total_views", sa.Integer()),
        sa.Column("channel_thumbnail", sa.String(512)),
        sa.Column(
            "last_fetched",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key", sa.String(128), unique=True, nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.String(64), unique=True, nullable=False),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("tags", sa.JSON()),
        sa.Column("category", sa.String(64)),
        sa.Column("duration_sec", sa.Integer()),
        sa.Column("view_count", sa.Integer()),
        sa.Column("like_count", sa.Integer()),
        sa.Column("comment_count", sa.Integer()),
        sa.Column("publish_date", sa.DateTime(timezone=True)),
        sa.Column("download_date", sa.DateTime(timezone=True)),
        sa.Column("last_stat_fetch", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["channel_id"], ["channels.id"]),
    )

    op.create_table(
        "video_stats_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("view_count", sa.Integer(), nullable=False),
        sa.Column("like_count", sa.Integer()),
        sa.Column("comment_count", sa.Integer()),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )
    op.create_index(
        "ix_video_stats_history_video_fetched",
        "video_stats_history",
        ["video_id", "fetched_at"],
    )

    op.create_table(
        "comments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("comment_id", sa.String(64), unique=True, nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("author", sa.String(128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("like_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("sentiment_score", sa.Float()),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="started"),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("config_snapshot", sa.JSON()),
        sa.Column("error_message", sa.Text()),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )

    op.create_table(
        "clips",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("pipeline_run_id", sa.Integer(), nullable=False),
        sa.Column("clip_index", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Float(), nullable=False),
        sa.Column("end_offset", sa.Float(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column("heuristic_score", sa.Float()),
        sa.Column("ai_score", sa.Float()),
        sa.Column("final_score", sa.Float()),
        sa.Column("vision_score", sa.Float()),
        sa.Column("person_presence", sa.Float()),
        sa.Column("subtitle_style", sa.String(32)),
        sa.Column("crop_params", sa.JSON()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
    )

    op.create_table(
        "clip_transcripts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("clip_id", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Float(), nullable=False),
        sa.Column("end_offset", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["clip_id"], ["clips.id"]),
    )

    op.create_table(
        "clip_uploads",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("clip_id", sa.Integer(), unique=True, nullable=False),
        sa.Column("youtube_video_id", sa.String(64)),
        sa.Column("title", sa.String(512)),
        sa.Column("description", sa.Text()),
        sa.Column("tags", sa.JSON()),
        sa.Column("privacy_status", sa.String(16)),
        sa.Column("scheduled_at", sa.DateTime(timezone=True)),
        sa.Column("uploaded_at", sa.DateTime(timezone=True)),
        sa.Column("view_count", sa.Integer()),
        sa.Column("like_count", sa.Integer()),
        sa.Column("comment_count", sa.Integer()),
        sa.ForeignKeyConstraint(["clip_id"], ["clips.id"]),
    )

    op.create_table(
        "clip_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("clip_id", sa.Integer(), unique=True, nullable=False),
        sa.Column("embedding", sa.ARRAY(sa.Float()), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False, server_default="BAAI/bge-m3"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["clip_id"], ["clips.id"]),
    )


def downgrade() -> None:
    op.drop_table("clip_embeddings")
    op.drop_table("clip_uploads")
    op.drop_table("clip_transcripts")
    op.drop_table("clips")
    op.drop_table("pipeline_runs")
    op.drop_table("comments")
    op.drop_table("video_stats_history")
    op.drop_table("videos")
    op.drop_table("user_preferences")
    op.drop_table("channels")
