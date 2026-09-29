"""YouTube Data API collector — stores channel, video, comment, and trending data into PostgreSQL."""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _ensure_db():
    """Session context manager that creates the schema on first use.

    Raises when the database is unreachable so callers' existing error handling
    takes over — the collector is always best-effort.
    """
    from database import ensure_schema, get_session

    if not ensure_schema():
        raise RuntimeError("database unreachable")
    return get_session()


def collect_channel_metadata(service, youtube_channel_id: str = None) -> Optional[int]:
    """Fetch channel info from YouTube API and upsert into DB.

    Args:
        service: Authenticated YouTube API service object.
        youtube_channel_id: If None, fetches the authenticated user's channel(s).

    Returns:
        The DB id of the upserted channel, or None on failure.
    """
    try:
        if youtube_channel_id:
            resp = service.channels().list(
                part="snippet,statistics", id=youtube_channel_id
            ).execute()
        else:
            resp = service.channels().list(
                part="snippet,statistics", mine=True
            ).execute()

        items = resp.get("items", [])
        if not items:
            logger.warning("No channel data returned from YouTube API")
            return None

        from database.repository import upsert_channel

        db_id = None
        for ch in items:
            snippet = ch.get("snippet", {})
            stats = ch.get("statistics", {})
            with _ensure_db() as session:
                channel = upsert_channel(
                    session,
                    channel_id=ch["id"],
                    title=snippet.get("title", ""),
                    subscriber_count=int(stats.get("subscriberCount", 0)),
                    total_views=int(stats.get("viewCount", 0)),
                    channel_thumbnail=(
                        snippet.get("thumbnails", {})
                        .get("default", {})
                        .get("url")
                    ),
                )
                if db_id is None:
                    db_id = channel.id
        return db_id
    except Exception as e:
        logger.error("Failed to collect channel metadata: %s", e)
        return None


def collect_video_metadata(
    service,
    video_id: str,
    channel_db_id: Optional[int] = None,
) -> Optional[int]:
    """Fetch video metadata from YouTube API and upsert into DB.

    Args:
        service: Authenticated YouTube API service object.
        video_id: YouTube video ID.
        channel_db_id: Optional DB channel id (resolved if not provided).

    Returns:
        The DB id of the upserted video, or None on failure.
    """
    try:
        resp = service.videos().list(
            part="snippet,statistics,contentDetails", id=video_id
        ).execute()
        items = resp.get("items", [])
        if not items:
            logger.warning("No video data returned for video_id=%s", video_id)
            return None

        vid = items[0]
        snippet = vid.get("snippet", {})
        stats = vid.get("statistics", {})
        content = vid.get("contentDetails", {})

        # Parse ISO 8601 duration
        duration_sec = _parse_iso_duration(content.get("duration", "PT0S"))

        channel_youtube_id = snippet.get("channelId")

        from database.repository import get_channel_by_youtube_id, upsert_video

        if channel_db_id is None and channel_youtube_id:
            with _ensure_db() as session:
                ch = get_channel_by_youtube_id(session, channel_youtube_id)
                if ch is not None:
                    channel_db_id = ch.id

        with _ensure_db() as session:
            video = upsert_video(
                session,
                video_id=vid["id"],
                channel_id=channel_db_id or 0,
                title=snippet.get("title", ""),
                description=snippet.get("description", ""),
                tags=snippet.get("tags"),
                category=snippet.get("categoryId"),
                duration_sec=duration_sec,
                view_count=int(stats.get("viewCount", 0)),
                like_count=int(stats.get("likeCount", 0)),
                comment_count=int(stats.get("commentCount", 0)),
                publish_date=_parse_datetime(snippet.get("publishedAt")),
            )
            return video.id
    except Exception as e:
        logger.error("Failed to collect video metadata for %s: %s", video_id, e)
        return None


def update_video_stats(service, video_id: str) -> bool:
    """Fetch current stats for a video and append a row to video_stats_history.

    Args:
        service: Authenticated YouTube API service object.
        video_id: YouTube video ID.

    Returns:
        True on success.
    """
    try:
        resp = service.videos().list(
            part="statistics", id=video_id
        ).execute()
        items = resp.get("items", [])
        if not items:
            logger.warning("No stats returned for video_id=%s", video_id)
            return False

        stats = items[0].get("statistics", {})
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))

        from database.repository import get_video_by_youtube_id, insert_video_stats

        with _ensure_db() as session:
            video = get_video_by_youtube_id(session, video_id)
            if video is None:
                logger.warning("Video %s not found in DB — skipping stats insert", video_id)
                return False
            insert_video_stats(session, video.id, view_count, like_count, comment_count)
            # Also update the latest counts on the video record
            video.view_count = view_count
            video.like_count = like_count
            video.comment_count = comment_count
            video.last_stat_fetch = datetime.now(timezone.utc)
        return True
    except Exception as e:
        logger.error("Failed to update video stats for %s: %s", video_id, e)
        return False


def collect_comments(
    service,
    video_id: str,
    max_results: int = 100,
) -> int:
    """Fetch comments for a video and store in DB.

    Args:
        service: Authenticated YouTube API service object.
        video_id: YouTube video ID.
        max_results: Maximum number of comments to fetch.

    Returns:
        Number of comments stored.
    """
    try:
        from database.repository import get_video_by_youtube_id, upsert_comment

        with _ensure_db() as session:
            video = get_video_by_youtube_id(session, video_id)
            if video is None:
                logger.warning("Video %s not found in DB — skipping comments", video_id)
                return 0

        count = 0
        next_page = None
        while count < max_results:
            params = dict(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - count),
                order="relevance",
            )
            if next_page:
                params["pageToken"] = next_page

            resp = service.commentThreads().list(**params).execute()
            items = resp.get("items", [])
            if not items:
                break

            with _ensure_db() as session:
                for item in items:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    upsert_comment(
                        session,
                        comment_id=item["id"],
                        video_id=video.id,
                        author=snippet.get("authorDisplayName", "unknown"),
                        text=snippet.get("textDisplay", ""),
                        like_count=int(snippet.get("likeCount", 0)),
                        published_at=_parse_datetime(snippet.get("publishedAt")),
                    )
                    count += 1

            next_page = resp.get("nextPageToken")
            if not next_page:
                break

        logger.info("Stored %d comments for video %s", count, video_id)
        return count
    except Exception as e:
        logger.error("Failed to collect comments for %s: %s", video_id, e)
        return 0


def collect_trending(
    service,
    region_code: str = "US",
    category_id: str = None,
    max_results: int = 50,
) -> int:
    """Fetch trending videos for a region and store in DB.

    Args:
        service: Authenticated YouTube API service object.
        region_code: ISO 3166-1 alpha-2 region code.
        category_id: Optional YouTube category ID filter.
        max_results: Maximum videos to fetch.

    Returns:
        Number of trending videos stored.
    """
    try:
        params = dict(
            part="snippet,statistics,contentDetails",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=min(50, max_results),
        )
        if category_id:
            params["categoryId"] = category_id

        resp = service.videos().list(**params).execute()
        items = resp.get("items", [])
        if not items:
            return 0

        from database.repository import upsert_video, get_channel_by_youtube_id

        count = 0
        with _ensure_db() as session:
            for vid in items:
                snippet = vid.get("snippet", {})
                stats = vid.get("statistics", {})
                content = vid.get("contentDetails", {})

                channel_youtube_id = snippet.get("channelId")
                channel_db_id = None
                if channel_youtube_id:
                    ch = get_channel_by_youtube_id(session, channel_youtube_id)
                    if ch is not None:
                        channel_db_id = ch.id

                upsert_video(
                    session,
                    video_id=vid["id"],
                    channel_id=channel_db_id or 0,
                    title=snippet.get("title", ""),
                    description=snippet.get("description", ""),
                    tags=snippet.get("tags"),
                    category=snippet.get("categoryId"),
                    duration_sec=_parse_iso_duration(content.get("duration", "PT0S")),
                    view_count=int(stats.get("viewCount", 0)),
                    like_count=int(stats.get("likeCount", 0)),
                    comment_count=int(stats.get("commentCount", 0)),
                    publish_date=_parse_datetime(snippet.get("publishedAt")),
                )
                count += 1
        return count
    except Exception as e:
        logger.error("Failed to collect trending for %s: %s", region_code, e)
        return 0


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_iso_duration(iso: str) -> Optional[int]:
    """Parse ISO 8601 duration string to seconds."""
    if not iso:
        return None
    try:
        import re
        m = re.match(
            r"PT?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
            iso,
        )
        if not m:
            return None
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = int(m.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return None


def _parse_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    """Parse RFC 3339 datetime string."""
    if not iso_str:
        return None
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt
    except Exception:
        return None
