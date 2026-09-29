"""Rule-based viral predictor using historical data from PostgreSQL."""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ViralPredictor:
    """Adjusts heuristic scores based on historical performance patterns.

    Queries the database for past clip statistics and applies rule-based
    adjustments (boost/penalty) to new clips.
    """

    def __init__(self):
        self._averages: dict[str, float] = {}
        self._last_refresh: Optional[datetime] = None

    def refresh_stats(self) -> None:
        """Query DB for historical averages and cache them."""
        try:
            from database import ensure_schema, get_session
            from database.models import Clip, ClipUpload
            from sqlalchemy import select, func

            if not ensure_schema():
                return

            with get_session() as session:
                # Average final score across all clips
                result = session.execute(
                    select(func.avg(Clip.final_score))
                ).scalar()
                self._averages["avg_final_score"] = float(result or 0.5)

                # Average view count per uploaded clip
                result = session.execute(
                    select(func.avg(ClipUpload.view_count))
                ).scalar()
                self._averages["avg_views"] = float(result or 0.0)

                # Average engagement rate (likes/views)
                result = session.execute(
                    select(
                        func.avg(
                            ClipUpload.like_count * 1.0 /
                            func.nullif(ClipUpload.view_count, 0)
                        )
                    )
                ).scalar()
                self._averages["avg_engagement_rate"] = float(result or 0.0)

                # Count of clips processed
                result = session.execute(
                    select(func.count(Clip.id))
                ).scalar()
                self._averages["total_clips"] = int(result or 0)

                # Best score range for reference
                result = session.execute(
                    select(func.max(Clip.final_score))
                ).scalar()
                self._averages["best_score"] = float(result or 1.0)

            self._last_refresh = datetime.now(timezone.utc)
            logger.debug(
                "Predictor stats refreshed: avg_score=%.2f, total=%d",
                self._averages["avg_final_score"],
                self._averages["total_clips"],
            )
        except Exception as e:
            logger.debug("Failed to refresh predictor stats: %s", e)
            self._averages = {}

    def predict_boost(self, clip_features: Optional[dict] = None) -> float:
        """Return a score multiplier based on historical patterns.

        Args:
            clip_features: Dict with keys like 'score', 'duration',
                          'person_presence', 'channel_subscribers', etc.

        Returns:
            Multiplier in range [0.7, 1.3].
        """
        if not self._averages:
            self.refresh_stats()

        # Too little history to say anything — including none at all when the
        # database is unreachable, which leaves _averages empty.
        if self._averages.get("total_clips", 0) < 3:
            return 1.0

        boost = 1.0

        if clip_features:
            score = clip_features.get("score", 0.5)
            avg_score = self._averages.get("avg_final_score", 0.5)

            # Boost if score is above historical average
            if score > avg_score and avg_score > 0:
                boost += 0.15 * min(1.0, (score - avg_score) / avg_score)

            # Penalize if well below average
            if score < avg_score * 0.5 and avg_score > 0:
                boost -= 0.1

            # Person presence = more engaging
            person = clip_features.get("person_presence")
            if person is not None and person > 0.5:
                boost += 0.05

            # Duration sweet spot
            duration = clip_features.get("duration", 25)
            if 20 <= duration <= 35:
                boost += 0.05
            elif duration > 60:
                boost -= 0.05

        return max(0.7, min(1.3, boost))

    def adjust_moments(
        self,
        moments: list[dict],
    ) -> list[dict]:
        """Apply predictor boost to each moment's score in-place."""
        if not self._averages:
            self.refresh_stats()

        if self._averages.get("total_clips", 0) < 3:
            return moments

        for m in moments:
            boost = self.predict_boost(m)
            original = m.get("score", 0.0)
            m["score"] = original * boost
            m["predictor_boost"] = boost
            m["predictor_original_score"] = original

        return moments

    @property
    def summary(self) -> dict:
        return {
            "avg_final_score": self._averages.get("avg_final_score", 0),
            "avg_views": self._averages.get("avg_views", 0),
            "avg_engagement_rate": self._averages.get("avg_engagement_rate", 0),
            "total_clips": self._averages.get("total_clips", 0),
            "best_score": self._averages.get("best_score", 0),
            "last_refresh": str(self._last_refresh) if self._last_refresh else None,
        }
