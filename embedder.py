"""Text embedding via BGE-M3 (sentence-transformers) for vector search."""

import logging
from typing import Optional

from config import EMBEDDING_MODEL
from database.vector import VECTOR_DIM

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
        import os
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            model_kwargs={"use_safetensors": True},
        )
        dim = _model.get_embedding_dimension()
        if dim != VECTOR_DIM:
            # A mismatched dimension would be rejected by vector(N) on insert.
            logger.error(
                "%s produces %d-dim vectors but the column is vector(%d). "
                "Update VECTOR_DIM in database/vector.py to match.",
                EMBEDDING_MODEL, dim, VECTOR_DIM,
            )
            _model = None
            return None
        logger.info("Embedding model loaded (dim=%d)", dim)
        return _model
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        return None


def embedding_model_ready() -> bool:
    return _load_model() is not None


def generate_text_embedding(text: str) -> Optional[list[float]]:
    """Embed a single string into a normalized vector.

    Returns None when the model is unavailable or the input is empty, so callers
    can skip the write instead of inserting a NULL vector.
    """
    if not text or not text.strip():
        return None
    model = _load_model()
    if model is None:
        return None
    try:
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return None


def embed_pipeline_run(pipeline_run_id: int) -> int:
    """Embed every transcript-backed clip in a run.

    Clips that already have an embedding are skipped, so the call is cheap on a
    re-run. Returns the number of embeddings now present for the run.
    """
    from database import get_session
    from database.repository import (
        get_clips_by_run,
        get_transcripts_by_clip,
        create_clip_embedding,
    )

    model = _load_model()
    if model is None:
        logger.warning("Embedding model unavailable — skipping")
        return 0

    with get_session() as session:
        clips = get_clips_by_run(session, pipeline_run_id)
        if not clips:
            logger.warning("No clips found for pipeline_run_id=%d", pipeline_run_id)
            return 0

        count = 0
        pending = []
        for clip in clips:
            if clip.embedding is not None:
                count += 1
                continue
            segments = get_transcripts_by_clip(session, clip.id)
            full_text = " ".join(s.text for s in segments).strip()
            if not full_text:
                logger.debug("No transcript for clip %d — skipping", clip.id)
                continue
            pending.append((clip.id, full_text))

        if not pending:
            return count

        # One batched forward pass instead of one per clip.
        try:
            vectors = model.encode(
                [text for _, text in pending], normalize_embeddings=True
            )
        except Exception as e:
            logger.error("Batch embedding failed: %s", e)
            return count

        for (clip_id, _), emb in zip(pending, vectors):
            try:
                create_clip_embedding(session, clip_id, emb.tolist(), EMBEDDING_MODEL)
                count += 1
            except Exception as e:
                logger.error("Failed to store embedding for clip %d: %s", clip_id, e)

        return count
