"""Text embedding via BGE-M3 (sentence-transformers) for vector search."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL

        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
        import os
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            model_kwargs={"use_safetensors": True},
        )
        logger.info(
            "Embedding model loaded (dim=%d)",
            _model.get_embedding_dimension(),
        )
        return _model
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        return None


def embedding_model_ready() -> bool:
    return _load_model() is not None


def generate_text_embedding(text: str) -> Optional[list[float]]:
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
        for clip in clips:
            if clip.embedding is not None:
                count += 1
                continue
            segments = get_transcripts_by_clip(session, clip.id)
            full_text = " ".join(s.text for s in segments).strip()
            if not full_text:
                logger.debug("No transcript for clip %d — skipping", clip.id)
                continue
            try:
                emb = model.encode(full_text, normalize_embeddings=True)
                create_clip_embedding(session, clip.id, emb.tolist())
                count += 1
            except Exception as e:
                logger.error("Failed to embed clip %d: %s", clip.id, e)

        return count
