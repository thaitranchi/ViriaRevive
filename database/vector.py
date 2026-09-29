"""pgvector backend: capability probing, column typing, and ANN index DDL.

The database stores clip embeddings in ``clip_embeddings.embedding``. When the
``vector`` extension is available the column is a real ``vector(N)`` so that
cosine distance is computed in PostgreSQL and served by an HNSW index. When the
extension is missing the column degrades to ``ARRAY(FLOAT)`` and similarity is
computed in Python. Callers should branch on :func:`vector_backend` instead of
assuming either layout.

Nothing here imports :mod:`database` at module scope: ``models.py`` imports this
file while :mod:`database` is still initialising its engine.
"""

import logging
from contextlib import contextmanager, nullcontext
from typing import Optional

from sqlalchemy import ARRAY, Float, text
from sqlalchemy.types import TypeDecorator

logger = logging.getLogger(__name__)

# Must match the dimensionality produced by config.EMBEDDING_MODEL.
# bge-m3 emits 1024-dim dense vectors.
VECTOR_DIM = 1024

# pgvector >= 0.5 supports HNSW; older builds only have ivfflat. We keep both
# DDL variants so a fresh install can pick the better one.
HNSW_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_clip_embeddings_hnsw "
    "ON clip_embeddings USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
)

IVFFLAT_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_clip_embeddings_ivfflat "
    "ON clip_embeddings USING ivfflat (embedding vector_cosine_ops) "
    "WITH (lists = 100)"
)

# Column typing -----------------------------------------------------------

# Set by database.init_db() when CREATE EXTENSION fails. When True the column is
# rendered as float[] instead of vector(N) so DDL matches the server.
_force_array = False


def force_array_type(value: Optional[bool] = None) -> bool:
    """Get or set the float[] fallback flag.

    Set by :func:`database.init_db` once it knows whether ``CREATE EXTENSION``
    succeeded; read as a getter everywhere else.
    """
    global _force_array
    if value is not None:
        _force_array = bool(value)
    return _force_array


class EmbeddingVector(TypeDecorator):
    """``vector(N)`` when pgvector is usable, ``float[]`` otherwise.

    The choice is made at DDL time rather than at import time. ``init_db()`` only
    learns whether ``CREATE EXTENSION`` succeeded after the models are already
    declared, and a column rendered as ``VECTOR(1024)`` on a server without the
    extension would abort ``create_all`` with "type vector does not exist".
    """

    impl = ARRAY(Float)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if force_array_type() or not _pgvector_importable():
            return dialect.type_descriptor(ARRAY(Float))
        from pgvector.sqlalchemy import Vector

        return dialect.type_descriptor(Vector(VECTOR_DIM))


def _pgvector_importable() -> bool:
    try:
        import pgvector.sqlalchemy  # noqa: F401
        return True
    except ImportError:
        return False


def embedding_column_type():
    """The type used for ``clip_embeddings.embedding``.

    Preferred form in model definitions; falls back to a plain ``ARRAY(Float)``
    when the pgvector package is not installed.
    """
    if _pgvector_importable() and not force_array_type():
        return EmbeddingVector()
    return ARRAY(Float)


def cosine_distance_expr(column, query_vec):
    """``column <=> :query`` cosine distance, or None when pgvector is unusable.

    The operator is written explicitly rather than via pgvector's comparator
    helper: the column is wrapped in a TypeDecorator whose impl is float[], so the
    comparator would not resolve to ``Vector.Comparator``.
    """
    if not use_sql_cosine():
        return None

    from pgvector.sqlalchemy import Vector
    from sqlalchemy import Float, bindparam

    query = bindparam("query_embedding", value=list(query_vec), type_=Vector(VECTOR_DIM))
    return column.op("<=>", return_type=Float)(query)


# Probes ------------------------------------------------------------------

_probe_result: Optional[bool] = None


def _probe_connection(conn) -> Optional[bool]:
    """Return True/False for vector support, or None if it could not be determined."""
    try:
        row = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).first()
        return bool(row)
    except Exception as e:  # connection is optional — DB may be offline
        logger.debug("pgvector probe failed: %s", e)
        return None


def probe_vector_support(conn=None) -> Optional[bool]:
    """Check whether the ``vector`` extension is installed on the target server.

    Args:
        conn: Optional existing SQLAlchemy connection. When omitted a short-lived
            connection is opened from the shared engine.

    Returns:
        True if ``vector`` is installed, False if the server answered and the
        extension is absent, None if the server could not be reached.
    """
    if conn is not None:
        return _probe_connection(conn)

    try:
        from database import get_engine
    except Exception as e:  # pragma: no cover - import cycle guard
        logger.debug("pgvector probe could not import engine: %s", e)
        return None

    try:
        with get_engine().connect() as c:
            return _probe_connection(c)
    except Exception as e:
        logger.debug("pgvector probe could not connect: %s", e)
        return None


def vector_backend(refresh: bool = False) -> bool:
    """Cached answer to "is pgvector usable right now?".

    Defaults to False when the database is unreachable so callers fall back to
    the pure-Python path rather than raising.
    """
    global _probe_result
    if refresh or _probe_result is None:
        _probe_result = bool(probe_vector_support())
    return _probe_result


def reset_backend_cache(reset_type: bool = True) -> None:
    """Drop the cached probe result (used by tests and by connection changes)."""
    global _probe_result
    _probe_result = None
    if reset_type:
        force_array_type(False)


def use_sql_cosine() -> bool:
    """True when cosine distance can be computed by Postgres.

    Reflects the column layout actually in the database, which is decided at
    DDL time by :func:`force_array_type` rather than at import time.
    """
    return _pgvector_importable() and not force_array_type()


# Extension bootstrap ------------------------------------------------------

@contextmanager
def _savepoint(conn):
    """Isolate DDL in a nested transaction so a failure does not abort the caller.

    PostgreSQL marks the whole transaction as failed after a rejected statement, so
    without this a denied ``CREATE EXTENSION`` would make every later statement in
    the caller's block fail with "current transaction is aborted". Connections that
    do not support savepoints (test doubles, some drivers) get a no-op.
    """
    begin = getattr(conn, "begin_nested", None)
    if begin is None:
        with nullcontext():
            yield
        return
    with begin():
        yield


def ensure_vector_extension(conn) -> bool:
    """Run ``CREATE EXTENSION IF NOT EXISTS vector``.

    Returns True when the extension is present afterwards. Requires a superuser
    or a database owner the first time; failure is non-fatal and leaves the
    connection usable for the float[] fallback.
    """
    try:
        with _savepoint(conn):
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as e:
        logger.warning("Could not create vector extension: %s", e)
        return False
    return _probe_connection(conn) is True


def create_ann_indexes(conn, use_hnsw: bool = True) -> bool:
    """Create the ANN index over ``clip_embeddings.embedding``.

    Args:
        conn: Open connection with the vector extension installed.
        use_hnsw: Prefer HNSW (pgvector >= 0.5). Falls back to ivfflat on error.

    Returns:
        True if an index was created.
    """
    order = [HNSW_INDEX_SQL, IVFFLAT_INDEX_SQL] if use_hnsw else [IVFFLAT_INDEX_SQL]
    for ddl in order:
        try:
            with _savepoint(conn):
                conn.execute(text(ddl))
            logger.info("ANN index created: %s", ddl.split(" USING ")[1].split()[0])
            return True
        except Exception as e:
            logger.warning("ANN index DDL failed: %s", e)
    return False


def embedding_column_is_vector(conn) -> bool:
    """True when ``clip_embeddings.embedding`` is stored as ``vector``.

    Used to reconcile the in-process fallback flag with the column that actually
    exists: ``create_all`` never alters an existing column, so a pre-pgvector
    database keeps ``float[]`` even when the extension is now installable.
    """
    try:
        row = conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'clip_embeddings' "
                "AND column_name = 'embedding' AND udt_name = 'vector'"
            )
        ).first()
        return bool(row)
    except Exception as e:
        logger.debug("column type probe failed: %s", e)
        return False


def table_exists(conn, table: str = "clip_embeddings") -> bool:
    try:
        row = conn.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = :t"
            ),
            {"t": table},
        ).first()
        return bool(row)
    except Exception as e:
        logger.debug("table probe failed: %s", e)
        return False


def alembic_revision(conn) -> Optional[str]:
    """Current Alembic head stamped in the database, or None if unmanaged."""
    try:
        row = conn.execute(text("SELECT version_num FROM alembic_version")).first()
        return row[0] if row else None
    except Exception:
        return None


def has_ann_index(conn) -> bool:
    """True when an HNSW or ivfflat index exists over clip_embeddings."""
    try:
        row = conn.execute(
            text(
                "SELECT 1 FROM pg_indexes "
                "WHERE tablename = 'clip_embeddings' "
                "AND indexdef ILIKE '%USING hnsw%' "
                "UNION ALL "
                "SELECT 1 FROM pg_indexes "
                "WHERE tablename = 'clip_embeddings' "
                "AND indexdef ILIKE '%USING ivfflat%' "
                "LIMIT 1"
            )
        ).first()
        return bool(row)
    except Exception as e:
        logger.debug("ANN index probe failed: %s", e)
        return False


# Search tuning -----------------------------------------------------------

# Higher ef_search trades recall for latency. 40 is pgvector's default; 100 is
# a reasonable middle ground for interactive GUI lookups over <1M rows.
HNSW_EF_SEARCH = 100

SET_EF_SEARCH_SQL = "SET LOCAL hnsw.ef_search = %d"


def hnsw_ef_search_sql(ef: int = HNSW_EF_SEARCH) -> str:
    """SQL for raising the HNSW candidate list size for this transaction."""
    return SET_EF_SEARCH_SQL % int(ef)


# Python fallback ---------------------------------------------------------

def cosine_similarity(a, b) -> float:
    """Cosine similarity of two equal-length dense vectors.

    Returns 0.0 for zero-length, mismatched, or degenerate inputs so callers
    never have to guard against ZeroDivisionError.
    """
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        denom = (na * nb) ** 0.5
        return dot / denom if denom else 0.0

    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.shape != vb.shape or va.size == 0:
        return 0.0
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm == 0.0:
        return 0.0
    return float(np.dot(va, vb) / norm)
