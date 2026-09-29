"""pgvector HNSW index

Revision ID: b7e2f1a4c9d0
Revises: 99c50c2a1d8f
Create Date: 2026-07-28 11:20:00.000000

Converts clip_embeddings.embedding from float[] to vector(1024) and adds an HNSW
index so cosine similarity is served by an approximate nearest-neighbour index
instead of a sequential scan.

Fresh installs get vector(1024) directly from the initial-schema migration; this
one handles databases created before pgvector was wired up.

The migration is defensive: if the ``vector`` extension cannot be created (no
superuser on a managed instance) it leaves the float[] column in place, and the
application falls back to in-Python similarity.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7e2f1a4c9d0"
down_revision: Union[str, Sequence[str], None] = "99c50c2a1d8f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VECTOR_DIM = 1024
TABLE = "clip_embeddings"
INDEX = "ix_clip_embeddings_hnsw"
IVFFLAT_INDEX = "ix_clip_embeddings_ivfflat"

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

HNSW_INDEX_DDL = (
    f"CREATE INDEX IF NOT EXISTS {INDEX} ON {TABLE} "
    "USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
)

IVFFLAT_INDEX_DDL = (
    f"CREATE INDEX IF NOT EXISTS {IVFFLAT_INDEX} ON {TABLE} "
    "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
)

ALTER_TO_VECTOR = (
    f"ALTER TABLE {TABLE} "
    f"ALTER COLUMN embedding TYPE vector({VECTOR_DIM}) "
    f"USING embedding::vector({VECTOR_DIM})"
)

ALTER_TO_FLOAT_ARRAY = (
    f"ALTER TABLE {TABLE} "
    "ALTER COLUMN embedding TYPE float[] "
    "USING embedding::real[]"
)


def _is_offline() -> bool:
    """True during `alembic upgrade --sql`, where there is no live server.

    Offline we cannot probe the extension or the column type, so we emit the
    full DDL sequence and let the operator review it before running.
    """
    try:
        return bool(op.get_context().as_sql)
    except Exception:
        return False


def _scalar(conn, sql: str, params: dict | None = None):
    """Run a scalar query in its own savepoint.

    Uses ``text()`` rather than raw ``exec_driver_sql`` so ``:name`` placeholders
    are bound by SQLAlchemy, and rolls back on failure — an aborted transaction
    would otherwise poison every later statement in the migration.
    """
    try:
        with conn.begin_nested():
            result = conn.execute(sa.text(sql), params or {})
            return result.scalar()
    except Exception:
        return None


def _execute_safely(conn, sql: str) -> bool:
    """Run DDL in its own savepoint; return True if it succeeded.

    PostgreSQL aborts the whole transaction on a failed statement, so without the
    nested block a missing extension or an unsupported index type would leave the
    migration unable to record its version. Rolling the savepoint back keeps the
    outer transaction usable and turns the failure into a plain False.
    """
    try:
        with conn.begin_nested():
            conn.execute(sa.text(sql))
        return True
    except Exception as exc:
        print(f"[migration] {sql.split(' USING ')[0].split()[0]} failed: {exc}")
        return False


def _has_vector_extension(conn) -> bool:
    return bool(
        _scalar(conn, "SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    )


def _column_is_vector(conn) -> bool:
    return bool(
        _scalar(
            conn,
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = 'embedding' "
            "AND udt_name = 'vector'",
            {"t": TABLE},
        )
    )


def _drop_mismatched_embeddings(conn) -> None:
    """Delete rows whose stored vector length does not match VECTOR_DIM.

    The ALTER TYPE below fails outright on any mismatch, and an embedding from a
    different model is unusable anyway.
    """
    try:
        with conn.begin_nested():
            result = conn.execute(
                sa.text(
                    f"DELETE FROM {TABLE} WHERE embedding IS NOT NULL "
                    f"AND array_length(embedding, 1) <> :dim"
                ),
                {"dim": VECTOR_DIM},
            )
            if result.rowcount:
                print(f"[migration] dropped {result.rowcount} embedding(s) with wrong dimension")
    except Exception as exc:
        print(f"[migration] could not prune mismatched embeddings: {exc}")


def upgrade() -> None:
    conn = op.get_bind()

    if _is_offline():
        op.execute(CREATE_EXTENSION)
        op.execute(ALTER_TO_VECTOR)
        op.execute(HNSW_INDEX_DDL)
        return

    if not _execute_safely(conn, CREATE_EXTENSION):
        print("[migration] vector extension unavailable; leaving float[] in place")
        return

    if not _has_vector_extension(conn):
        return

    if not _column_is_vector(conn):
        _drop_mismatched_embeddings(conn)
        if not _execute_safely(conn, ALTER_TO_VECTOR):
            print("[migration] float[] -> vector conversion failed; leaving float[]")
            return

    _create_ann_index(conn)


def _create_ann_index(conn) -> None:
    """hnsw needs pgvector >= 0.5; fall back to ivfflat otherwise."""
    if _execute_safely(conn, HNSW_INDEX_DDL):
        return
    print("[migration] hnsw unavailable; trying ivfflat")
    if not _execute_safely(conn, IVFFLAT_INDEX_DDL):
        print("[migration] ivfflat unavailable; search will scan sequentially")


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {INDEX}")
    op.execute(f"DROP INDEX IF EXISTS {IVFFLAT_INDEX}")

    if _is_offline() or _column_is_vector(op.get_bind()):
        op.execute(ALTER_TO_FLOAT_ARRAY)
