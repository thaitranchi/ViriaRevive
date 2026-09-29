import logging

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_engine():
    """Shared SQLAlchemy engine. Used for capability probes outside a session."""
    return _get_engine()


def _get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=_get_engine()
        )
    return _SessionLocal


class Base(DeclarativeBase):
    pass


def _run_migrations() -> None:
    """Bring the database to the newest Alembic revision in-process.

    ``create_all`` only creates missing tables, so it can neither convert an
    existing ``float[]`` column to ``vector(1024)`` nor record a revision — a
    database bootstrapped that way would fail the next ``alembic upgrade head``.
    Running the migrations is therefore the primary path and ``create_all`` is
    only the fallback for installs that ship without ``alembic.ini``.
    """
    from alembic import command
    from alembic.config import Config

    ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"alembic.ini not found at {ini_path}")

    cfg = Config(str(ini_path))
    # Tell env.py not to reconfigure logging: this runs inside a live process
    # whose handlers are already set up, and fileConfig() would reset them.
    cfg.attributes["viria_inprocess"] = True
    command.upgrade(cfg, "head")


def init_db() -> None:
    """Create or migrate the schema, enabling pgvector and its ANN index.

    Safe to call repeatedly. When the ``vector`` extension is unavailable the
    embedding column stays ``float[]`` and similarity search runs in Python (see
    :mod:`database.vector`).

    Existing databases go through Alembic so the ``float[]`` -> ``vector(1024)``
    conversion is applied; the in-process fallback flag is then derived from the
    column that actually exists rather than from the DDL we just rendered.
    """
    from database import models, vector  # noqa: F401 — registers the tables

    engine = _get_engine()

    # Decide the column type before any DDL is rendered. Managed Postgres often
    # blocks CREATE EXTENSION, in which case float[] is the only option.
    extension_ready = False
    try:
        with engine.begin() as conn:
            extension_ready = vector.ensure_vector_extension(conn)
    except Exception as e:
        logger.warning("Vector setup failed (%s) — using float[] storage", e)

    if not extension_ready:
        logger.warning("pgvector unavailable — embedding column uses float[] storage")
    vector.force_array_type(not extension_ready)

    try:
        _run_migrations()
    except Exception as e:
        logger.debug("Alembic upgrade unavailable (%s); using create_all", e)
        Base.metadata.create_all(bind=engine)

    # The migration may have converted the column even when the extension probe
    # disagreed, so derive the flag from the database rather than the guess above.
    try:
        with engine.connect() as conn:
            if vector.table_exists(conn):
                vector.force_array_type(not vector.embedding_column_is_vector(conn))
    except Exception as e:
        logger.debug("column type reconciliation skipped: %s", e)

    # Index creation needs the table to exist, so it runs after create_all.
    if not vector.force_array_type():
        try:
            with engine.begin() as conn:
                if not vector.has_ann_index(conn):
                    vector.create_ann_indexes(conn)
        except Exception as e:
            logger.debug("ANN index setup skipped: %s", e)


def drop_db() -> None:
    Base.metadata.drop_all(bind=_get_engine())


_schema_ready = False


def ensure_schema(force: bool = False) -> bool:
    """Create the schema on first use. Returns False if the DB is unreachable.

    Idempotent and attempted at most once per process, so an offline database
    costs one failed connection rather than blocking every pipeline stage.
    """
    global _schema_ready
    if _schema_ready and not force:
        return True
    try:
        init_db()
        _schema_ready = True
        return True
    except Exception as e:
        logger.debug("Schema init skipped (%s)", e)
        return False


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = _get_session_local()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
