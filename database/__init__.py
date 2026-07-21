from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def _get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=_get_engine()
        )
    return _SessionLocal


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    Base.metadata.create_all(bind=_get_engine())


def drop_db() -> None:
    Base.metadata.drop_all(bind=_get_engine())


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
