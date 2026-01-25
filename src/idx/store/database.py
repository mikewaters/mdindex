"""idx.store.database - SQLAlchemy engine and session management.

Provides database connectivity for the idx library using SQLite.
Uses database_path from idx.core.settings for configuration.

Example usage:
    from idx.store.database import get_engine, get_session

    engine = get_engine()
    with get_session() as session:
        # perform database operations
        pass
"""

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "get_session",
    "create_engine_for_path",
]


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all idx models."""

    pass


def _set_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
    """Set SQLite pragmas for optimal performance and data integrity.

    Enables:
    - WAL mode for better concurrent access
    - Foreign key enforcement
    - Synchronous mode for durability
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def create_engine_for_path(database_path: Path, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for the given database path.

    Creates the parent directory if it doesn't exist.

    Args:
        database_path: Path to the SQLite database file.
        echo: If True, log all SQL statements (default: False).

    Returns:
        A configured SQLAlchemy Engine instance.
    """
    # Ensure parent directory exists
    database_path.parent.mkdir(parents=True, exist_ok=True)

    # Create engine with SQLite-specific settings
    engine = create_engine(
        f"sqlite:///{database_path}",
        echo=echo,
        pool_pre_ping=True,
    )
    # Register pragma listener BEFORE using the engine
    event.listen(engine, "connect", _set_sqlite_pragmas)
    Base.metadata.create_all(engine)

    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Get the singleton SQLAlchemy engine.

    Creates the engine on first call using database_path from settings.
    Subsequent calls return the cached engine.

    Returns:
        The singleton Engine instance.
    """
    from idx.core.settings import get_settings

    settings = get_settings()
    return create_engine_for_path(settings.database_path)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Get the singleton session factory.

    Returns:
        A sessionmaker bound to the singleton engine.
    """
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for database sessions.

    Yields a session that is automatically committed on success
    and rolled back on exception.

    Yields:
        A SQLAlchemy Session instance.

    Example:
        with get_session() as session:
            session.add(some_model)
            # commits automatically on exit
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
