"""Helper module for running database migrations programmatically.

This module provides a simple interface for running database migrations
using the custom migration runner.

Example:
    from pmd.store.migrate import upgrade, get_current_revision

    # Run all pending migrations
    upgrade("sqlite:///my_database.db")

    # Check current revision
    revision = get_current_revision("sqlite:///my_database.db")
    print(f"Current revision: {revision}")
"""

import sqlite3
from pathlib import Path

from .migrations.runner import MigrationRunner


def _get_db_path(db_url: str) -> str:
    """Extract file path from SQLite database URL.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        File path to the database.
    """
    # Handle both sqlite:/// and sqlite+aiosqlite:/// URLs
    if db_url.startswith("sqlite+aiosqlite:///"):
        return db_url.replace("sqlite+aiosqlite:///", "")
    elif db_url.startswith("sqlite:///"):
        return db_url.replace("sqlite:///", "")
    else:
        raise ValueError(f"Unsupported database URL: {db_url}")


def upgrade(db_url: str, revision: str = "head") -> None:
    """Run migrations to the specified revision.

    Args:
        db_url: SQLAlchemy database URL (e.g., "sqlite:///path/to/db.sqlite").
        revision: Target revision, defaults to "head" (latest).
            Note: revision parameter is ignored, always upgrades to latest.

    Raises:
        Exception: If migration fails.
    """
    db_path = _get_db_path(db_url)

    # Ensure parent directory exists
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        runner = MigrationRunner(conn)
        runner.run()
    finally:
        conn.close()


def downgrade(db_url: str, revision: str) -> None:
    """Downgrade is not supported in the custom migration system.

    Args:
        db_url: SQLAlchemy database URL.
        revision: Target revision.

    Raises:
        NotImplementedError: Always raised, downgrade not supported.
    """
    raise NotImplementedError("Downgrade not supported in custom migration system")


def get_current_revision(db_url: str) -> int | None:
    """Get the current migration version of the database.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        Current version number, or None if database doesn't exist.
    """
    db_path = _get_db_path(db_url)

    if not Path(db_path).exists():
        return None

    conn = sqlite3.connect(db_path)
    try:
        runner = MigrationRunner(conn)
        return runner.get_version()
    finally:
        conn.close()


def get_head_revision(db_url: str) -> int:
    """Get the head (latest) revision available.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        Head revision number.
    """
    db_path = _get_db_path(db_url)

    conn = sqlite3.connect(db_path)
    try:
        runner = MigrationRunner(conn)
        return runner.get_latest_version()
    finally:
        conn.close()


def is_up_to_date(db_url: str) -> bool:
    """Check if the database is at the latest migration revision.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        True if current revision matches head revision.
    """
    db_path = _get_db_path(db_url)

    if not Path(db_path).exists():
        return False

    conn = sqlite3.connect(db_path)
    try:
        runner = MigrationRunner(conn)
        return runner.is_up_to_date()
    finally:
        conn.close()
