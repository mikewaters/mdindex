"""Helper module for running Alembic migrations programmatically.

This module provides a simple interface for running database migrations
without requiring the alembic CLI or configuration files.

Example:
    from pmd.store.migrate import upgrade, get_current_revision

    # Run all pending migrations
    upgrade("sqlite:///my_database.db")

    # Check current revision
    revision = get_current_revision("sqlite:///my_database.db")
    print(f"Current revision: {revision}")
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text


def _get_alembic_config(db_url: str) -> Config:
    """Create an Alembic Config object for programmatic use.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        Configured Alembic Config object.
    """
    # Get the directory containing this module (pmd/store)
    store_dir = Path(__file__).parent
    alembic_dir = store_dir / "alembic"

    # Create config without an ini file
    config = Config()
    config.set_main_option("script_location", str(alembic_dir))
    config.set_main_option("sqlalchemy.url", db_url)

    return config


def upgrade(db_url: str, revision: str = "head") -> None:
    """Run Alembic upgrade to the specified revision.

    Args:
        db_url: SQLAlchemy database URL (e.g., "sqlite:///path/to/db.sqlite").
        revision: Target revision, defaults to "head" (latest).

    Raises:
        alembic.util.exc.CommandError: If migration fails.
    """
    config = _get_alembic_config(db_url)

    # Create engine and run with connection
    engine = create_engine(db_url)
    with engine.connect() as connection:
        # Enable foreign keys for SQLite
        if engine.dialect.name == "sqlite":
            connection.execute(text("PRAGMA foreign_keys = ON"))

        # Pass connection to Alembic config for reuse
        config.attributes["connection"] = connection
        command.upgrade(config, revision)
        connection.commit()


def downgrade(db_url: str, revision: str) -> None:
    """Run Alembic downgrade to the specified revision.

    Args:
        db_url: SQLAlchemy database URL.
        revision: Target revision (e.g., "-1" for one step back, or specific revision).

    Raises:
        alembic.util.exc.CommandError: If migration fails.
    """
    config = _get_alembic_config(db_url)

    engine = create_engine(db_url)
    with engine.connect() as connection:
        if engine.dialect.name == "sqlite":
            connection.execute(text("PRAGMA foreign_keys = ON"))

        config.attributes["connection"] = connection
        command.downgrade(config, revision)
        connection.commit()


def get_current_revision(db_url: str) -> str | None:
    """Get the current Alembic revision of the database.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        Current revision string, or None if no migrations have been applied.
    """
    engine = create_engine(db_url)
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        return context.get_current_revision()


def get_head_revision(db_url: str) -> str | None:
    """Get the head (latest) revision available.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        Head revision string, or None if no migrations exist.
    """
    from alembic.script import ScriptDirectory

    config = _get_alembic_config(db_url)
    script = ScriptDirectory.from_config(config)
    return script.get_current_head()


def is_up_to_date(db_url: str) -> bool:
    """Check if the database is at the latest migration revision.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        True if current revision matches head revision.
    """
    current = get_current_revision(db_url)
    head = get_head_revision(db_url)
    return current == head
