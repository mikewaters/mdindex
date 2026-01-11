"""Alembic environment configuration for PMD store.

This module configures Alembic for programmatic use with SQLite databases.
It supports both offline (SQL script generation) and online (direct execution)
migrations, with SQLite-specific settings like batch mode for DDL operations.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

# Alembic Config object for access to values in the .ini file
config = context.config

# Interpret the config file for Python logging if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Try to import ORM models for autogenerate support
# If models module doesn't exist yet, use None for target_metadata
try:
    from pmd.store.models import Base

    target_metadata = Base.metadata
except ImportError:
    target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This generates SQL scripts without connecting to the database.
    Useful for reviewing migrations before applying them.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # SQLite-specific: use batch mode for ALTER TABLE operations
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    This connects to the database and executes migrations directly.
    Uses batch mode for SQLite DDL operations.
    """
    # Check if a connectable was passed in (programmatic use)
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        # Create engine from config (alembic.ini use)
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    # Handle both Engine and Connection objects
    if hasattr(connectable, "connect"):
        # It's an Engine, get a connection
        with connectable.connect() as connection:
            # Enable foreign keys for SQLite
            if connection.dialect.name == "sqlite":
                connection.execute(text("PRAGMA foreign_keys = ON"))

            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                # SQLite-specific: use batch mode for ALTER TABLE operations
                render_as_batch=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    else:
        # It's already a Connection (programmatic use)
        connection = connectable

        # Enable foreign keys for SQLite
        if connection.dialect.name == "sqlite":
            connection.execute(text("PRAGMA foreign_keys = ON"))

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite-specific: use batch mode for ALTER TABLE operations
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
