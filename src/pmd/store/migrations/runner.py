"""Database migration runner for PMD.

Uses SQLite PRAGMA user_version for tracking schema version.
Migrations are applied in order and are idempotent.
"""

from __future__ import annotations

import importlib
import pkgutil
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from loguru import logger

if TYPE_CHECKING:
    from types import ModuleType


@dataclass
class Migration:
    """A database migration."""

    version: int
    description: str
    up: Callable[[sqlite3.Connection], None]

    def __repr__(self) -> str:
        return f"Migration({self.version}, {self.description!r})"


class MigrationRunner:
    """Applies versioned migrations to a SQLite database.

    Uses PRAGMA user_version to track the current schema version.
    Migrations are discovered from the versions subpackage and applied
    in order.

    Example:
        runner = MigrationRunner(connection)
        applied = runner.run()
        print(f"Applied {applied} migrations, now at version {runner.get_version()}")
    """

    def __init__(self, connection: sqlite3.Connection):
        """Initialize with database connection.

        Args:
            connection: SQLite connection to migrate.
        """
        self.conn = connection
        self._migrations: list[Migration] | None = None

    def get_version(self) -> int:
        """Get current schema version from user_version pragma."""
        cursor = self.conn.execute("PRAGMA user_version")
        return cursor.fetchone()[0]

    def set_version(self, version: int) -> None:
        """Set schema version.

        Args:
            version: New version number to set.
        """
        # PRAGMA doesn't support parameters, must use string formatting
        # This is safe since version is an int
        self.conn.execute(f"PRAGMA user_version = {version}")

    def get_migrations(self) -> list[Migration]:
        """Get all available migrations, sorted by version.

        Returns:
            List of Migration objects sorted by version number.
        """
        if self._migrations is not None:
            return self._migrations

        self._migrations = []

        # Import the versions package
        from pmd.store.migrations import versions

        # Discover all migration modules
        for importer, modname, ispkg in pkgutil.iter_modules(versions.__path__):
            if ispkg:
                continue

            # Import the module
            module = importlib.import_module(f"pmd.store.migrations.versions.{modname}")

            # Validate required attributes
            if not hasattr(module, "VERSION") or not hasattr(module, "up"):
                logger.warning(f"Skipping invalid migration module: {modname}")
                continue

            migration = Migration(
                version=module.VERSION,
                description=getattr(module, "DESCRIPTION", modname),
                up=module.up,
            )
            self._migrations.append(migration)

        # Sort by version
        self._migrations.sort(key=lambda m: m.version)

        return self._migrations

    def get_pending_migrations(self) -> list[Migration]:
        """Get migrations that haven't been applied yet.

        Returns:
            List of Migration objects with version > current version.
        """
        current = self.get_version()
        return [m for m in self.get_migrations() if m.version > current]

    def run(self) -> int:
        """Apply all pending migrations.

        Each migration is run in its own transaction. On success,
        the version is updated. On failure, the transaction is
        rolled back and the error is raised.

        Returns:
            Number of migrations applied.

        Raises:
            Exception: If any migration fails.
        """
        pending = self.get_pending_migrations()

        if not pending:
            current = self.get_version()
            logger.debug(f"Database at version {current}, no migrations to apply")
            return 0

        applied = 0
        current = self.get_version()

        for migration in pending:
            logger.info(
                f"Applying migration {migration.version}: {migration.description}"
            )

            try:
                # Run migration
                migration.up(self.conn)

                # Update version (this commits implicitly for PRAGMA)
                self.set_version(migration.version)
                self.conn.commit()

                applied += 1
                logger.debug(f"Migration {migration.version} applied successfully")

            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                self.conn.rollback()
                raise

        logger.info(
            f"Applied {applied} migration(s), "
            f"database now at version {self.get_version()}"
        )
        return applied

    def get_latest_version(self) -> int:
        """Get the latest available migration version.

        Returns:
            Highest version number among migrations, or 0 if none.
        """
        migrations = self.get_migrations()
        if not migrations:
            return 0
        return migrations[-1].version

    def is_up_to_date(self) -> bool:
        """Check if database is at latest version.

        Returns:
            True if current version equals latest migration version.
        """
        return self.get_version() >= self.get_latest_version()
