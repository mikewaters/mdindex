"""Database migrations for PMD.

This module provides versioned schema migrations using SQLite's
PRAGMA user_version for tracking.

Example:
    from pmd.store.migrations import MigrationRunner

    runner = MigrationRunner(connection)
    applied = runner.run()
"""

from .runner import Migration, MigrationRunner

__all__ = [
    "Migration",
    "MigrationRunner",
]
