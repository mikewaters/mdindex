"""SQLite database connection manager for PMD."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..core.exceptions import DatabaseError
from .schema import get_schema


class Database:
    """SQLite database connection manager."""

    def __init__(self, path: Path):
        """Initialize database with path.

        Args:
            path: Path to the SQLite database file.
        """
        self.path = path
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Initialize database connection and load extensions."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.path))
            self._connection.row_factory = sqlite3.Row
            self._enable_fts5()
            self._load_vec_extension()
            self._init_schema()
        except Exception as e:
            raise DatabaseError(f"Failed to connect to database: {e}") from e

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                raise DatabaseError(f"Failed to close database: {e}") from e
            finally:
                self._connection = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions.

        Yields:
            A cursor for executing SQL statements.

        Raises:
            DatabaseError: If connection is not available or transaction fails.
        """
        if not self._connection:
            raise DatabaseError("Database not connected")

        cursor = self._connection.cursor()
        try:
            yield cursor
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            raise DatabaseError(f"Transaction failed: {e}") from e
        finally:
            cursor.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL query.

        Args:
            sql: SQL statement to execute.
            params: Parameters for the SQL statement.

        Returns:
            A cursor with the query results.

        Raises:
            DatabaseError: If connection is not available or query fails.
        """
        if not self._connection:
            raise DatabaseError("Database not connected")

        try:
            return self._connection.execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}") from e

    def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements.

        Args:
            sql: SQL script with multiple statements.

        Raises:
            DatabaseError: If connection is not available or script fails.
        """
        if not self._connection:
            raise DatabaseError("Database not connected")

        try:
            self._connection.executescript(sql)
        except Exception as e:
            raise DatabaseError(f"Script execution failed: {e}") from e

    def _enable_fts5(self) -> None:
        """Enable FTS5 extension."""
        try:
            if self._connection:
                self._connection.enable_load_extension(True)
                self._connection.execute("PRAGMA foreign_keys = ON")
        except Exception:
            # FTS5 is built-in on most Python sqlite3 installations
            pass

    def _load_vec_extension(self) -> None:
        """Load sqlite-vec extension.

        Note: sqlite-vec is optional for this phase. If unavailable,
        vector search will be disabled.
        """
        # TODO: Implement proper sqlite-vec loading
        # For now, we skip this as vector search is Phase 2
        pass

    def _init_schema(self) -> None:
        """Initialize database schema."""
        try:
            schema = get_schema()
            self.executescript(schema)
        except Exception as e:
            raise DatabaseError(f"Failed to initialize schema: {e}") from e
