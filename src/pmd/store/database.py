"""SQLite database connection manager for PMD."""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from loguru import logger

from ..core.exceptions import DatabaseError
from .migrations import MigrationRunner
from .schema import get_vector_schema


class Database:
    """SQLite database connection manager."""

    def __init__(self, path: Path, embedding_dimension: int | None = None):
        """Initialize database with path.

        Args:
            path: Path to the SQLite database file.
            embedding_dimension: Vector embedding dimension for schema.
                If None, uses the default from schema.py.
        """
        self.path = path
        self.embedding_dimension = embedding_dimension
        self._connection: sqlite3.Connection | None = None
        self._vec_available: bool = False

    def connect(self) -> None:
        """Initialize database connection and load extensions."""
        logger.debug(f"Connecting to database: {self.path}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.path))
            self._connection.row_factory = sqlite3.Row
            self._enable_fts5()
            self._load_vec_extension()
            self._init_schema()
            logger.info(f"Database connected: {self.path} (vec_available={self._vec_available})")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}") from e

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            logger.debug(f"Closing database connection: {self.path}")
            try:
                self._connection.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.error(f"Failed to close database: {e}")
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
        start_time = time.perf_counter()
        #logger.debug("Transaction started")
        try:
            yield cursor
            self._connection.commit()
            #elapsed = (time.perf_counter() - start_time) * 1000
            #logger.debug(f"Transaction committed ({elapsed:.2f}ms)")
        except Exception as e:
            self._connection.rollback()
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Transaction rolled back after {elapsed:.2f}ms: {e}")
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

        sqlite-vec is optional. If unavailable, vector search will be disabled.
        """
        if not self._connection:
            return

        logger.debug("Attempting to load sqlite-vec extension")
        try:
            # Try to load sqlite-vec via the Python package
            import sqlite_vec

            self._connection.enable_load_extension(True)
            sqlite_vec.load(self._connection)
            self._connection.enable_load_extension(False)
            self._vec_available = True
            logger.debug("sqlite-vec extension loaded successfully")
        except ImportError:
            # sqlite-vec Python package not installed
            self._vec_available = False
            logger.debug("sqlite-vec not available: package not installed")
        except Exception as e:
            # Extension loading failed
            self._vec_available = False
            logger.debug(f"sqlite-vec loading failed: {e}")

    @property
    def vec_available(self) -> bool:
        """Check if sqlite-vec extension is available."""
        return self._vec_available

    def _init_schema(self) -> None:
        """Initialize database schema using migrations."""
        if not self._connection:
            raise DatabaseError("Database not connected")

        logger.debug("Running database migrations")
        try:
            # Run versioned migrations
            runner = MigrationRunner(self._connection)
            applied = runner.run()

            if applied > 0:
                logger.debug(f"Applied {applied} migration(s)")
            else:
                logger.debug(
                    f"Database at version {runner.get_version()}, no migrations needed"
                )

            # Create vector table if sqlite-vec is available
            # This is separate from migrations since it requires the extension
            if self._vec_available:
                if self.embedding_dimension:
                    vec_schema = get_vector_schema(dimension=self.embedding_dimension)
                    logger.debug(f"Using embedding dimension: {self.embedding_dimension}")
                else:
                    vec_schema = get_vector_schema()
                self.executescript(vec_schema)
                logger.debug("Vector schema initialized")

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise DatabaseError(f"Failed to initialize schema: {e}") from e
