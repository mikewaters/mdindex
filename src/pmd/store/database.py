"""SQLAlchemy database connection managers for PMD.

This module provides both synchronous and asynchronous database access via SQLAlchemy.
The Database class (sync) and AsyncDatabase class (async) manage engine lifecycle,
session factories, and schema initialization including Alembic migrations.

Note: FTS5 and sqlite-vec virtual tables require raw SQL and are managed via
the connection() context managers rather than ORM sessions.
"""

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

from loguru import logger
from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from ..core.exceptions import DatabaseError
from .migrate import upgrade as alembic_upgrade
from .schema import get_vector_schema


def _setup_sqlite_connection(dbapi_connection, connection_record) -> None:
    """Configure SQLite connection with required pragmas.

    This is called by SQLAlchemy's connection pool event system.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


def _load_vec_extension_sync(dbapi_connection) -> bool:
    """Attempt to load sqlite-vec extension on a raw dbapi connection.

    Args:
        dbapi_connection: Raw sqlite3 connection from SQLAlchemy.

    Returns:
        True if sqlite-vec was loaded successfully, False otherwise.
    """
    try:
        import sqlite_vec

        dbapi_connection.enable_load_extension(True)
        sqlite_vec.load(dbapi_connection)
        dbapi_connection.enable_load_extension(False)
        return True
    except ImportError:
        logger.debug("sqlite-vec not available: package not installed")
        return False
    except Exception as e:
        logger.debug(f"sqlite-vec loading failed: {e}")
        return False


class LegacyRow:
    """Row wrapper that provides sqlite3.Row-compatible interface.

    Wraps SQLAlchemy Row objects to provide .keys() method and
    other sqlite3.Row-like behaviors.
    """

    def __init__(self, row):
        """Initialize with a SQLAlchemy Row.

        Args:
            row: SQLAlchemy Row object.
        """
        self._row = row
        self._mapping = row._mapping

    def __getitem__(self, key):
        """Get value by column name or index."""
        if isinstance(key, int):
            return self._row[key]
        return self._mapping[key]

    def keys(self):
        """Get column names (sqlite3.Row compatibility)."""
        return self._mapping.keys()

    def __contains__(self, key):
        """Check if column exists."""
        return key in self._mapping

    def __iter__(self):
        """Iterate over values."""
        return iter(self._row)

    def __len__(self):
        """Get number of columns."""
        return len(self._row)


class LegacyResult:
    """Result wrapper that returns LegacyRow objects.

    Wraps SQLAlchemy CursorResult to return sqlite3.Row-compatible rows.
    Can either wrap a live result or buffer rows for disconnected access.
    """

    def __init__(self, result=None, rows=None, lastrowid=None, rowcount=0):
        """Initialize with a SQLAlchemy result or buffered rows.

        Args:
            result: SQLAlchemy CursorResult object (live mode).
            rows: Pre-fetched list of LegacyRow objects (buffered mode).
            lastrowid: Last inserted row ID.
            rowcount: Number of affected rows.
        """
        self._result = result
        self._rows = rows if rows is not None else []
        self._index = 0
        self._lastrowid = lastrowid
        self._rowcount = rowcount

    @classmethod
    def from_buffered(cls, rows, lastrowid=None, rowcount=0):
        """Create a buffered result from pre-fetched rows.

        Args:
            rows: List of LegacyRow objects.
            lastrowid: Last inserted row ID.
            rowcount: Number of affected rows.

        Returns:
            LegacyResult instance.
        """
        return cls(result=None, rows=rows, lastrowid=lastrowid, rowcount=rowcount)

    def fetchone(self):
        """Fetch the next row.

        Returns:
            A LegacyRow or None.
        """
        if self._result is not None:
            row = self._result.fetchone()
            return LegacyRow(row) if row is not None else None
        else:
            if self._index < len(self._rows):
                row = self._rows[self._index]
                self._index += 1
                return row
            return None

    def fetchall(self):
        """Fetch all remaining rows.

        Returns:
            List of LegacyRow objects.
        """
        if self._result is not None:
            return [LegacyRow(row) for row in self._result.fetchall()]
        else:
            remaining = self._rows[self._index:]
            self._index = len(self._rows)
            return remaining

    def fetchmany(self, size=None):
        """Fetch the next set of rows.

        Args:
            size: Number of rows to fetch.

        Returns:
            List of LegacyRow objects.
        """
        if self._result is not None:
            return [LegacyRow(row) for row in self._result.fetchmany(size)]
        else:
            if size is None:
                size = 1
            end = min(self._index + size, len(self._rows))
            result = self._rows[self._index:end]
            self._index = end
            return result

    @property
    def lastrowid(self):
        """Get the row ID of the last INSERT."""
        if self._result is not None:
            return self._result.lastrowid
        return self._lastrowid

    @property
    def rowcount(self):
        """Get the number of rows affected."""
        if self._result is not None:
            return self._result.rowcount
        return self._rowcount

    def __iter__(self):
        """Iterate over rows."""
        if self._result is not None:
            for row in self._result:
                yield LegacyRow(row)
        else:
            for row in self._rows[self._index:]:
                yield row
            self._index = len(self._rows)


class LegacyCursor:
    """Cursor-like wrapper for SQLAlchemy Connection (legacy compatibility).

    Provides an interface compatible with sqlite3.Cursor for code that
    uses the old transaction() context manager pattern.
    """

    def __init__(self, conn: Connection):
        """Initialize with a SQLAlchemy Connection.

        Args:
            conn: SQLAlchemy Connection object.
        """
        self._conn = conn
        self._last_result = None

    def execute(self, sql: str, params: tuple = ()):
        """Execute a SQL statement with positional parameters.

        Args:
            sql: SQL statement with ? placeholders.
            params: Tuple of parameter values.

        Returns:
            Self for method chaining (like sqlite3.Cursor).
        """
        # Convert positional ? placeholders to named :p0, :p1, etc.
        converted_sql = sql
        param_dict = {}
        for i, param in enumerate(params):
            placeholder = f":p{i}"
            converted_sql = converted_sql.replace("?", placeholder, 1)
            param_dict[f"p{i}"] = param

        self._last_result = self._conn.execute(text(converted_sql), param_dict)
        return self

    def executemany(self, sql: str, params_list: list[tuple]):
        """Execute a SQL statement multiple times with different parameters.

        Args:
            sql: SQL statement with ? placeholders.
            params_list: List of parameter tuples.

        Returns:
            Self for method chaining.
        """
        for params in params_list:
            self.execute(sql, params)
        return self

    def fetchone(self):
        """Fetch the next row of a query result set.

        Returns:
            A LegacyRow wrapping the Row, or None if no more rows.
        """
        if self._last_result is None:
            return None
        row = self._last_result.fetchone()
        return LegacyRow(row) if row is not None else None

    def fetchall(self):
        """Fetch all remaining rows of a query result set.

        Returns:
            List of LegacyRow objects.
        """
        if self._last_result is None:
            return []
        return [LegacyRow(row) for row in self._last_result.fetchall()]

    @property
    def lastrowid(self) -> int | None:
        """Get the row ID of the last INSERT.

        Returns:
            The row ID or None.
        """
        if self._last_result is None:
            return None
        return self._last_result.lastrowid

    @property
    def rowcount(self) -> int:
        """Get the number of rows affected by the last statement.

        Returns:
            Number of affected rows.
        """
        if self._last_result is None:
            return 0
        return self._last_result.rowcount


class Database:
    """Synchronous SQLAlchemy database manager for PMD.

    Manages a SQLAlchemy Engine and Session factory for synchronous database access.
    On connect, runs Alembic migrations and optionally loads sqlite-vec extension.

    Example:
        db = Database(Path("index.db"))
        db.connect()

        with db.session() as session:
            collections = session.query(SourceCollectionModel).all()

        db.close()
    """

    def __init__(self, path: Path, embedding_dimension: int | None = None):
        """Initialize database with path.

        Args:
            path: Path to the SQLite database file.
            embedding_dimension: Vector embedding dimension for schema.
                If None, uses the default from schema.py.
        """
        self.path = path
        self.embedding_dimension = embedding_dimension
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._vec_available: bool = False

    @property
    def db_url(self) -> str:
        """Get SQLAlchemy database URL for this database."""
        return f"sqlite:///{self.path.resolve()}"

    def connect(self) -> None:
        """Initialize database connection, run migrations, and load extensions."""
        logger.debug(f"Connecting to database: {self.path}")
        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Create engine
            self._engine = create_engine(
                self.db_url,
                echo=False,
                pool_pre_ping=True,
            )

            # Set up connection event for pragmas
            event.listen(self._engine, "connect", _setup_sqlite_connection)

            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

            # Run Alembic migrations
            logger.debug("Running Alembic migrations")
            alembic_upgrade(self.db_url)
            logger.debug("Alembic migrations complete")

            # Load sqlite-vec extension via a connection
            with self._engine.connect() as conn:
                raw_conn = conn.connection.dbapi_connection
                self._vec_available = _load_vec_extension_sync(raw_conn)

                # Create vector table if extension available
                if self._vec_available:
                    if self.embedding_dimension:
                        vec_schema = get_vector_schema(dimension=self.embedding_dimension)
                    else:
                        vec_schema = get_vector_schema()
                    conn.execute(text(vec_schema))
                    conn.commit()
                    logger.debug("Vector schema initialized")

            logger.info(
                f"Database connected: {self.path} (vec_available={self._vec_available})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}") from e

    def close(self) -> None:
        """Close database connection and dispose of engine."""
        if self._engine:
            logger.debug(f"Closing database connection: {self.path}")
            try:
                self._engine.dispose()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.error(f"Failed to close database: {e}")
                raise DatabaseError(f"Failed to close database: {e}") from e
            finally:
                self._engine = None
                self._session_factory = None

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Context manager for ORM session access.

        Yields:
            SQLAlchemy Session for ORM operations.

        Raises:
            DatabaseError: If database not connected.

        Example:
            with db.session() as session:
                collection = SourceCollectionModel(name="docs", ...)
                session.add(collection)
        """
        if not self._session_factory:
            raise DatabaseError("Database not connected")

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Session failed: {e}") from e
        finally:
            session.close()

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Context manager for raw connection access (Core SQL).

        Use this for operations that require raw SQL, such as FTS5 queries
        or sqlite-vec operations.

        Yields:
            SQLAlchemy Connection for Core SQL operations.

        Raises:
            DatabaseError: If database not connected.

        Example:
            with db.connection() as conn:
                result = conn.execute(text("SELECT * FROM documents_fts WHERE ..."))
        """
        if not self._engine:
            raise DatabaseError("Database not connected")

        with self._engine.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise DatabaseError(f"Connection failed: {e}") from e

    @contextmanager
    def transaction(self) -> Iterator["LegacyCursor"]:
        """Context manager for database transactions (legacy compatibility).

        Yields a cursor-like object for executing SQL statements with automatic
        commit on success or rollback on error.

        Yields:
            LegacyCursor wrapping a SQLAlchemy connection.

        Raises:
            DatabaseError: If connection is not available or transaction fails.
        """
        if not self._engine:
            raise DatabaseError("Database not connected")

        with self._engine.connect() as conn:
            cursor = LegacyCursor(conn)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise DatabaseError(f"Transaction failed: {e}") from e

    @property
    def vec_available(self) -> bool:
        """Check if sqlite-vec extension is available."""
        return self._vec_available

    @property
    def engine(self) -> Engine:
        """Get the underlying SQLAlchemy engine.

        Raises:
            DatabaseError: If database not connected.
        """
        if not self._engine:
            raise DatabaseError("Database not connected")
        return self._engine

    # Legacy compatibility methods (delegate to connection context)

    def execute(self, sql: str, params: tuple = ()) -> LegacyResult:
        """Execute a SQL query (legacy compatibility).

        Prefer using session() or connection() context managers instead.

        Args:
            sql: SQL statement to execute.
            params: Parameters for the SQL statement (positional ? placeholders).

        Returns:
            LegacyResult with buffered rows.

        Raises:
            DatabaseError: If connection is not available or query fails.
        """
        if not self._engine:
            raise DatabaseError("Database not connected")

        try:
            with self._engine.connect() as conn:
                # Convert positional ? placeholders to named :p0, :p1, etc.
                # for SQLAlchemy text() compatibility
                converted_sql = sql
                param_dict = {}
                for i, param in enumerate(params):
                    placeholder = f":p{i}"
                    converted_sql = converted_sql.replace("?", placeholder, 1)
                    param_dict[f"p{i}"] = param

                result = conn.execute(text(converted_sql), param_dict)
                # Buffer results before connection closes
                # Only fetch rows for SELECT-like statements that return data
                lastrowid = result.lastrowid
                rowcount = result.rowcount
                if result.returns_rows:
                    rows = [LegacyRow(row) for row in result.fetchall()]
                else:
                    rows = []
                conn.commit()
                return LegacyResult.from_buffered(rows, lastrowid, rowcount)
        except Exception as e:
            raise DatabaseError(f"Query failed: {e}") from e

    def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements (legacy compatibility).

        Prefer using connection() context manager instead.

        Args:
            sql: SQL script with multiple statements.

        Raises:
            DatabaseError: If connection is not available or script fails.
        """
        if not self._engine:
            raise DatabaseError("Database not connected")

        with self._engine.connect() as conn:
            # Split and execute statements individually
            for statement in sql.strip().split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            conn.commit()


class AsyncDatabase:
    """Asynchronous SQLAlchemy database manager for PMD.

    Manages an async SQLAlchemy Engine and Session factory for non-blocking
    database access. Note that sqlite-vec extension loading may have limitations
    in async mode.

    Example:
        db = AsyncDatabase(Path("index.db"))
        await db.connect()

        async with db.session() as session:
            result = await session.execute(select(SourceCollectionModel))
            collections = result.scalars().all()

        await db.close()
    """

    def __init__(self, path: Path, embedding_dimension: int | None = None):
        """Initialize async database with path.

        Args:
            path: Path to the SQLite database file.
            embedding_dimension: Vector embedding dimension for schema.
                If None, uses the default from schema.py.
        """
        self.path = path
        self.embedding_dimension = embedding_dimension
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._vec_available: bool = False

    @property
    def db_url(self) -> str:
        """Get async SQLAlchemy database URL for this database."""
        return f"sqlite+aiosqlite:///{self.path.resolve()}"

    async def connect(self) -> None:
        """Initialize async database connection and run migrations.

        Note: Alembic migrations are run synchronously before async engine setup,
        and sqlite-vec loading requires special handling in async mode.
        """
        logger.debug(f"Connecting to async database: {self.path}")
        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Run Alembic migrations synchronously first
            # (Alembic doesn't have full async support)
            sync_url = f"sqlite:///{self.path.resolve()}"
            logger.debug("Running Alembic migrations (sync)")
            alembic_upgrade(sync_url)
            logger.debug("Alembic migrations complete")

            # Create async engine
            self._engine = create_async_engine(
                self.db_url,
                echo=False,
            )

            # Create async session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine, expire_on_commit=False
            )

            # Set up pragmas and try to load sqlite-vec
            async with self._engine.connect() as conn:
                await conn.execute(text("PRAGMA foreign_keys = ON"))

                # Try to load sqlite-vec extension
                raw_conn = await conn.get_raw_connection()
                dbapi_conn = raw_conn.dbapi_connection
                self._vec_available = _load_vec_extension_sync(dbapi_conn)

                # Create vector table if extension available
                if self._vec_available:
                    if self.embedding_dimension:
                        vec_schema = get_vector_schema(dimension=self.embedding_dimension)
                    else:
                        vec_schema = get_vector_schema()
                    await conn.execute(text(vec_schema))
                    await conn.commit()
                    logger.debug("Vector schema initialized (async)")

            logger.info(
                f"Async database connected: {self.path} (vec_available={self._vec_available})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to async database: {e}")
            raise DatabaseError(f"Failed to connect to async database: {e}") from e

    async def close(self) -> None:
        """Close async database connection and dispose of engine."""
        if self._engine:
            logger.debug(f"Closing async database connection: {self.path}")
            try:
                await self._engine.dispose()
                logger.debug("Async database connection closed")
            except Exception as e:
                logger.error(f"Failed to close async database: {e}")
                raise DatabaseError(f"Failed to close async database: {e}") from e
            finally:
                self._engine = None
                self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Async context manager for ORM session access.

        Yields:
            SQLAlchemy AsyncSession for ORM operations.

        Raises:
            DatabaseError: If database not connected.

        Example:
            async with db.session() as session:
                result = await session.execute(select(SourceCollectionModel))
                collections = result.scalars().all()
        """
        if not self._session_factory:
            raise DatabaseError("Database not connected")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise DatabaseError(f"Async session failed: {e}") from e

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        """Async context manager for raw connection access (Core SQL).

        Use this for operations that require raw SQL, such as FTS5 queries
        or sqlite-vec operations.

        Yields:
            SQLAlchemy AsyncConnection for Core SQL operations.

        Raises:
            DatabaseError: If database not connected.

        Example:
            async with db.connection() as conn:
                result = await conn.execute(text("SELECT * FROM documents_fts WHERE ..."))
        """
        if not self._engine:
            raise DatabaseError("Database not connected")

        async with self._engine.connect() as conn:
            try:
                yield conn
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                raise DatabaseError(f"Async connection failed: {e}") from e

    @property
    def vec_available(self) -> bool:
        """Check if sqlite-vec extension is available."""
        return self._vec_available

    @property
    def engine(self) -> AsyncEngine:
        """Get the underlying SQLAlchemy async engine.

        Raises:
            DatabaseError: If database not connected.
        """
        if not self._engine:
            raise DatabaseError("Database not connected")
        return self._engine
