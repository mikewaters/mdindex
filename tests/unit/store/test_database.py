"""Tests for database connection and management."""

import pytest
import sqlite3
from pathlib import Path

from pmd.store.database import Database
from pmd.core.exceptions import DatabaseError


class TestDatabaseConnection:
    """Tests for database connection lifecycle."""

    def test_connect_creates_database_file(self, test_db_path: Path):
        """Database file should be created on connect."""
        db = Database(test_db_path)
        db.connect()

        assert test_db_path.exists()
        db.close()

    def test_connect_creates_parent_directories(self, tmp_path: Path):
        """Connect should create parent directories if needed."""
        db_path = tmp_path / "subdir" / "nested" / "test.db"
        db = Database(db_path)
        db.connect()

        assert db_path.exists()
        db.close()

    def test_close_without_connect(self, test_db_path: Path):
        """Close should not raise if not connected."""
        db = Database(test_db_path)
        db.close()  # Should not raise

    def test_double_connect(self, test_db_path: Path):
        """Connecting twice should work without error."""
        db = Database(test_db_path)
        db.connect()
        db.connect()  # Should not raise
        db.close()

    def test_close_clears_connection(self, test_db_path: Path):
        """Close should clear the connection."""
        db = Database(test_db_path)
        db.connect()
        db.close()

        with pytest.raises(DatabaseError, match="not connected"):
            db.execute("SELECT 1")


class TestDatabaseSchema:
    """Tests for schema initialization."""

    def test_schema_creates_collections_table(self, db: Database):
        """Schema should create collections table."""
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='collections'"
        )
        assert cursor.fetchone() is not None

    def test_schema_creates_documents_table(self, db: Database):
        """Schema should create documents table."""
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        assert cursor.fetchone() is not None

    def test_schema_creates_content_table(self, db: Database):
        """Schema should create content table."""
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='content'"
        )
        assert cursor.fetchone() is not None

    def test_schema_creates_fts_table(self, db: Database):
        """Schema should create FTS5 virtual table."""
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        assert cursor.fetchone() is not None

    def test_schema_creates_content_vectors_table(self, db: Database):
        """Schema should create content_vectors metadata table."""
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='content_vectors'"
        )
        assert cursor.fetchone() is not None

    def test_schema_creates_indexes(self, db: Database):
        """Schema should create performance indexes."""
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in cursor.fetchall()}

        assert "idx_documents_collection" in indexes
        assert "idx_documents_hash" in indexes


class TestDatabaseTransactions:
    """Tests for transaction management."""

    def test_transaction_commits_on_success(self, db: Database):
        """Successful transaction should commit."""
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test", "/tmp", "**/*.md", "2024-01-01", "2024-01-01"),
            )

        cursor = db.execute("SELECT name FROM collections WHERE name = 'test'")
        assert cursor.fetchone() is not None

    def test_transaction_rollbacks_on_error(self, db: Database):
        """Failed transaction should rollback."""
        try:
            with db.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("test-rollback", "/tmp", "**/*.md", "2024-01-01", "2024-01-01"),
                )
                raise ValueError("Simulated error")
        except DatabaseError:
            pass

        cursor = db.execute("SELECT name FROM collections WHERE name = 'test-rollback'")
        assert cursor.fetchone() is None

    def test_transaction_without_connection_raises(self, test_db_path: Path):
        """Transaction without connection should raise."""
        db = Database(test_db_path)

        with pytest.raises(DatabaseError, match="not connected"):
            with db.transaction():
                pass


class TestDatabaseExecute:
    """Tests for SQL execution."""

    def test_execute_simple_query(self, db: Database):
        """Execute should run simple queries."""
        cursor = db.execute("SELECT 1 as num")
        result = cursor.fetchone()
        assert result["num"] == 1

    def test_execute_with_parameters(self, db: Database):
        """Execute should handle parameters."""
        cursor = db.execute("SELECT ? as val", ("test",))
        result = cursor.fetchone()
        assert result["val"] == "test"

    def test_execute_without_connection_raises(self, test_db_path: Path):
        """Execute without connection should raise."""
        db = Database(test_db_path)

        with pytest.raises(DatabaseError, match="not connected"):
            db.execute("SELECT 1")

    def test_execute_invalid_sql_raises(self, db: Database):
        """Execute with invalid SQL should raise DatabaseError."""
        with pytest.raises(DatabaseError):
            db.execute("INVALID SQL SYNTAX")


class TestDatabaseExecutescript:
    """Tests for script execution."""

    def test_executescript_multiple_statements(self, db: Database):
        """Executescript should handle multiple statements."""
        db.executescript("""
            INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at)
            VALUES ('script1', '/tmp', '**/*.md', '2024-01-01', '2024-01-01');
            INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at)
            VALUES ('script2', '/tmp', '**/*.md', '2024-01-01', '2024-01-01');
        """)

        cursor = db.execute("SELECT COUNT(*) as count FROM collections")
        assert cursor.fetchone()["count"] == 2

    def test_executescript_without_connection_raises(self, test_db_path: Path):
        """Executescript without connection should raise."""
        db = Database(test_db_path)

        with pytest.raises(DatabaseError, match="not connected"):
            db.executescript("SELECT 1;")


class TestDatabaseVecExtension:
    """Tests for sqlite-vec extension support."""

    def test_vec_available_property_exists(self, db: Database):
        """vec_available property should exist."""
        assert hasattr(db, 'vec_available')
        assert isinstance(db.vec_available, bool)

    def test_vec_table_created_when_available(self, db: Database):
        """Vector table should be created if extension is available."""
        if db.vec_available:
            cursor = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='content_vectors_vec'"
            )
            assert cursor.fetchone() is not None
