"""Tests for database migration system."""

import sqlite3
from pathlib import Path

import pytest

from pmd.store.migrations import MigrationRunner, Migration
from pmd.store.database import Database


class TestMigrationRunner:
    """Tests for MigrationRunner."""

    def test_fresh_database_starts_at_version_zero(self, tmp_path: Path):
        """Fresh database should have version 0."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)
        assert runner.get_version() == 0

        conn.close()

    def test_run_applies_pending_migrations(self, tmp_path: Path):
        """run() should apply all pending migrations."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)

        # Should have migrations to apply
        assert not runner.is_up_to_date()

        # Apply migrations
        applied = runner.run()

        # Should have applied at least one migration
        assert applied >= 1
        assert runner.is_up_to_date()
        assert runner.get_version() >= 1

        conn.close()

    def test_run_is_idempotent(self, tmp_path: Path):
        """Running migrations twice should be safe (no-op on second run)."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)

        # First run
        first_applied = runner.run()
        first_version = runner.get_version()

        # Second run should be no-op
        second_applied = runner.run()
        second_version = runner.get_version()

        assert second_applied == 0
        assert second_version == first_version

        conn.close()

    def test_version_persists_across_connections(self, tmp_path: Path):
        """Schema version should persist when reopening database."""
        db_path = tmp_path / "test.db"

        # First connection - apply migrations
        conn1 = sqlite3.connect(str(db_path))
        runner1 = MigrationRunner(conn1)
        runner1.run()
        version = runner1.get_version()
        conn1.close()

        # Second connection - should see same version
        conn2 = sqlite3.connect(str(db_path))
        runner2 = MigrationRunner(conn2)
        assert runner2.get_version() == version
        assert runner2.is_up_to_date()
        conn2.close()

    def test_get_migrations_returns_sorted_list(self, tmp_path: Path):
        """get_migrations() should return migrations sorted by version."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)
        migrations = runner.get_migrations()

        # Should have at least the initial migration
        assert len(migrations) >= 1

        # Should be sorted by version
        versions = [m.version for m in migrations]
        assert versions == sorted(versions)

        conn.close()

    def test_get_pending_migrations_filters_by_version(self, tmp_path: Path):
        """get_pending_migrations() should only return unapplied migrations."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)

        # Before running - all migrations are pending
        all_migrations = runner.get_migrations()
        pending_before = runner.get_pending_migrations()
        assert len(pending_before) == len(all_migrations)

        # After running - no migrations are pending
        runner.run()
        pending_after = runner.get_pending_migrations()
        assert len(pending_after) == 0

        conn.close()

    def test_set_version_updates_user_version(self, tmp_path: Path):
        """set_version() should update PRAGMA user_version."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)

        runner.set_version(42)
        assert runner.get_version() == 42

        # Verify directly with pragma
        cursor = conn.execute("PRAGMA user_version")
        assert cursor.fetchone()[0] == 42

        conn.close()

    def test_initial_migration_creates_tables(self, tmp_path: Path):
        """Initial migration should create all required tables."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)
        runner.run()

        # Check that core tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "content",
            "collections",
            "documents",
            "documents_fts",
            "content_vectors",
            "path_contexts",
            "ollama_cache",
            "source_metadata",
            "document_metadata",
            "document_tags",
        }

        for table in expected_tables:
            assert table in tables, f"Missing table: {table}"

        conn.close()


class TestDatabaseMigrationIntegration:
    """Integration tests for Database class with migrations."""

    def test_database_connect_runs_migrations(self, tmp_path: Path):
        """Database.connect() should automatically run migrations."""
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        db.connect()

        # Check version is set
        cursor = db.execute("PRAGMA user_version")
        version = cursor.fetchone()[0]
        assert version >= 1

        db.close()

    def test_database_connect_is_idempotent(self, tmp_path: Path):
        """Reconnecting to existing database should not fail."""
        db_path = tmp_path / "test.db"

        # First connection
        db1 = Database(db_path)
        db1.connect()
        db1.close()

        # Second connection
        db2 = Database(db_path)
        db2.connect()

        # Should still work
        cursor = db2.execute("SELECT COUNT(*) FROM collections")
        count = cursor.fetchone()[0]
        assert count == 0  # Empty but table exists

        db2.close()

    def test_database_preserves_data_across_migrations(self, tmp_path: Path):
        """Data should be preserved when migrations run."""
        db_path = tmp_path / "test.db"

        # Create database and add data
        db = Database(db_path)
        db.connect()

        db.execute(
            """
            INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at)
            VALUES ('test', '/path', '*.md', datetime('now'), datetime('now'))
            """
        )
        db._connection.commit()
        db.close()

        # Reconnect - migrations should not lose data
        db2 = Database(db_path)
        db2.connect()

        cursor = db2.execute("SELECT name FROM collections")
        row = cursor.fetchone()
        assert row["name"] == "test"

        db2.close()


class TestMigrationClass:
    """Tests for Migration dataclass."""

    def test_migration_repr(self):
        """Migration should have useful repr."""
        migration = Migration(
            version=1,
            description="Test migration",
            up=lambda conn: None,
        )

        assert "1" in repr(migration)
        assert "Test migration" in repr(migration)


class TestUpgradeFromPriorVersion:
    """Tests simulating upgrade from older database versions."""

    def test_upgrade_from_version_zero(self, tmp_path: Path):
        """Database at version 0 should upgrade to latest."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Verify starts at 0
        runner = MigrationRunner(conn)
        assert runner.get_version() == 0

        # Run migrations
        applied = runner.run()
        assert applied > 0
        assert runner.get_version() > 0

        conn.close()

    def test_already_migrated_database_no_op(self, tmp_path: Path):
        """Database already at latest version should not re-run migrations."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner = MigrationRunner(conn)

        # First run
        runner.run()
        latest = runner.get_version()

        # Manually create a spy to track calls
        call_count = [0]
        original_migrations = runner.get_migrations()

        for m in original_migrations:
            original_up = m.up

            def wrapped_up(c, _orig=original_up):
                call_count[0] += 1
                return _orig(c)

            m.up = wrapped_up

        # Clear cache to use wrapped versions
        runner._migrations = original_migrations

        # Run again - should be no-op
        applied = runner.run()

        assert applied == 0
        assert call_count[0] == 0  # No migrations actually ran

        conn.close()
