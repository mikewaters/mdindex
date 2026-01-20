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
            "source_collections",
            "documents",
            "documents_fts",
            "content_vectors",
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

        # Check that migrations ran by verifying user_version pragma
        cursor = db.execute("PRAGMA user_version")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] >= 1  # At least one migration applied

        # Also verify core tables exist
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='source_collections'"
        )
        row = cursor.fetchone()
        assert row is not None

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
        cursor = db2.execute("SELECT COUNT(*) FROM source_collections")
        count = cursor.fetchone()[0]
        assert count == 0  # Empty but table exists

        db2.close()

    def test_database_preserves_data_across_migrations(self, tmp_path: Path):
        """Data should be preserved when reconnecting."""
        db_path = tmp_path / "test.db"

        # Create database and add data
        db = Database(db_path)
        db.connect()

        with db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO source_collections (name, pwd, glob_pattern, created_at, updated_at)
                VALUES (?, ?, ?, datetime('now'), datetime('now'))
                """,
                ("test", "/path", "*.md"),
            )
        db.close()

        # Reconnect - data should be preserved
        db2 = Database(db_path)
        db2.connect()

        cursor = db2.execute("SELECT name FROM source_collections")
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


class TestResourceBackfillMigration:
    """Tests for v0003 resource backfill migration."""

    def test_backfill_creates_resources_for_filesystem_documents(self, tmp_path: Path):
        """Backfill should create Resource rows for filesystem documents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        runner = MigrationRunner(conn)

        # Apply only v0001 and v0002 to set up schema without resources
        for migration in runner.get_migrations():
            if migration.version <= 2:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # Insert test data: a filesystem collection with documents
        conn.execute(
            """
            INSERT INTO source_collections (id, name, pwd, glob_pattern, source_type, created_at, updated_at)
            VALUES (1, 'test-fs', '/home/user/notes', '**/*.md', 'filesystem', '2024-01-01T00:00:00', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO content (hash, doc, created_at)
            VALUES ('abc123', '# Test Doc', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO documents (id, source_collection_id, path, title, hash, active, modified_at)
            VALUES (1, 1, 'folder/doc.md', 'Test Doc', 'abc123', 1, '2024-01-15T10:30:00')
            """
        )
        conn.commit()

        # Now apply v0003 which includes backfill
        for migration in runner.get_migrations():
            if migration.version == 3:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # Verify Resource was created
        cursor = conn.execute("SELECT * FROM resources")
        resources = cursor.fetchall()
        assert len(resources) == 1

        resource = dict(resources[0])
        assert resource["source_collection_id"] == 1
        assert resource["uri"] == "file:///home/user/notes/folder/doc.md"
        assert resource["hash"] == "abc123"
        assert resource["load_status"] == "loaded"
        assert resource["index_state"] == "indexed"
        assert resource["load_method"] == "backfill"
        assert resource["index_method"] == "backfill"
        assert resource["loaded_at"] == "2024-01-15T10:30:00"
        assert resource["indexed_at"] == "2024-01-15T10:30:00"

        # Verify document was updated with resource_id
        cursor = conn.execute("SELECT resource_id FROM documents WHERE id = 1")
        doc = cursor.fetchone()
        assert doc["resource_id"] == resource["id"]

        conn.close()

    def test_v0003_adds_missing_source_type_column(self, tmp_path: Path):
        """v0003 should handle older schemas missing source_collections.source_type.

        Some older databases may have a source_collections table (typically
        renamed from collections) without the source_type/source_config columns.
        v0003's backfill queries reference sc.source_type; ensure the migration
        adds the column with a safe default and proceeds.
        """
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Simulate a pre-source_type schema at user_version=2.
        conn.executescript(
            """
            CREATE TABLE source_collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                pwd TEXT NOT NULL,
                glob_pattern TEXT NOT NULL DEFAULT '**/*.md',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_collection_id INTEGER NOT NULL REFERENCES source_collections(id),
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                modified_at TEXT NOT NULL,
                UNIQUE(source_collection_id, path)
            );

            CREATE TABLE source_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id),
                source_uri TEXT NOT NULL,
                etag TEXT,
                last_modified TEXT,
                last_fetched_at TEXT NOT NULL,
                fetch_duration_ms INTEGER,
                http_status INTEGER,
                content_type TEXT,
                extra_metadata TEXT
            );

            PRAGMA user_version = 2;
            """
        )

        conn.execute(
            """
            INSERT INTO source_collections (id, name, pwd, glob_pattern, created_at, updated_at)
            VALUES (1, 'test-fs', '/home/user/notes', '**/*.md', '2024-01-01T00:00:00', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO documents (id, source_collection_id, path, title, hash, active, modified_at)
            VALUES (1, 1, 'doc.md', 'Doc', 'abc123', 1, '2024-01-15T10:30:00')
            """
        )
        conn.commit()

        runner = MigrationRunner(conn)
        assert runner.get_version() == 2

        runner.run()

        # Migration should have added source_type and backfilled resources.
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(source_collections)")
        }
        assert "source_type" in cols
        assert "source_config" in cols

        resource = dict(conn.execute("SELECT * FROM resources").fetchone())
        assert resource["uri"] == "file:///home/user/notes/doc.md"

        doc = conn.execute("SELECT resource_id FROM documents WHERE id = 1").fetchone()
        assert doc["resource_id"] == resource["id"]

        conn.close()

    def test_backfill_handles_remote_documents_with_source_metadata(self, tmp_path: Path):
        """Backfill should use source_metadata.source_uri for non-filesystem docs."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        runner = MigrationRunner(conn)

        # Apply v0001 and v0002
        for migration in runner.get_migrations():
            if migration.version <= 2:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # Insert test data: an HTTP collection with a document that has source_metadata
        conn.execute(
            """
            INSERT INTO source_collections (id, name, pwd, glob_pattern, source_type, created_at, updated_at)
            VALUES (1, 'test-http', '/cache', '*', 'http', '2024-01-01T00:00:00', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO content (hash, doc, created_at)
            VALUES ('def456', '# Remote Doc', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO documents (id, source_collection_id, path, title, hash, active, modified_at)
            VALUES (1, 1, 'cached_doc.md', 'Remote Doc', 'def456', 1, '2024-01-20T14:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO source_metadata (document_id, source_uri, last_fetched_at)
            VALUES (1, 'https://example.com/api/doc.md', '2024-01-20T14:00:00')
            """
        )
        conn.commit()

        # Apply v0003 (includes backfill)
        for migration in runner.get_migrations():
            if migration.version == 3:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # Verify Resource was created with the source_uri
        cursor = conn.execute("SELECT * FROM resources")
        resources = cursor.fetchall()
        assert len(resources) == 1

        resource = dict(resources[0])
        assert resource["uri"] == "https://example.com/api/doc.md"
        assert resource["hash"] == "def456"
        assert resource["load_status"] == "loaded"
        assert resource["index_state"] == "indexed"

        # Verify document was linked
        cursor = conn.execute("SELECT resource_id FROM documents WHERE id = 1")
        doc = cursor.fetchone()
        assert doc["resource_id"] == resource["id"]

        conn.close()

    def test_backfill_skips_inactive_documents(self, tmp_path: Path):
        """Backfill should not create Resources for inactive documents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        runner = MigrationRunner(conn)

        # Apply v0001 and v0002
        for migration in runner.get_migrations():
            if migration.version <= 2:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # Insert an inactive document
        conn.execute(
            """
            INSERT INTO source_collections (id, name, pwd, glob_pattern, source_type, created_at, updated_at)
            VALUES (1, 'test-fs', '/home/user/notes', '**/*.md', 'filesystem', '2024-01-01T00:00:00', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO content (hash, doc, created_at)
            VALUES ('inactive123', '# Inactive', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO documents (id, source_collection_id, path, title, hash, active, modified_at)
            VALUES (1, 1, 'deleted.md', 'Deleted Doc', 'inactive123', 0, '2024-01-15T10:30:00')
            """
        )
        conn.commit()

        # Apply v0003
        for migration in runner.get_migrations():
            if migration.version == 3:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # No Resource should be created for inactive document
        cursor = conn.execute("SELECT COUNT(*) FROM resources")
        count = cursor.fetchone()[0]
        assert count == 0

        # Document should still have null resource_id
        cursor = conn.execute("SELECT resource_id FROM documents WHERE id = 1")
        doc = cursor.fetchone()
        assert doc["resource_id"] is None

        conn.close()

    def test_backfill_handles_duplicate_uris_gracefully(self, tmp_path: Path):
        """Backfill should handle multiple docs with same source_uri."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        runner = MigrationRunner(conn)

        # Apply v0001 and v0002
        for migration in runner.get_migrations():
            if migration.version <= 2:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # Insert two documents pointing to the same source_uri (edge case)
        conn.execute(
            """
            INSERT INTO source_collections (id, name, pwd, glob_pattern, source_type, created_at, updated_at)
            VALUES (1, 'test-http', '/cache', '*', 'http', '2024-01-01T00:00:00', '2024-01-01T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO content (hash, doc, created_at)
            VALUES ('hash1', '# Doc 1', '2024-01-01T00:00:00'),
                   ('hash2', '# Doc 2', '2024-01-02T00:00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO documents (id, source_collection_id, path, title, hash, active, modified_at)
            VALUES (1, 1, 'doc1.md', 'Doc 1', 'hash1', 1, '2024-01-20T14:00:00'),
                   (2, 1, 'doc2.md', 'Doc 2', 'hash2', 1, '2024-01-21T14:00:00')
            """
        )
        # Both docs point to the same source_uri (unusual but possible)
        conn.execute(
            """
            INSERT INTO source_metadata (document_id, source_uri, last_fetched_at)
            VALUES (1, 'https://example.com/shared.md', '2024-01-20T14:00:00'),
                   (2, 'https://example.com/shared.md', '2024-01-21T14:00:00')
            """
        )
        conn.commit()

        # Apply v0003 - should not fail due to unique constraint
        for migration in runner.get_migrations():
            if migration.version == 3:
                migration.up(conn)
                runner.set_version(migration.version)
                conn.commit()

        # Should have exactly one Resource (due to UNIQUE constraint and INSERT OR IGNORE)
        cursor = conn.execute("SELECT * FROM resources")
        resources = cursor.fetchall()
        assert len(resources) == 1

        # Both documents should be linked to the same resource
        cursor = conn.execute("SELECT resource_id FROM documents ORDER BY id")
        docs = cursor.fetchall()
        assert docs[0]["resource_id"] == docs[1]["resource_id"]
        assert docs[0]["resource_id"] == resources[0]["id"]

        conn.close()


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
