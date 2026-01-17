"""Integration tests for LoadingService + IndexingService flow.

These tests verify that the loader and indexer work together correctly
through the Application composition root.
"""

import pytest
from pathlib import Path

from pmd.app import create_application
from pmd.core.config import Config
from pmd.services import IndexResult
from pmd.services.loading import LoadingService


class TestLoaderIndexerIntegration:
    """Integration tests for loader + indexer flow."""

    @pytest.mark.asyncio
    async def test_full_index_flow_via_loader(self, config: Config, tmp_path: Path):
        """Index collection through Application uses the loader internally."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Document One\n\nContent one.")
        (tmp_path / "doc2.md").write_text("# Document Two\n\nContent two.")

        app = await create_application(config)
        async with app:
            # Create collection
            app.db.execute(
                """
                INSERT INTO source_collections (name, pwd, glob_pattern, source_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                ("test", str(tmp_path), "**/*.md", "filesystem"),
            )

            # Index without passing source - loader should resolve it
            result = await app.indexing.index_collection("test")

            assert isinstance(result, IndexResult)
            assert result.indexed == 2
            assert result.skipped == 0
            assert result.errors == []

            # Verify documents are indexed
            cursor = app.db.execute(
                "SELECT COUNT(*) as count FROM documents WHERE active = 1"
            )
            assert cursor.fetchone()["count"] == 2

    @pytest.mark.asyncio
    async def test_stale_document_cleanup(self, config: Config, tmp_path: Path):
        """Removed files are marked inactive after reindex."""
        # Create initial files
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc1.write_text("# Document One\n\nContent one.")
        doc2.write_text("# Document Two\n\nContent two.")

        app = await create_application(config)
        async with app:
            # Create collection
            app.db.execute(
                """
                INSERT INTO source_collections (name, pwd, glob_pattern, source_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                ("test", str(tmp_path), "**/*.md", "filesystem"),
            )

            # First index - both docs
            result1 = await app.indexing.index_collection("test")
            assert result1.indexed == 2

            # Remove one file
            doc2.unlink()

            # Reindex - should mark doc2 as inactive
            result2 = await app.indexing.index_collection("test")
            # doc1 unchanged, doc2 removed
            assert result2.indexed == 0  # doc1 skipped (unchanged)
            assert result2.skipped == 1  # doc1 skipped during load

            # Verify doc2 is marked inactive
            cursor = app.db.execute(
                "SELECT active FROM documents WHERE path LIKE '%doc2.md'"
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["active"] == 0

            # Verify doc1 is still active
            cursor = app.db.execute(
                "SELECT active FROM documents WHERE path LIKE '%doc1.md'"
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["active"] == 1

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, config: Config, tmp_path: Path):
        """Only changed documents are reloaded and persisted."""
        # Create initial files
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc1.write_text("# Document One\n\nContent one.")
        doc2.write_text("# Document Two\n\nContent two.")

        app = await create_application(config)
        async with app:
            # Create collection
            app.db.execute(
                """
                INSERT INTO source_collections (name, pwd, glob_pattern, source_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                ("test", str(tmp_path), "**/*.md", "filesystem"),
            )

            # First index
            result1 = await app.indexing.index_collection("test")
            assert result1.indexed == 2

            # Second index without changes - should skip both
            result2 = await app.indexing.index_collection("test")
            assert result2.indexed == 0
            # Skipped during loading, not during persistence
            # So skipped count may be 0 since we skip at load time

            # Modify one file
            doc1.write_text("# Document One Updated\n\nNew content.")

            # Third index - should only index the changed file
            result3 = await app.indexing.index_collection("test")
            assert result3.indexed == 1  # Only doc1 changed

    @pytest.mark.asyncio
    async def test_loader_accessible_on_application(self, config: Config, tmp_path: Path):
        """Application exposes the loading service."""
        app = await create_application(config)
        async with app:
            # Verify loading service is accessible
            assert hasattr(app, "loading")
            assert app.loading is not None
            assert isinstance(app.loading, LoadingService)


class TestApplicationWithLoader:
    """Tests for Application composition with LoadingService."""

    @pytest.mark.asyncio
    async def test_create_application_wires_loading_service(self, config: Config):
        """create_application wires LoadingService correctly."""
        app = await create_application(config)
        async with app:
            # Loading service should be wired
            assert app.loading is not None

            # Indexing service should have the loader
            assert app.indexing._loader is not None
            assert app.indexing._loader is app.loading


