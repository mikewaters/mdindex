"""Tests for StatusService."""

import pytest
from pathlib import Path

from pmd.core.config import Config
from pmd.core.types import IndexStatus
from pmd.services import ServiceContainer
from pmd.services.status import StatusService
from pmd.store.schema import EMBEDDING_DIMENSION


class TestStatusServiceInit:
    """Tests for StatusService initialization."""

    def test_init_with_explicit_deps(self, connected_container: ServiceContainer):
        """StatusService should work with explicit dependencies."""
        service = StatusService(
            db=connected_container.db,
            collection_repo=connected_container.collection_repo,
            db_path=connected_container.config.db_path,
        )

        assert service._db is connected_container.db
        assert service._collection_repo is connected_container.collection_repo

    def test_from_container_factory(self, connected_container: ServiceContainer):
        """StatusService.from_container should create service with container deps."""
        service = StatusService.from_container(connected_container)

        assert service._db is connected_container.db
        assert service._collection_repo is connected_container.collection_repo


class TestStatusServiceGetIndexStatus:
    """Tests for StatusService.get_index_status method."""

    def test_get_index_status_returns_index_status(
        self, connected_container: ServiceContainer
    ):
        """get_index_status should return IndexStatus object."""
        status = connected_container.status.get_index_status()

        assert isinstance(status, IndexStatus)

    def test_get_index_status_empty_database(
        self, connected_container: ServiceContainer
    ):
        """get_index_status should return zeros for empty database."""
        status = connected_container.status.get_index_status()

        assert status.total_documents == 0
        assert status.embedded_documents == 0
        assert status.collections == []
        assert status.cache_entries == 0

    def test_get_index_status_with_collections(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """get_index_status should include collections."""
        connected_container.collection_repo.create("test", str(tmp_path), "**/*.md")

        status = connected_container.status.get_index_status()

        assert len(status.collections) == 1
        assert status.collections[0].name == "test"

    def test_get_index_status_with_documents(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """get_index_status should count documents."""
        collection = connected_container.collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        connected_container.document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content 1"
        )
        connected_container.document_repo.add_or_update(
            collection.id, "doc2.md", "Doc 2", "# Content 2"
        )

        status = connected_container.status.get_index_status()

        assert status.total_documents == 2


class TestStatusServiceGetFullStatus:
    """Tests for StatusService.get_full_status method."""

    @pytest.mark.asyncio
    async def test_get_full_status_returns_dict(self, config: Config):
        """get_full_status should return a dictionary."""
        async with ServiceContainer(config) as services:
            status = await services.status.get_full_status()

            assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_get_full_status_includes_required_keys(self, config: Config):
        """get_full_status should include all required keys."""
        async with ServiceContainer(config) as services:
            status = await services.status.get_full_status()

            assert "collections_count" in status
            assert "collections" in status
            assert "total_documents" in status
            assert "embedded_documents" in status
            assert "index_size_bytes" in status
            assert "database_path" in status
            assert "llm_provider" in status
            assert "llm_available" in status
            assert "vec_available" in status

    @pytest.mark.asyncio
    async def test_get_full_status_with_collections(
        self, config: Config, tmp_path: Path
    ):
        """get_full_status should include collection details."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create("my-collection", str(tmp_path), "*.txt")

            status = await services.status.get_full_status()

            assert status["collections_count"] == 1
            assert len(status["collections"]) == 1
            assert status["collections"][0]["name"] == "my-collection"
            assert status["collections"][0]["path"] == str(tmp_path)
            assert status["collections"][0]["glob_pattern"] == "*.txt"


class TestStatusServiceGetCollectionStats:
    """Tests for StatusService.get_collection_stats method."""

    def test_get_collection_stats_not_found(
        self, connected_container: ServiceContainer
    ):
        """get_collection_stats should return None for unknown collection."""
        stats = connected_container.status.get_collection_stats("nonexistent")

        assert stats is None

    def test_get_collection_stats_found(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """get_collection_stats should return stats for existing collection."""
        connected_container.collection_repo.create("test", str(tmp_path), "**/*.md")

        stats = connected_container.status.get_collection_stats("test")

        assert stats is not None
        assert stats["name"] == "test"
        assert stats["path"] == str(tmp_path)
        assert stats["glob_pattern"] == "**/*.md"
        assert stats["documents"] == 0
        assert stats["embedded"] == 0

    def test_get_collection_stats_with_documents(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """get_collection_stats should count documents."""
        collection = connected_container.collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        connected_container.document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content"
        )

        stats = connected_container.status.get_collection_stats("test")

        assert stats["documents"] == 1


class TestStatusServiceGetIndexSyncReport:
    """Tests for StatusService.get_index_sync_report method."""

    def test_sync_report_missing_fts_and_vectors(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """Report should flag documents missing FTS and vectors."""
        collection = connected_container.collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        connected_container.document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content 1"
        )

        report = connected_container.status.get_index_sync_report()

        assert report["missing_fts_count"] == 1
        assert report["missing_vectors_count"] == 1

    def test_sync_report_with_fts_and_vectors(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """Report should show zero missing when FTS and vectors are present."""
        collection = connected_container.collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        doc, _ = connected_container.document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content 1"
        )

        # Add FTS entry
        cursor = connected_container.db.execute(
            "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
            (collection.id, doc.filepath),
        )
        doc_id = cursor.fetchone()["id"]
        connected_container.fts_repo.index_document(doc_id, doc.filepath, "# Content 1")

        # Add vector metadata entry
        connected_container.embedding_repo.store_embedding(
            doc.hash, 0, 0, [0.1] * EMBEDDING_DIMENSION, "test-model"
        )

        report = connected_container.status.get_index_sync_report()

        assert report["missing_fts_count"] == 0
        assert report["missing_vectors_count"] == 0
