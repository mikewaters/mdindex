"""Tests for StatusService."""

import pytest
from pathlib import Path

from pmd.app import create_application
from pmd.core.config import Config
from pmd.core.types import IndexStatus
from pmd.services.status import StatusService
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.database import Database
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.embeddings import EmbeddingRepository
from pmd.store.repositories.fts import FTS5SearchRepository
from pmd.store.schema import EMBEDDING_DIMENSION


def make_status_service(
    db: Database,
    config: Config,
    source_collection_repo: SourceCollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
    embedding_repo: EmbeddingRepository | None = None,
    fts_repo: FTS5SearchRepository | None = None,
) -> StatusService:
    """Create a StatusService with default repositories.

    Args:
        db: Database instance.
        config: Application config.
        source_collection_repo: Optional custom source collection repo.
        document_repo: Optional custom document repo.
        embedding_repo: Optional custom embedding repo.
        fts_repo: Optional custom FTS repo.

    Returns:
        Configured StatusService.
    """
    return StatusService(
        document_repo=document_repo or DocumentRepository(db),
        embedding_repo=embedding_repo or EmbeddingRepository(db),
        fts_repo=fts_repo or FTS5SearchRepository(db),
        source_collection_repo=source_collection_repo or SourceCollectionRepository(db),
        db_path=config.db_path,
        vec_available=db.vec_available,
    )


class TestStatusServiceGetIndexStatus:
    """Tests for StatusService.get_index_status method."""

    def test_get_index_status_returns_index_status(self, db: Database, config: Config):
        """get_index_status should return IndexStatus object."""
        service = make_status_service(db, config)

        status = service.get_index_status()

        assert isinstance(status, IndexStatus)

    def test_get_index_status_empty_database(self, db: Database, config: Config):
        """get_index_status should return zeros for empty database."""
        service = make_status_service(db, config)

        status = service.get_index_status()

        assert status.total_documents == 0
        assert status.embedded_documents == 0
        assert status.source_collections == []
        assert status.cache_entries == 0

    def test_get_index_status_with_collections(
        self, db: Database, config: Config, tmp_path: Path
    ):
        """get_index_status should include collections."""
        source_collection_repo = SourceCollectionRepository(db)
        service = make_status_service(
            db, config, source_collection_repo=source_collection_repo
        )

        source_collection_repo.create("test", str(tmp_path), "**/*.md")

        status = service.get_index_status()

        assert len(status.source_collections) == 1
        assert status.source_collections[0].name == "test"

    def test_get_index_status_with_documents(
        self, db: Database, config: Config, tmp_path: Path
    ):
        """get_index_status should count documents."""
        source_collection_repo = SourceCollectionRepository(db)
        document_repo = DocumentRepository(db)
        service = make_status_service(
            db,
            config,
            source_collection_repo=source_collection_repo,
            document_repo=document_repo,
        )

        collection = source_collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content 1"
        )
        document_repo.add_or_update(
            collection.id, "doc2.md", "Doc 2", "# Content 2"
        )

        status = service.get_index_status()

        assert status.total_documents == 2


class TestStatusServiceGetFullStatus:
    """Tests for StatusService.get_full_status method."""

    @pytest.mark.asyncio
    async def test_get_full_status_returns_dict(self, config: Config):
        """get_full_status should return a dictionary."""
        async with await create_application(config) as app:
            status = await app.status.get_full_status()

            assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_get_full_status_includes_required_keys(self, config: Config):
        """get_full_status should include all required keys."""
        async with await create_application(config) as app:
            status = await app.status.get_full_status()

            assert "source_collections_count" in status
            assert "source_collections" in status
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
        async with await create_application(config) as app:
            app.source_collection_repo.create("my-collection", str(tmp_path), "*.txt")

            status = await app.status.get_full_status()

            assert status["source_collections_count"] == 1
            assert len(status["source_collections"]) == 1
            assert status["source_collections"][0]["name"] == "my-collection"
            assert status["source_collections"][0]["path"] == str(tmp_path)
            assert status["source_collections"][0]["glob_pattern"] == "*.txt"


class TestStatusServiceGetCollectionStats:
    """Tests for StatusService.get_collection_stats method."""

    def test_get_collection_stats_not_found(self, db: Database, config: Config):
        """get_collection_stats should return None for unknown collection."""
        service = make_status_service(db, config)

        stats = service.get_collection_stats("nonexistent")

        assert stats is None

    def test_get_collection_stats_found(
        self, db: Database, config: Config, tmp_path: Path
    ):
        """get_collection_stats should return stats for existing collection."""
        source_collection_repo = SourceCollectionRepository(db)
        service = make_status_service(
            db, config, source_collection_repo=source_collection_repo
        )

        source_collection_repo.create("test", str(tmp_path), "**/*.md")

        stats = service.get_collection_stats("test")

        assert stats is not None
        assert stats["name"] == "test"
        assert stats["path"] == str(tmp_path)
        assert stats["glob_pattern"] == "**/*.md"
        assert stats["documents"] == 0
        assert stats["embedded"] == 0

    def test_get_collection_stats_with_documents(
        self, db: Database, config: Config, tmp_path: Path
    ):
        """get_collection_stats should count documents."""
        source_collection_repo = SourceCollectionRepository(db)
        document_repo = DocumentRepository(db)
        service = make_status_service(
            db,
            config,
            source_collection_repo=source_collection_repo,
            document_repo=document_repo,
        )

        collection = source_collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content"
        )

        stats = service.get_collection_stats("test")

        assert stats["documents"] == 1


class TestStatusServiceGetIndexSyncReport:
    """Tests for StatusService.get_index_sync_report method."""

    def test_sync_report_missing_fts_and_vectors(
        self, db: Database, config: Config, tmp_path: Path
    ):
        """Report should flag documents missing FTS and vectors."""
        source_collection_repo = SourceCollectionRepository(db)
        document_repo = DocumentRepository(db)
        service = make_status_service(
            db,
            config,
            source_collection_repo=source_collection_repo,
            document_repo=document_repo,
        )

        collection = source_collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content 1"
        )

        report = service.get_index_sync_report()

        assert report["missing_fts_count"] == 1
        assert report["missing_vectors_count"] == 1

    def test_sync_report_with_fts_and_vectors(
        self, db: Database, config: Config, tmp_path: Path
    ):
        """Report should show zero missing when FTS and vectors are present."""
        source_collection_repo = SourceCollectionRepository(db)
        document_repo = DocumentRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)
        service = make_status_service(
            db,
            config,
            source_collection_repo=source_collection_repo,
            document_repo=document_repo,
            fts_repo=fts_repo,
            embedding_repo=embedding_repo,
        )

        collection = source_collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        doc, _ = document_repo.add_or_update(
            collection.id, "doc1.md", "Doc 1", "# Content 1"
        )

        # Add FTS entry
        cursor = db.execute(
            "SELECT id FROM documents WHERE source_collection_id = ? AND path = ?",
            (collection.id, doc.filepath),
        )
        doc_id = cursor.fetchone()["id"]
        fts_repo.index_document(doc_id, doc.filepath, "# Content 1")

        # Add vector metadata entry
        embedding_repo.store_embedding(
            doc.hash, 0, 0, [0.1] * EMBEDDING_DIMENSION, "test-model"
        )

        report = service.get_index_sync_report()

        assert report["missing_fts_count"] == 0
        assert report["missing_vectors_count"] == 0
