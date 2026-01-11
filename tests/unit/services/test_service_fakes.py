"""Tests demonstrating services work with in-memory fakes.

These tests verify that services can be constructed and used without
a real database, enabling fast unit tests with test doubles.
"""

import pytest

from pmd.services.indexing import IndexingService
from pmd.services.search import SearchService
from pmd.services.status import StatusService
from tests.fakes import (
    InMemoryDatabase,
    InMemorySourceCollectionRepository,
    InMemoryDocumentRepository,
    InMemoryFTSRepository,
    InMemoryEmbeddingRepository,
    InMemoryLoadingService,
)


class TestIndexingServiceWithFakes:
    """Tests for IndexingService with in-memory fakes."""

    def test_can_construct_with_fakes(self):
        """IndexingService should accept in-memory fakes."""
        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()
        document_repo = InMemoryDocumentRepository()
        fts_repo = InMemoryFTSRepository()
        loader = InMemoryLoadingService()

        service = IndexingService(
            db=db,
            source_collection_repo=collection_repo,
            document_repo=document_repo,
            fts_repo=fts_repo,
            loader=loader,
        )

        assert service._db is db
        assert service._source_collection_repo is collection_repo
        assert service._document_repo is document_repo
        assert service._fts_repo is fts_repo
        assert service._loader is loader

    def test_can_construct_with_optional_embedding_repo(self):
        """IndexingService should accept optional embedding repo."""
        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()
        document_repo = InMemoryDocumentRepository()
        fts_repo = InMemoryFTSRepository()
        loader = InMemoryLoadingService()
        embedding_repo = InMemoryEmbeddingRepository()

        service = IndexingService(
            db=db,
            source_collection_repo=collection_repo,
            document_repo=document_repo,
            fts_repo=fts_repo,
            loader=loader,
            embedding_repo=embedding_repo,
        )

        assert service._embedding_repo is embedding_repo

    def test_vec_available_reflects_database(self):
        """vec_available should reflect database capability."""
        db_with_vec = InMemoryDatabase(_vec_available=True)
        db_without_vec = InMemoryDatabase(_vec_available=False)

        service_with = IndexingService(
            db=db_with_vec,
            source_collection_repo=InMemorySourceCollectionRepository(),
            document_repo=InMemoryDocumentRepository(),
            fts_repo=InMemoryFTSRepository(),
            loader=InMemoryLoadingService(),
        )

        service_without = IndexingService(
            db=db_without_vec,
            source_collection_repo=InMemorySourceCollectionRepository(),
            document_repo=InMemoryDocumentRepository(),
            fts_repo=InMemoryFTSRepository(),
            loader=InMemoryLoadingService(),
        )

        assert service_with.vec_available is True
        assert service_without.vec_available is False


class TestSearchServiceWithFakes:
    """Tests for SearchService with in-memory fakes."""

    def test_can_construct_with_fakes(self):
        """SearchService should accept in-memory fakes."""
        db = InMemoryDatabase()
        fts_repo = InMemoryFTSRepository()
        collection_repo = InMemorySourceCollectionRepository()

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=collection_repo,
        )

        assert service._db is db
        assert service._fts_repo is fts_repo
        assert service._source_collection_repo is collection_repo

    def test_can_construct_with_all_optional_deps(self):
        """SearchService should accept all optional dependencies."""
        db = InMemoryDatabase()
        fts_repo = InMemoryFTSRepository()
        collection_repo = InMemorySourceCollectionRepository()
        embedding_repo = InMemoryEmbeddingRepository()

        async def fake_embedding_generator():
            return None

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=collection_repo,
            embedding_repo=embedding_repo,
            embedding_generator_factory=fake_embedding_generator,
            fts_weight=2.0,
            vec_weight=0.5,
            rrf_k=100,
        )

        assert service._embedding_repo is embedding_repo
        assert service._fts_weight == 2.0
        assert service._vec_weight == 0.5
        assert service._rrf_k == 100

    def test_fts_search_with_fake_repo(self):
        """FTS search should work with fake repo."""
        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()
        fts_repo = InMemoryFTSRepository()

        # Pre-configure results using make_search_result helper
        from tests.fakes import make_search_result
        from pmd.core.types import SearchSource
        fts_repo.add_result(make_search_result(
            filepath="test.md",
            score=0.9,
            source=SearchSource.FTS,
            title="Test",
        ))

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=collection_repo,
        )

        results = service.fts_search("query")

        assert len(results) == 1
        assert results[0].filepath == "test.md"
        assert results[0].score == 0.9


class TestStatusServiceWithFakes:
    """Tests for StatusService with in-memory fakes."""

    def test_can_construct_with_fakes(self):
        """StatusService should accept in-memory fakes."""
        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()

        service = StatusService(
            db=db,
            source_collection_repo=collection_repo,
        )

        assert service._db is db
        assert service._source_collection_repo is collection_repo

    def test_can_construct_with_all_optional_deps(self):
        """StatusService should accept all optional dependencies."""
        from pathlib import Path

        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()

        async def fake_llm_check():
            return True

        service = StatusService(
            db=db,
            source_collection_repo=collection_repo,
            db_path=Path("/tmp/test.db"),
            llm_provider="test-provider",
            llm_available_check=fake_llm_check,
        )

        assert service._db_path == Path("/tmp/test.db")
        assert service._llm_provider == "test-provider"

    def test_get_index_status_with_empty_fakes(self):
        """get_index_status should work with empty fakes."""
        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()

        service = StatusService(
            db=db,
            source_collection_repo=collection_repo,
        )

        status = service.get_index_status()

        assert status.total_documents == 0
        assert status.source_collections == []

    def test_get_index_status_with_collections(self):
        """get_index_status should list collections from fake repo."""
        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()

        # Add a collection
        collection_repo.create("test-collection", "/path/to/docs")

        service = StatusService(
            db=db,
            source_collection_repo=collection_repo,
        )

        status = service.get_index_status()

        assert len(status.source_collections) == 1
        assert status.source_collections[0].name == "test-collection"


class TestServiceIsolation:
    """Tests demonstrating service isolation with fakes."""

    def test_services_with_independent_fakes(self):
        """Each service should work with its own set of fakes."""
        # Create separate fakes for each service
        indexing_db = InMemoryDatabase()
        search_db = InMemoryDatabase()
        status_db = InMemoryDatabase()

        indexing = IndexingService(
            db=indexing_db,
            source_collection_repo=InMemorySourceCollectionRepository(),
            document_repo=InMemoryDocumentRepository(),
            fts_repo=InMemoryFTSRepository(),
            loader=InMemoryLoadingService(),
        )

        search = SearchService(
            db=search_db,
            fts_repo=InMemoryFTSRepository(),
            source_collection_repo=InMemorySourceCollectionRepository(),
        )

        status = StatusService(
            db=status_db,
            source_collection_repo=InMemorySourceCollectionRepository(),
        )

        # Each service should work independently
        assert indexing.vec_available is True
        assert search.vec_available is True
        assert status.vec_available is True

    def test_services_with_shared_repos(self):
        """Services can share repository instances for integration tests."""
        # Shared infrastructure
        db = InMemoryDatabase()
        collection_repo = InMemorySourceCollectionRepository()

        # Create collection in shared repo
        collection_repo.create("shared", "/path")

        indexing = IndexingService(
            db=db,
            source_collection_repo=collection_repo,
            document_repo=InMemoryDocumentRepository(),
            fts_repo=InMemoryFTSRepository(),
            loader=InMemoryLoadingService(),
        )

        status = StatusService(
            db=db,
            source_collection_repo=collection_repo,
        )

        # Both services see the same collection
        status_result = status.get_index_status()
        assert len(status_result.source_collections) == 1
        assert status_result.source_collections[0].name == "shared"
