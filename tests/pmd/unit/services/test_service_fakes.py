"""Tests demonstrating services work with data access layer.

These tests verify that services can be constructed with data access classes
using real in-memory SQLite databases, enabling fast unit tests.
"""

from pathlib import Path
from unittest.mock import Mock

from pmd.data import IndexingData, LoadingData, SearchData, StatusData
from pmd.search.adapters import FTS5TextSearcher
from pmd.services.indexing import IndexingService
from pmd.services.loading import LoadingService
from pmd.services.search import SearchService
from pmd.services.status import StatusService
from pmd.sources import get_default_registry
from pmd.store.database import Database
from pmd.store.repositories.fts import FTS5SearchRepository


class TestIndexingServiceWithDataAccess:
    """Tests for IndexingService with data access layer."""

    def test_can_construct_with_data_access(self, db: Database):
        """IndexingService should accept data access layer."""
        data = IndexingData(db)
        loader = Mock(spec=LoadingService)

        service = IndexingService(
            data=data,
            loader=loader,
        )

        assert service._data is data
        assert service._loader is loader

    def test_vec_available_reflects_database(self, db: Database):
        """vec_available should reflect database capability."""
        data = IndexingData(db)
        loader = Mock(spec=LoadingService)

        service = IndexingService(
            data=data,
            loader=loader,
        )

        # DB should have vec available since we use sqlite-vec
        assert service.vec_available == db.vec_available


class TestSearchServiceWithDataAccess:
    """Tests for SearchService with data access layer."""

    def test_can_construct_with_data_access(self, db: Database):
        """SearchService should accept data access layer."""
        data = SearchData(db)
        fts_repo = FTS5SearchRepository(db)
        text_searcher = FTS5TextSearcher(fts_repo)

        service = SearchService(
            data=data,
            text_searcher=text_searcher,
        )

        assert service._data is data
        assert service._text_searcher is text_searcher

    def test_can_construct_with_all_optional_deps(self, db: Database):
        """SearchService should accept all optional dependencies."""
        data = SearchData(db)
        fts_repo = FTS5SearchRepository(db)
        text_searcher = FTS5TextSearcher(fts_repo)

        service = SearchService(
            data=data,
            text_searcher=text_searcher,
            fts_weight=2.0,
            vec_weight=0.5,
            rrf_k=100,
        )

        assert service._fts_weight == 2.0
        assert service._vec_weight == 0.5
        assert service._rrf_k == 100


class TestStatusServiceWithDataAccess:
    """Tests for StatusService with data access layer."""

    def test_can_construct_with_data_access(self, db: Database):
        """StatusService should accept data access layer."""
        data = StatusData(db)

        service = StatusService(
            data=data,
        )

        assert service._data is data

    def test_can_construct_with_all_optional_deps(self, db: Database):
        """StatusService should accept all optional dependencies."""
        data = StatusData(db)

        async def fake_llm_check():
            return True

        service = StatusService(
            data=data,
            db_path=Path("/tmp/test.db"),
            llm_provider="test-provider",
            llm_available_check=fake_llm_check,
            vec_available=True,
        )

        assert service._db_path == Path("/tmp/test.db")
        assert service._llm_provider == "test-provider"

    def test_get_index_status_with_empty_database(self, db: Database):
        """get_index_status should work with empty database."""
        data = StatusData(db)

        service = StatusService(
            data=data,
        )

        status = service.get_index_status()

        assert status.total_documents == 0
        assert status.source_collections == []


class TestLoadingServiceWithDataAccess:
    """Tests for LoadingService with data access layer."""

    def test_can_construct_with_data_access(self, db: Database):
        """LoadingService should accept data access layer."""
        data = LoadingData(db)
        source_registry = get_default_registry()

        service = LoadingService(
            data=data,
            source_registry=source_registry,
        )

        assert service._data is data
        assert service._source_registry is source_registry


class TestServiceIsolation:
    """Tests demonstrating service isolation with data access layers."""

    def test_services_with_independent_data_access(self, db: Database):
        """Each service should work with its own data access layer."""
        # Create separate data access layers for each service
        indexing_data = IndexingData(db)
        search_data = SearchData(db)
        status_data = StatusData(db)
        loading_data = LoadingData(db)

        indexing = IndexingService(
            data=indexing_data,
            loader=Mock(spec=LoadingService),
        )

        fts_repo = FTS5SearchRepository(db)
        search = SearchService(
            data=search_data,
            text_searcher=FTS5TextSearcher(fts_repo),
        )

        status = StatusService(
            data=status_data,
            vec_available=True,
        )

        loading = LoadingService(
            data=loading_data,
            source_registry=get_default_registry(),
        )

        # Each service should work independently
        assert indexing.vec_available == db.vec_available
        assert search.vec_available == db.vec_available
        assert status.vec_available is True
        # loading service doesn't have vec_available
