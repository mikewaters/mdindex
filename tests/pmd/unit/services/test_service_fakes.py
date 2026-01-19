"""Tests demonstrating services work with facades.

These tests verify that services can be constructed with facade classes
using real in-memory SQLite databases, enabling fast unit tests.
"""

from pathlib import Path
from unittest.mock import Mock

from pmd.store import IndexFacade, LoadFacade, SearchFacade, StatusFacade
from pmd.search.adapters import FTS5TextSearcher
from pmd.services.indexing import IndexingService
from pmd.services.loading import LoadingService
from pmd.services.search import SearchService
from pmd.services.status import StatusService
from pmd.store.database import Database
from pmd.store.repositories.fts import FTS5SearchRepository


class TestIndexingServiceWithFacade:
    """Tests for IndexingService with facade."""

    def test_can_construct_with_facade(self, db: Database):
        """IndexingService should accept facade."""
        facade = IndexFacade(db)
        loader = Mock(spec=LoadingService)

        service = IndexingService(
            facade=facade,
            loader=loader,
        )

        assert service._data is facade
        assert service._loader is loader

    def test_vec_available_reflects_database(self, db: Database):
        """vec_available should reflect database capability."""
        facade = IndexFacade(db)
        loader = Mock(spec=LoadingService)

        service = IndexingService(
            facade=facade,
            loader=loader,
        )

        # DB should have vec available since we use sqlite-vec
        assert service.vec_available == db.vec_available


class TestStatusServiceWithFacade:
    """Tests for StatusService with facade."""

    def test_can_construct_with_facade(self, db: Database):
        """StatusService should accept facade."""
        facade = StatusFacade(db)

        service = StatusService(facade=facade)

        assert service._data is facade

    def test_can_construct_with_all_optional_deps(self, db: Database):
        """StatusService should accept all optional dependencies."""
        facade = StatusFacade(db)

        service = StatusService(
            facade=facade,
            db_path=Path("/tmp/test.db"),
            llm_provider_name="test-provider",
            vec_available=True,
        )

        assert service._db_path == Path("/tmp/test.db")
        assert service._llm_provider_name == "test-provider"

    def test_get_index_status_with_empty_database(self, db: Database):
        """get_index_status should work with empty database."""
        facade = StatusFacade(db)

        service = StatusService(facade=facade)

        status = service.get_index_status()

        assert status.total_documents == 0
        assert status.source_collections == []


class TestLoadingServiceWithFacade:
    """Tests for LoadingService with facade."""

    def test_can_construct_with_facade(self, db: Database):
        """LoadingService should accept facade."""
        facade = LoadFacade(db)

        service = LoadingService(facade=facade)

        assert service._data is facade
        assert service._source_registry is not None


class TestServiceIsolation:
    """Tests demonstrating service isolation with facades."""

    def test_services_with_independent_facades(self, db: Database):
        """Each service should work with its own facade."""
        # Create separate facades for each service
        index_facade = IndexFacade(db)
        status_facade = StatusFacade(db)
        load_facade = LoadFacade(db)

        indexing = IndexingService(
            facade=index_facade,
            loader=Mock(spec=LoadingService),
        )

        status = StatusService(
            facade=status_facade,
            vec_available=True,
        )

        loading = LoadingService(facade=load_facade)

        # Each service should work independently
        assert indexing.vec_available == db.vec_available
        assert status.vec_available is True
        # loading service doesn't have vec_available
