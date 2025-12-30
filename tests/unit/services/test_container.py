"""Tests for ServiceContainer."""

import pytest
from pathlib import Path

from pmd.core.config import Config
from pmd.services import ServiceContainer
from pmd.services.indexing import IndexingService
from pmd.services.search import SearchService
from pmd.services.status import StatusService
from pmd.store.collections import CollectionRepository
from pmd.store.documents import DocumentRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.search import FTS5SearchRepository


class TestServiceContainerInit:
    """Tests for ServiceContainer initialization."""

    def test_init_creates_database(self, config: Config):
        """Container should create Database instance."""
        container = ServiceContainer(config)

        assert container.db is not None
        assert container.config is config

    def test_init_not_connected(self, config: Config):
        """Container should not be connected after init."""
        container = ServiceContainer(config)

        assert container._connected is False

    def test_init_services_none(self, config: Config):
        """Container should have None services before lazy init."""
        container = ServiceContainer(config)

        assert container._indexing is None
        assert container._search is None
        assert container._status is None


class TestServiceContainerConnect:
    """Tests for ServiceContainer connection."""

    def test_connect_sets_connected(self, container: ServiceContainer):
        """connect() should set _connected to True."""
        container.connect()

        assert container._connected is True
        container.db.close()

    def test_connect_idempotent(self, container: ServiceContainer):
        """connect() should be idempotent."""
        container.connect()
        container.connect()  # Should not raise

        assert container._connected is True
        container.db.close()


class TestServiceContainerClose:
    """Tests for ServiceContainer close."""

    @pytest.mark.asyncio
    async def test_close_disconnects(self, connected_container: ServiceContainer):
        """close() should disconnect."""
        await connected_container.close()

        assert connected_container._connected is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self, connected_container: ServiceContainer):
        """close() should be idempotent."""
        await connected_container.close()
        await connected_container.close()  # Should not raise

        assert connected_container._connected is False


class TestServiceContainerContextManager:
    """Tests for ServiceContainer async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connects(self, config: Config):
        """Context manager should connect on enter."""
        async with ServiceContainer(config) as services:
            assert services._connected is True

    @pytest.mark.asyncio
    async def test_context_manager_disconnects(self, config: Config):
        """Context manager should disconnect on exit."""
        container = ServiceContainer(config)
        async with container:
            pass

        assert container._connected is False


class TestServiceContainerRepositories:
    """Tests for ServiceContainer repository accessors."""

    def test_collection_repo_lazy_init(self, connected_container: ServiceContainer):
        """collection_repo should be lazily initialized."""
        assert connected_container._collection_repo is None

        repo = connected_container.collection_repo

        assert repo is not None
        assert isinstance(repo, CollectionRepository)
        assert connected_container._collection_repo is repo

    def test_document_repo_lazy_init(self, connected_container: ServiceContainer):
        """document_repo should be lazily initialized."""
        assert connected_container._document_repo is None

        repo = connected_container.document_repo

        assert repo is not None
        assert isinstance(repo, DocumentRepository)
        assert connected_container._document_repo is repo

    def test_embedding_repo_lazy_init(self, connected_container: ServiceContainer):
        """embedding_repo should be lazily initialized."""
        assert connected_container._embedding_repo is None

        repo = connected_container.embedding_repo

        assert repo is not None
        assert isinstance(repo, EmbeddingRepository)
        assert connected_container._embedding_repo is repo

    def test_fts_repo_lazy_init(self, connected_container: ServiceContainer):
        """fts_repo should be lazily initialized."""
        assert connected_container._fts_repo is None

        repo = connected_container.fts_repo

        assert repo is not None
        assert isinstance(repo, FTS5SearchRepository)
        assert connected_container._fts_repo is repo

    def test_repos_reuse_same_instance(self, connected_container: ServiceContainer):
        """Repository accessors should return same instance."""
        repo1 = connected_container.collection_repo
        repo2 = connected_container.collection_repo

        assert repo1 is repo2


class TestServiceContainerServices:
    """Tests for ServiceContainer service accessors."""

    def test_indexing_service_lazy_init(self, connected_container: ServiceContainer):
        """indexing should be lazily initialized."""
        assert connected_container._indexing is None

        service = connected_container.indexing

        assert service is not None
        assert isinstance(service, IndexingService)
        assert connected_container._indexing is service

    def test_search_service_lazy_init(self, connected_container: ServiceContainer):
        """search should be lazily initialized."""
        assert connected_container._search is None

        service = connected_container.search

        assert service is not None
        assert isinstance(service, SearchService)
        assert connected_container._search is service

    def test_status_service_lazy_init(self, connected_container: ServiceContainer):
        """status should be lazily initialized."""
        assert connected_container._status is None

        service = connected_container.status

        assert service is not None
        assert isinstance(service, StatusService)
        assert connected_container._status is service

    def test_services_reuse_same_instance(self, connected_container: ServiceContainer):
        """Service accessors should return same instance."""
        service1 = connected_container.indexing
        service2 = connected_container.indexing

        assert service1 is service2


class TestServiceContainerVecAvailable:
    """Tests for ServiceContainer vec_available property."""

    def test_vec_available_reflects_db(self, connected_container: ServiceContainer):
        """vec_available should reflect db.vec_available."""
        # This will be True or False depending on sqlite-vec installation
        assert connected_container.vec_available == connected_container.db.vec_available
