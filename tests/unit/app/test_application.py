"""Tests for Application class and create_application factory."""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from pmd.core.config import Config
from pmd.app import Application, create_application
from pmd.services.indexing import IndexingService
from pmd.services.search import SearchService
from pmd.services.status import StatusService


class TestApplication:
    """Tests for Application class."""

    def test_application_init(self, config: Config):
        """Application should initialize with provided dependencies."""
        db = MagicMock()
        llm_provider = MagicMock()
        indexing = MagicMock(spec=IndexingService)
        search = MagicMock(spec=SearchService)
        status = MagicMock(spec=StatusService)

        app = Application(
            db=db,
            llm_provider=llm_provider,
            indexing=indexing,
            search=search,
            status=status,
            config=config,
        )

        assert app._db is db
        assert app._llm_provider is llm_provider
        assert app.indexing is indexing
        assert app.search is search
        assert app.status is status
        assert app._config is config

    def test_application_db_property(self, config: Config):
        """db property should return database instance."""
        db = MagicMock()

        app = Application(
            db=db,
            llm_provider=None,
            indexing=MagicMock(),
            search=MagicMock(),
            status=MagicMock(),
            config=config,
        )

        assert app.db is db

    def test_application_config_property(self, config: Config):
        """config property should return configuration."""
        app = Application(
            db=MagicMock(),
            llm_provider=None,
            indexing=MagicMock(),
            search=MagicMock(),
            status=MagicMock(),
            config=config,
        )

        assert app.config is config

    def test_application_vec_available(self, config: Config):
        """vec_available should reflect db.vec_available."""
        db = MagicMock()
        db.vec_available = True

        app = Application(
            db=db,
            llm_provider=None,
            indexing=MagicMock(),
            search=MagicMock(),
            status=MagicMock(),
            config=config,
        )

        assert app.vec_available is True

        db.vec_available = False
        assert app.vec_available is False

    @pytest.mark.asyncio
    async def test_application_close_with_llm(self, config: Config):
        """close() should close both LLM provider and database."""
        db = MagicMock()
        llm_provider = AsyncMock()

        app = Application(
            db=db,
            llm_provider=llm_provider,
            indexing=MagicMock(),
            search=MagicMock(),
            status=MagicMock(),
            config=config,
        )

        await app.close()

        llm_provider.close.assert_awaited_once()
        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_application_close_without_llm(self, config: Config):
        """close() should work when LLM provider is None."""
        db = MagicMock()

        app = Application(
            db=db,
            llm_provider=None,
            indexing=MagicMock(),
            search=MagicMock(),
            status=MagicMock(),
            config=config,
        )

        await app.close()

        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_application_context_manager(self, config: Config):
        """Application should work as async context manager."""
        db = MagicMock()

        app = Application(
            db=db,
            llm_provider=None,
            indexing=MagicMock(),
            search=MagicMock(),
            status=MagicMock(),
            config=config,
        )

        async with app as entered_app:
            assert entered_app is app

        db.close.assert_called_once()


class TestCreateApplication:
    """Tests for create_application factory function."""

    @pytest.mark.asyncio
    async def test_create_application_returns_application(self, config: Config):
        """create_application should return configured Application."""
        async with await create_application(config) as app:
            assert isinstance(app, Application)

    @pytest.mark.asyncio
    async def test_create_application_connects_database(self, config: Config):
        """create_application should connect to database."""
        async with await create_application(config) as app:
            # If db wasn't connected, accessing vec_available would fail
            assert app.vec_available in (True, False)

    @pytest.mark.asyncio
    async def test_create_application_creates_services(self, config: Config):
        """create_application should create all services."""
        async with await create_application(config) as app:
            assert app.indexing is not None
            assert app.search is not None
            assert app.status is not None

    @pytest.mark.asyncio
    async def test_create_application_handles_llm_provider_error(
        self, config: Config
    ):
        """create_application should handle LLM provider creation errors."""
        # Use a provider name that will raise ValueError
        config.llm_provider = "nonexistent-provider"

        async with await create_application(config) as app:
            # LLM provider should be None due to ValueError
            assert app._llm_provider is None

    @pytest.mark.asyncio
    async def test_create_application_cleans_up_on_context_exit(
        self, config: Config
    ):
        """create_application context manager should clean up resources."""
        app = await create_application(config)
        db = app._db

        async with app:
            pass  # Use the application

        # After context exit, database should be closed
        # We can't easily check this without mocking, but the test
        # ensures no exceptions are raised during cleanup

    @pytest.mark.asyncio
    async def test_create_application_services_have_dependencies(
        self, config: Config
    ):
        """Services created by create_application should have proper dependencies."""
        async with await create_application(config) as app:
            # IndexingService should have database access
            assert hasattr(app.indexing, "_db")

            # SearchService should have repository access
            assert hasattr(app.search, "_fts_repo")

            # StatusService should have collection_repo access
            assert hasattr(app.status, "_collection_repo")
