"""Pytest configuration and fixtures for service layer tests."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from pmd.app import Application, create_application
from pmd.core.config import Config
from pmd.store.database import Database


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Provide a Config instance for testing."""
    cfg = Config()
    cfg.db_path = tmp_path / "test.db"
    return cfg


@pytest.fixture
def db(config: Config) -> Database:
    """Provide a connected database instance."""
    database = Database(config.db_path)
    database.connect()
    yield database
    database.close()


@pytest.fixture
async def app(config: Config) -> Application:
    """Provide an Application instance for service tests."""
    application = await create_application(config)
    async with application:
        yield application


@pytest.fixture
def mock_llm_provider():
    """Provide a mock LLM provider."""
    provider = AsyncMock()
    provider.is_available = AsyncMock(return_value=True)
    provider.embed = AsyncMock(return_value=[[0.1] * 768])
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_embedding_generator():
    """Provide a mock embedding generator."""
    generator = AsyncMock()
    generator.embed_query = AsyncMock(return_value=[0.1] * 768)
    generator.embed_document = AsyncMock(return_value=3)
    return generator
