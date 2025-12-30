"""Pytest configuration and fixtures for service layer tests."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pmd.core.config import Config
from pmd.services import ServiceContainer
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
def container(config: Config) -> ServiceContainer:
    """Provide a ServiceContainer instance (not connected)."""
    return ServiceContainer(config)


@pytest.fixture
def connected_container(config: Config) -> ServiceContainer:
    """Provide a connected ServiceContainer instance."""
    container = ServiceContainer(config)
    container.connect()
    yield container
    # Close synchronously (no async cleanup in unit tests)
    if container._connected:
        container.db.close()


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
