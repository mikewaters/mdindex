"""Pytest fixtures for app module tests."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from pmd.core.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Provide a Config instance for testing."""
    cfg = Config()
    cfg.db_path = tmp_path / "test.db"
    return cfg


@pytest.fixture
def mock_llm_provider():
    """Provide a mock LLM provider."""
    provider = AsyncMock()
    provider.is_available = AsyncMock(return_value=True)
    provider.embed = AsyncMock(return_value=[[0.1] * 768])
    provider.close = AsyncMock()
    return provider
