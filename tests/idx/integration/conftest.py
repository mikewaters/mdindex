"""Fixtures for integration tests.

Provides common mocks for embedding models and vector stores to avoid
loading real models during testing.
"""

from unittest.mock import MagicMock, patch

import pytest

from idx.ingest.pipelines import IngestPipeline


@pytest.fixture
def mock_embed_model():
    """Create a mock embedding model that returns fake embeddings."""
    mock = MagicMock()
    # Return fake embeddings (384 dimensions to match common models)
    mock.get_text_embedding_batch.return_value = [
        [0.1] * 384 for _ in range(100)  # Return enough embeddings for most tests
    ]
    mock.get_text_embedding.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_vector_manager():
    """Create a mock VectorStoreManager."""
    mock = MagicMock()
    mock_index = MagicMock()
    mock.load_or_create.return_value = mock_index
    return mock


@pytest.fixture
def patched_embedding(mock_embed_model, mock_vector_manager):
    """Patch IngestPipeline to use mock embedding model and vector store.

    This fixture automatically patches the embedding and vector store
    for tests that use it, avoiding loading real models.
    """
    with patch.object(IngestPipeline, "_get_embed_model", return_value=mock_embed_model):
        with patch.object(IngestPipeline, "_get_vector_store_manager", return_value=mock_vector_manager):
            yield {
                "embed_model": mock_embed_model,
                "vector_manager": mock_vector_manager,
            }
