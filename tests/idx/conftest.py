"""Shared fixtures for idx tests.

Provides common mocks for embedding models and vector stores to avoid
loading real models during testing.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.embeddings import BaseEmbedding

from idx.ingest.pipelines import IngestPipeline


class MockEmbedding(BaseEmbedding):
    """Mock embedding model that extends LlamaIndex BaseEmbedding.

    Returns deterministic fake embeddings without loading any ML models.
    Compatible with LlamaIndex's IngestionPipeline transformations.
    """

    embed_dim: int = 384
    _call_count: int = 0

    def __init__(self, embed_dim: int = 384, **kwargs: Any):
        """Initialize with embedding dimension."""
        super().__init__(
            model_name="mock-embedding",
            embed_batch_size=32,
            **kwargs,
        )
        self.embed_dim = embed_dim
        self._call_count = 0

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        # Use hash of text to generate deterministic but varied embeddings
        hash_val = hash(text) % 1000
        return [0.1 + (hash_val / 10000.0)] * self.embed_dim

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query."""
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async version of get_text_embedding."""
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async version of get_query_embedding."""
        return self._get_query_embedding(query)


@pytest.fixture
def mock_embed_model():
    """Create a mock embedding model that returns fake embeddings."""
    return MockEmbedding(embed_dim=384)


@pytest.fixture
def mock_vector_store():
    """Create a mock SimpleVectorStore."""
    from llama_index.core.vector_stores import SimpleVectorStore
    return SimpleVectorStore()


@pytest.fixture
def mock_vector_manager(mock_vector_store):
    """Create a mock VectorStoreManager."""
    mock = MagicMock()
    mock_index = MagicMock()
    mock.load_or_create.return_value = mock_index
    mock.get_vector_store.return_value = mock_vector_store
    mock.delete_by_dataset.return_value = 0
    mock.persist_vector_store.return_value = None
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
