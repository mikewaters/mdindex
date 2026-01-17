"""Tests for vector search adapter."""

from unittest.mock import MagicMock

from pmd.store.vector_search import VectorSearchRepository


def test_vector_search_delegates_to_embedding_repo():
    """VectorSearchRepository.search should call EmbeddingRepository.search_vectors."""
    embedding_repo = MagicMock()
    embedding_repo.search_vectors.return_value = ["result"]

    repo = VectorSearchRepository(embedding_repo)

    result = repo.search([0.1, 0.2], limit=7, source_collection_id=3, min_score=0.42)

    embedding_repo.search_vectors.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        limit=7,
        source_collection_id=3,
        min_score=0.42,
    )
    assert result == ["result"]
