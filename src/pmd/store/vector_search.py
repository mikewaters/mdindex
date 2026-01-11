"""Vector search adapter for the generic SearchRepository interface."""

from __future__ import annotations

from .embeddings import EmbeddingRepository
from .search import SearchRepository
from ..core.types import SearchResult


class VectorSearchRepository(SearchRepository[list[float]]):
    """Adapter that exposes EmbeddingRepository via the SearchRepository interface."""

    def __init__(self, embedding_repo: EmbeddingRepository):
        """Initialize with an EmbeddingRepository.

        Args:
            embedding_repo: Repository providing vector search operations.
        """
        self.embedding_repo = embedding_repo

    def search(
        self,
        query: list[float],
        limit: int = 5,
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform vector search using the embedding repository."""
        return self.embedding_repo.search_vectors(
            query_embedding=query,
            limit=limit,
            source_collection_id=source_collection_id,
            min_score=min_score,
        )
