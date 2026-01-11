"""Embedding-based vector search adapter.

Wraps EmbeddingGenerator to implement the VectorSearcher protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pmd.core.types import SearchResult
    from pmd.llm.embeddings import EmbeddingGenerator


class EmbeddingVectorSearcher:
    """Adapter that wraps EmbeddingGenerator for the VectorSearcher protocol.

    This adapter handles query embedding and vector search in a single operation,
    hiding the two-step process (embed query, then search) from the pipeline.

    Example:
        >>> from pmd.llm.embeddings import EmbeddingGenerator
        >>> embedding_gen = EmbeddingGenerator(llm, embedding_repo, config)
        >>> searcher = EmbeddingVectorSearcher(embedding_gen)
        >>> results = await searcher.search("machine learning", limit=10)
    """

    def __init__(self, embedding_generator: "EmbeddingGenerator"):
        """Initialize with embedding generator.

        Args:
            embedding_generator: EmbeddingGenerator instance that provides
                both query embedding and access to the embedding repository.
        """
        self._generator = embedding_generator

    async def search(
        self,
        query: str,
        limit: int,
        source_collection_id: int | None = None,
    ) -> list["SearchResult"]:
        """Search documents using vector similarity.

        Embeds the query and performs vector similarity search.

        Args:
            query: Search query string (will be embedded internally).
            limit: Maximum number of results to return.
            source_collection_id: Optional collection to scope search.

        Returns:
            List of SearchResult objects sorted by similarity score.
            Returns empty list if embedding fails.
        """
        # Embed the query
        query_embedding = await self._generator.embed_query(query)

        if not query_embedding:
            logger.warning("Vector search skipped: query embedding failed")
            return []

        # Perform vector search
        return self._generator.embedding_repo.search_vectors(
            query_embedding,
            limit=limit,
            source_collection_id=source_collection_id,
        )
