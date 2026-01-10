"""Search service for document retrieval operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..core.types import RankedResult, SearchResult
from ..search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from ..search.adapters import (
    FTS5TextSearcher,
    EmbeddingVectorSearcher,
    TagRetrieverAdapter,
    LexicalTagInferencer,
    LLMQueryExpanderAdapter,
    LLMRerankerAdapter,
    OntologyMetadataBooster,
)

if TYPE_CHECKING:
    from .container import ServiceContainer


class SearchService:
    """Service for document search operations.

    This service provides a unified interface for:
    - FTS5 full-text search (BM25)
    - Vector semantic search
    - Hybrid search combining FTS and vector with optional reranking

    Example:

        async with ServiceContainer(config) as services:
            # FTS search
            results = services.search.fts_search("machine learning", limit=10)

            # Vector search
            results = await services.search.vector_search("clustering algorithms")

            # Hybrid search (recommended)
            results = await services.search.hybrid_search(
                "graph database relationships",
                limit=5,
                enable_reranking=True,
            )
    """

    def __init__(self, container: "ServiceContainer"):
        """Initialize SearchService.

        Args:
            container: Service container with shared resources.
        """
        self._container = container

    def fts_search(
        self,
        query: str,
        limit: int = 5,
        collection_name: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Execute FTS5 full-text search.

        Uses BM25 scoring for lexical matching.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            collection_name: Optional collection name to filter.
            min_score: Minimum score threshold.

        Returns:
            List of SearchResult objects.
        """
        collection_id = self._resolve_collection_id(collection_name)

        logger.debug(
            f"FTS search: query={query!r}, limit={limit}, "
            f"collection={collection_name}, min_score={min_score}"
        )

        results = self._container.fts_repo.search(
            query,
            limit=limit,
            collection_id=collection_id,
            min_score=min_score,
        )

        logger.debug(f"FTS search returned {len(results)} results")
        return results

    async def vector_search(
        self,
        query: str,
        limit: int = 5,
        collection_name: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Execute vector semantic search.

        Uses embedding similarity for semantic matching.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            collection_name: Optional collection name to filter.
            min_score: Minimum score threshold.

        Returns:
            List of SearchResult objects.

        Raises:
            RuntimeError: If vector search is not available.
        """
        if not self._container.vec_available:
            raise RuntimeError(
                "Vector search not available (sqlite-vec extension not loaded)"
            )

        collection_id = self._resolve_collection_id(collection_name)

        logger.debug(
            f"Vector search: query={query!r}, limit={limit}, "
            f"collection={collection_name}, min_score={min_score}"
        )

        # Generate query embedding
        embedding_generator = await self._container.get_embedding_generator()
        query_embedding = await embedding_generator.embed_query(query)

        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []

        # Execute vector search
        results = self._container.embedding_repo.search_vectors(
            query_embedding,
            limit=limit,
            collection_id=collection_id,
            min_score=min_score,
        )

        logger.debug(f"Vector search returned {len(results)} results")
        return results

    async def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        collection_name: str | None = None,
        min_score: float = 0.0,
        enable_query_expansion: bool = False,
        enable_reranking: bool = False,
        enable_tag_retrieval: bool = False,
        enable_metadata_boost: bool = False,
    ) -> list[RankedResult]:
        """Execute hybrid search combining FTS and vector search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from FTS5
        and vector search, with optional LLM-based query expansion,
        reranking, tag-based retrieval, and metadata boosting.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            collection_name: Optional collection name to filter.
            min_score: Minimum score threshold.
            enable_query_expansion: Enable LLM query expansion.
            enable_reranking: Enable LLM reranking.
            enable_tag_retrieval: Enable tag-based retrieval in RRF.
            enable_metadata_boost: Enable metadata-based score boosting.

        Returns:
            List of RankedResult objects with combined scores.
        """
        collection_id = self._resolve_collection_id(collection_name)

        logger.debug(
            f"Hybrid search: query={query!r}, limit={limit}, "
            f"collection={collection_name}, expansion={enable_query_expansion}, "
            f"reranking={enable_reranking}, tags={enable_tag_retrieval}, "
            f"boost={enable_metadata_boost}"
        )

        # Create port adapters from container resources
        text_searcher = FTS5TextSearcher(self._container.fts_repo)

        # Vector searcher (requires embedding generator)
        vector_searcher = None
        if self._container.vec_available:
            embedding_generator = await self._container.get_embedding_generator()
            vector_searcher = EmbeddingVectorSearcher(embedding_generator)

        # Query expander adapter
        query_expander = None
        if enable_query_expansion:
            llm_expander = await self._container.get_query_expander()
            query_expander = LLMQueryExpanderAdapter(llm_expander)

        # Reranker adapter
        reranker = None
        if enable_reranking:
            llm_reranker = await self._container.get_reranker()
            reranker = LLMRerankerAdapter(llm_reranker)

        # Tag inferencer (used by both tag retrieval and metadata boost)
        tag_inferencer = None
        tag_searcher = None
        metadata_booster = None

        if enable_tag_retrieval or enable_metadata_boost:
            # Get tag matcher and optionally ontology
            tag_matcher = self._container.get_tag_matcher()
            ontology = self._container.get_ontology()

            if tag_matcher:
                tag_inferencer = LexicalTagInferencer(tag_matcher, ontology)

                # Tag searcher for tag-based retrieval
                if enable_tag_retrieval:
                    tag_retriever = self._container.get_tag_retriever()
                    if tag_retriever:
                        tag_searcher = TagRetrieverAdapter(tag_retriever)

                # Metadata booster for score boosting
                if enable_metadata_boost:
                    metadata_repo = self._container.get_metadata_repo()
                    if metadata_repo:
                        metadata_booster = OntologyMetadataBooster(
                            self._container.db,
                            metadata_repo,
                            ontology,
                        )

        # Configure pipeline
        pipeline_config = SearchPipelineConfig(
            fts_weight=self._container.config.search.fts_weight,
            vec_weight=self._container.config.search.vec_weight,
            rrf_k=self._container.config.search.rrf_k,
            rerank_candidates=self._container.config.search.rerank_candidates,
            enable_query_expansion=enable_query_expansion,
            enable_reranking=enable_reranking,
            enable_tag_retrieval=enable_tag_retrieval,
            enable_metadata_boost=enable_metadata_boost,
        )

        # Create and execute pipeline with port adapters
        pipeline = HybridSearchPipeline(
            text_searcher=text_searcher,
            vector_searcher=vector_searcher,
            tag_searcher=tag_searcher,
            query_expander=query_expander,
            reranker=reranker,
            metadata_booster=metadata_booster,
            tag_inferencer=tag_inferencer,
            config=pipeline_config,
        )

        results = await pipeline.search(
            query,
            limit=limit,
            collection_id=collection_id,
            min_score=min_score,
        )

        logger.debug(f"Hybrid search returned {len(results)} results")
        return results

    def _resolve_collection_id(self, collection_name: str | None) -> int | None:
        """Resolve collection name to ID.

        Args:
            collection_name: Collection name or None.

        Returns:
            Collection ID or None if not specified/found.
        """
        if not collection_name:
            return None

        collection = self._container.collection_repo.get_by_name(collection_name)
        if collection:
            return collection.id

        logger.warning(f"Collection not found: {collection_name}")
        return None
