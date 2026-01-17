"""Search service for document retrieval operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..core.types import RankedResult, SearchResult
from ..search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from ..app.protocols import (
    SourceCollectionRepositoryProtocol,
    DatabaseProtocol,
)

if TYPE_CHECKING:
    from ..app.protocols import (
        TextSearcher,
        VectorSearcher,
        TagSearcher,
        QueryExpander,
        Reranker,
        MetadataBooster,
        TagInferencer,
        FTSRepositoryProtocol,
        EmbeddingGeneratorProtocol,
    )


class SearchService:
    """Service for document search operations.

    This service provides a unified interface for:
    - FTS5 full-text search (BM25)
    - Vector semantic search
    - Hybrid search combining FTS and vector with optional reranking

    The service receives pre-created adapter instances rather than factories,
    simplifying the dependency graph and making the service easier to test.

    Example:
        search = SearchService(
            db=db,
            source_collection_repo=source_collection_repo,
            text_searcher=fts_text_searcher,
            fts_weight=1.0,
            vec_weight=1.0,
        )
        results = await search.hybrid_search("machine learning", limit=10)
    """

    def __init__(
        self,
        db: DatabaseProtocol,
        source_collection_repo: SourceCollectionRepositoryProtocol,
        fts_repo: "FTSRepositoryProtocol",
        # Pre-created adapters
        text_searcher: "TextSearcher",
        vector_searcher: "VectorSearcher | None" = None,
        query_expander: "QueryExpander | None" = None,
        reranker: "Reranker | None" = None,
        tag_inferencer: "TagInferencer | None" = None,
        tag_searcher: "TagSearcher | None" = None,
        metadata_booster: "MetadataBooster | None" = None,
        embedding_generator: "EmbeddingGeneratorProtocol | None" = None,
        # Config
        fts_weight: float = 1.0,
        vec_weight: float = 1.0,
        rrf_k: int = 60,
        rerank_candidates: int = 30,
    ):
        """Initialize SearchService with pre-created adapters.

        Args:
            db: Database for direct SQL operations.
            source_collection_repo: Repository for source collection lookup.
            fts_repo: Repository for FTS search.
            text_searcher: Pre-created text searcher adapter.
            vector_searcher: Pre-created vector searcher adapter (optional).
            query_expander: Pre-created query expander adapter (optional).
            reranker: Pre-created reranker adapter (optional).
            tag_inferencer: Pre-created tag inferencer adapter (optional).
            tag_searcher: Pre-created tag searcher adapter (optional).
            metadata_booster: Pre-created metadata booster adapter (optional).
            embedding_generator: Embedding generator for vector search (optional).
            fts_weight: Weight for FTS results in hybrid search.
            vec_weight: Weight for vector results in hybrid search.
            rrf_k: RRF parameter k.
            rerank_candidates: Number of candidates for reranking.
        """
        self._db = db
        self._source_collection_repo = source_collection_repo
        self._fts_repo = fts_repo

        # Store adapters
        self._text_searcher = text_searcher
        self._vector_searcher = vector_searcher
        self._query_expander = query_expander
        self._reranker = reranker
        self._tag_inferencer = tag_inferencer
        self._tag_searcher = tag_searcher
        self._metadata_booster = metadata_booster
        self._embedding_generator = embedding_generator

        # Config
        self._fts_weight = fts_weight
        self._vec_weight = vec_weight
        self._rrf_k = rrf_k
        self._rerank_candidates = rerank_candidates

    @property
    def vec_available(self) -> bool:
        """Check if vector storage is available."""
        return self._db.vec_available

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

        results = self._fts_repo.search(
            query,
            limit=limit,
            source_collection_id=collection_id,
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
        if not self.vec_available:
            raise RuntimeError(
                "Vector search not available (sqlite-vec extension not loaded)"
            )

        if not self._embedding_generator:
            raise RuntimeError("Embedding generator not configured")

        collection_id = self._resolve_collection_id(collection_name)

        logger.debug(
            f"Vector search: query={query!r}, limit={limit}, "
            f"collection={collection_name}, min_score={min_score}"
        )

        # Generate query embedding
        query_embedding = await self._embedding_generator.embed_query(query)

        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []

        # Execute vector search via the vector searcher adapter
        if not self._vector_searcher:
            logger.warning("Vector searcher not available")
            return []

        results = await self._vector_searcher.search(
            query,
            limit=limit,
            source_collection_id=collection_id,
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

        # Configure pipeline with feature flags
        pipeline_config = SearchPipelineConfig(
            fts_weight=self._fts_weight,
            vec_weight=self._vec_weight,
            rrf_k=self._rrf_k,
            rerank_candidates=self._rerank_candidates,
            enable_query_expansion=enable_query_expansion,
            enable_reranking=enable_reranking,
            enable_tag_retrieval=enable_tag_retrieval,
            enable_metadata_boost=enable_metadata_boost,
        )

        # Create pipeline with pre-created adapters
        # The pipeline will respect the config flags to enable/disable features
        pipeline = HybridSearchPipeline(
            text_searcher=self._text_searcher,
            vector_searcher=self._vector_searcher if self.vec_available else None,
            tag_searcher=self._tag_searcher,
            query_expander=self._query_expander,
            reranker=self._reranker,
            metadata_booster=self._metadata_booster,
            tag_inferencer=self._tag_inferencer,
            config=pipeline_config,
        )

        results = await pipeline.search(
            query,
            limit=limit,
            source_collection_id=collection_id,
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

        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if source_collection:
            return source_collection.id

        logger.warning(f"Source collection not found: {collection_name}")
        return None
