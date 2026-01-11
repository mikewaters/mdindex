"""Search service for document retrieval operations."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Awaitable

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
from ..app.types import (
    SourceCollectionRepositoryProtocol,
    DatabaseProtocol,
    DocumentMetadataRepositoryProtocol,
    EmbeddingGeneratorProtocol,
    EmbeddingRepositoryProtocol,
    FTSRepositoryProtocol,
    OntologyProtocol,
    QueryExpanderProtocol,
    DocumentRerankerProtocol,
    TagMatcherProtocol,
    TagRetrieverProtocol,
)

if TYPE_CHECKING:
    from .container import ServiceContainer


class SearchService:
    """Service for document search operations.

    This service provides a unified interface for:
    - FTS5 full-text search (BM25)
    - Vector semantic search
    - Hybrid search combining FTS and vector with optional reranking

    Example with explicit dependencies (recommended):

        search = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            search_config=SearchConfig(fts_weight=1.0, vec_weight=1.0),
        )
        results = search.fts_search("machine learning", limit=10)

    Example with ServiceContainer (deprecated):

        async with ServiceContainer(config) as services:
            results = await services.search.hybrid_search("query")
    """

    def __init__(
        self,
        # Explicit dependencies (new API)
        db: DatabaseProtocol | None = None,
        fts_repo: FTSRepositoryProtocol | None = None,
        source_collection_repo: SourceCollectionRepositoryProtocol | None = None,
        embedding_repo: EmbeddingRepositoryProtocol | None = None,
        embedding_generator_factory: Callable[[], Awaitable[EmbeddingGeneratorProtocol]] | None = None,
        query_expander_factory: Callable[[], Awaitable[QueryExpanderProtocol]] | None = None,
        reranker_factory: Callable[[], Awaitable[DocumentRerankerProtocol]] | None = None,
        tag_matcher_factory: Callable[[], TagMatcherProtocol | None] | None = None,
        ontology_factory: Callable[[], OntologyProtocol | None] | None = None,
        tag_retriever_factory: Callable[[], TagRetrieverProtocol | None] | None = None,
        metadata_repo_factory: Callable[[], DocumentMetadataRepositoryProtocol | None] | None = None,
        fts_weight: float = 1.0,
        vec_weight: float = 1.0,
        rrf_k: int = 60,
        rerank_candidates: int = 30,
        # Deprecated: ServiceContainer
        container: "ServiceContainer | None" = None,
    ):
        """Initialize SearchService.

        Args:
            db: Database for direct SQL operations.
            fts_repo: Repository for FTS search.
            source_collection_repo: Repository for source collection lookup.
            embedding_repo: Repository for embedding storage and vector search.
            embedding_generator_factory: Async factory for embedding generator.
            query_expander_factory: Async factory for query expander.
            reranker_factory: Async factory for document reranker.
            tag_matcher_factory: Factory for tag matcher.
            ontology_factory: Factory for ontology.
            tag_retriever_factory: Factory for tag retriever.
            metadata_repo_factory: Factory for metadata repository.
            fts_weight: Weight for FTS results in hybrid search.
            vec_weight: Weight for vector results in hybrid search.
            rrf_k: RRF parameter k.
            rerank_candidates: Number of candidates for reranking.
            container: DEPRECATED. Use explicit dependencies instead.
        """
        # Support backward compatibility with container
        if container is not None:
            warnings.warn(
                "Passing 'container' to SearchService is deprecated. "
                "Use explicit dependencies instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._container = container
            self._db = container.db
            self._fts_repo = container.fts_repo
            self._source_collection_repo = container.source_collection_repo
            self._embedding_repo = container.embedding_repo
            self._embedding_generator_factory = container.get_embedding_generator
            self._query_expander_factory = container.get_query_expander
            self._reranker_factory = container.get_reranker
            self._tag_matcher_factory = container.get_tag_matcher
            self._ontology_factory = container.get_ontology
            self._tag_retriever_factory = container.get_tag_retriever
            self._metadata_repo_factory = container.get_metadata_repo
            self._fts_weight = container.config.search.fts_weight
            self._vec_weight = container.config.search.vec_weight
            self._rrf_k = container.config.search.rrf_k
            self._rerank_candidates = container.config.search.rerank_candidates
        else:
            self._container = None
            if db is None or fts_repo is None or source_collection_repo is None:
                raise ValueError(
                    "SearchService requires db, fts_repo, and source_collection_repo"
                )
            self._db = db
            self._fts_repo = fts_repo
            self._source_collection_repo = source_collection_repo
            self._embedding_repo = embedding_repo
            self._embedding_generator_factory = embedding_generator_factory
            self._query_expander_factory = query_expander_factory
            self._reranker_factory = reranker_factory
            self._tag_matcher_factory = tag_matcher_factory
            self._ontology_factory = ontology_factory
            self._tag_retriever_factory = tag_retriever_factory
            self._metadata_repo_factory = metadata_repo_factory
            self._fts_weight = fts_weight
            self._vec_weight = vec_weight
            self._rrf_k = rrf_k
            self._rerank_candidates = rerank_candidates

    @classmethod
    def from_container(cls, container: "ServiceContainer") -> "SearchService":
        """Create SearchService from a ServiceContainer.

        This is a convenience method for backward compatibility.
        Prefer using explicit dependencies in new code.

        Args:
            container: Service container with shared resources.

        Returns:
            SearchService instance.
        """
        return cls(
            db=container.db,
            fts_repo=container.fts_repo,
            source_collection_repo=container.source_collection_repo,
            embedding_repo=container.embedding_repo,
            embedding_generator_factory=container.get_embedding_generator,
            query_expander_factory=container.get_query_expander,  # type: ignore
            reranker_factory=container.get_reranker, # type: ignore
            tag_matcher_factory=container.get_tag_matcher, # type: ignore
            ontology_factory=container.get_ontology, # type: ignore
            tag_retriever_factory=container.get_tag_retriever, # type: ignore
            metadata_repo_factory=container.get_metadata_repo, # type: ignore
            fts_weight=container.config.search.fts_weight,
            vec_weight=container.config.search.vec_weight,
            rrf_k=container.config.search.rrf_k,
            rerank_candidates=container.config.search.rerank_candidates,
        )

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
        if not self.vec_available:
            raise RuntimeError(
                "Vector search not available (sqlite-vec extension not loaded)"
            )

        if not self._embedding_generator_factory:
            raise RuntimeError("Embedding generator not configured")

        collection_id = self._resolve_collection_id(collection_name)

        logger.debug(
            f"Vector search: query={query!r}, limit={limit}, "
            f"collection={collection_name}, min_score={min_score}"
        )

        # Generate query embedding
        embedding_generator = await self._embedding_generator_factory()
        query_embedding = await embedding_generator.embed_query(query)

        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []

        # Execute vector search
        if not self._embedding_repo:
            logger.warning("Vector search requires embedding_repo")
            return []

        results = self._embedding_repo.search_vectors(
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

        # Create port adapters from explicit dependencies
        text_searcher = FTS5TextSearcher(self._fts_repo) # type: ignore

        # Vector searcher (requires embedding generator)
        vector_searcher = None
        if self.vec_available and self._embedding_generator_factory:
            embedding_generator = await self._embedding_generator_factory()
            vector_searcher = EmbeddingVectorSearcher(embedding_generator) # type: ignore

        # Query expander adapter
        query_expander = None
        if enable_query_expansion and self._query_expander_factory:
            llm_expander = await self._query_expander_factory()
            query_expander = LLMQueryExpanderAdapter(llm_expander) # type: ignore

        # Reranker adapter
        reranker = None
        if enable_reranking and self._reranker_factory:
            llm_reranker = await self._reranker_factory()
            reranker = LLMRerankerAdapter(llm_reranker) # type: ignore

        # Tag inferencer (used by both tag retrieval and metadata boost)
        tag_inferencer = None
        tag_searcher = None
        metadata_booster = None

        if enable_tag_retrieval or enable_metadata_boost:
            # Get tag matcher and optionally ontology
            tag_matcher = self._tag_matcher_factory() if self._tag_matcher_factory else None
            ontology = self._ontology_factory() if self._ontology_factory else None

            if tag_matcher:
                tag_inferencer = LexicalTagInferencer(tag_matcher, ontology) # type: ignore

                # Tag searcher for tag-based retrieval
                if enable_tag_retrieval and self._tag_retriever_factory:
                    tag_retriever = self._tag_retriever_factory()
                    if tag_retriever:
                        tag_searcher = TagRetrieverAdapter(tag_retriever) # type: ignore

                # Metadata booster for score boosting
                if enable_metadata_boost and self._metadata_repo_factory:
                    metadata_repo = self._metadata_repo_factory()
                    if metadata_repo:
                        metadata_booster = OntologyMetadataBooster(
                            self._db, # type: ignore
                            metadata_repo, # type: ignore
                            ontology, # type: ignore
                        )

        # Configure pipeline
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

        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if source_collection:
            return source_collection.id

        logger.warning(f"Source collection not found: {collection_name}")
        return None
