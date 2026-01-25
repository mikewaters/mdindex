"""idx.search.service - Unified search service.

Provides a unified entry point for all search modes (FTS, vector, hybrid)
with optional LLM-as-judge reranking. Dispatches to appropriate search
implementations based on SearchCriteria.mode.

Example usage:
    from idx.search.service import SearchService
    from idx.search.models import SearchCriteria
    from idx.store.database import get_session
    from idx.store.session_context import use_session

    with get_session() as session:
        with use_session(session):
            service = SearchService()

            # FTS search
            results = service.search(SearchCriteria(
                query="python async patterns",
                mode="fts",
                limit=10,
            ))

            # Hybrid search with reranking
            results = service.search(SearchCriteria(
                query="machine learning",
                mode="hybrid",
                rerank=True,
                rerank_candidates=20,
                limit=5,
            ))
"""

from typing import TYPE_CHECKING, Any

from idx.core.logging import get_logger
from idx.search.models import SearchCriteria, SearchResult, SearchResults

if TYPE_CHECKING:
    from idx.search.fts import FTSSearch
    from idx.search.hybrid import HybridSearch
    from idx.search.rerank import Reranker
    from idx.search.vector import VectorSearch

__all__ = ["SearchService"]

logger = get_logger(__name__)


class SearchService:
    """Unified search service orchestrating FTS, vector, and hybrid search.

    Provides a single entry point for all search operations, dispatching
    to the appropriate implementation based on SearchCriteria.mode.
    Supports optional LLM-as-judge reranking when criteria.rerank=True.

    All retrievers and the reranker are lazy-loaded on first use to
    minimize startup time and resource consumption.

    Attributes:
        _fts_search: Lazy-loaded FTSSearch instance.
        _vector_search: Lazy-loaded VectorSearch instance.
        _hybrid_search: Lazy-loaded HybridSearch instance.
        _reranker: Lazy-loaded Reranker instance.
        _debug: Whether to include debug information in results.

    Example:
        service = SearchService()

        # Basic FTS search
        results = service.search(SearchCriteria(
            query="python tutorials",
            mode="fts",
        ))

        # Vector search with dataset filter
        results = service.search(SearchCriteria(
            query="meeting notes about Q4",
            mode="vector",
            dataset_name="obsidian",
            limit=20,
        ))

        # Hybrid search with reranking
        results = service.search(SearchCriteria(
            query="async error handling patterns",
            mode="hybrid",
            rerank=True,
            rerank_candidates=30,
            limit=10,
        ))
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the SearchService.

        Args:
            debug: Whether to include debug information in SearchResults.
                When True, debug_info will contain timing breakdowns and
                intermediate scores.
        """
        self._debug = debug
        self._fts_search: "FTSSearch | None" = None
        self._vector_search: "VectorSearch | None" = None
        self._hybrid_search: "HybridSearch | None" = None
        self._reranker: "Reranker | None" = None

    def _ensure_fts_search(self) -> "FTSSearch":
        """Lazy-load the FTSSearch instance.

        Returns:
            Initialized FTSSearch instance.
        """
        if self._fts_search is None:
            from idx.search.fts import FTSSearch

            logger.debug("Lazy-loading FTSSearch")
            self._fts_search = FTSSearch()
        return self._fts_search

    def _ensure_vector_search(self) -> "VectorSearch":
        """Lazy-load the VectorSearch instance.

        Returns:
            Initialized VectorSearch instance.
        """
        if self._vector_search is None:
            from idx.search.vector import VectorSearch

            logger.debug("Lazy-loading VectorSearch")
            self._vector_search = VectorSearch()
        return self._vector_search

    def _ensure_hybrid_search(self) -> "HybridSearch":
        """Lazy-load the HybridSearch instance.

        Returns:
            Initialized HybridSearch instance.
        """
        if self._hybrid_search is None:
            from idx.search.hybrid import HybridSearch

            logger.debug("Lazy-loading HybridSearch")
            self._hybrid_search = HybridSearch()
        return self._hybrid_search

    def _ensure_reranker(self) -> "Reranker":
        """Lazy-load the Reranker instance.

        Returns:
            Initialized Reranker instance.
        """
        if self._reranker is None:
            from idx.search.rerank import Reranker

            logger.debug("Lazy-loading Reranker")
            self._reranker = Reranker()
        return self._reranker

    def search(self, criteria: SearchCriteria) -> SearchResults:
        """Execute a search based on the provided criteria.

        Dispatches to the appropriate search implementation based on
        criteria.mode, optionally applying LLM-as-judge reranking.

        When reranking is enabled (criteria.rerank=True):
        1. Retrieves criteria.rerank_candidates results from the primary search
        2. Converts results to LlamaIndex NodeWithScore format
        3. Applies LLMRerank postprocessor
        4. Converts back to SearchResult format, limited to criteria.limit

        Args:
            criteria: Search criteria specifying query, mode, filters,
                and reranking options.

        Returns:
            SearchResults containing matching documents ordered by score,
            with optional debug_info when debug=True.

        Raises:
            ValueError: If criteria.mode is not one of "fts", "vector", "hybrid".
        """
        import time

        start = time.perf_counter()
        debug_info: dict[str, Any] = {} if self._debug else {}

        # Determine effective limit (use more candidates if reranking)
        effective_limit = (
            criteria.rerank_candidates if criteria.rerank else criteria.limit
        )

        # Dispatch based on mode
        if criteria.mode == "fts":
            results = self._search_fts(criteria, effective_limit)
        elif criteria.mode == "vector":
            results = self._search_vector(criteria, effective_limit)
        elif criteria.mode == "hybrid":
            results = self._search_hybrid(criteria, effective_limit)
        else:
            raise ValueError(f"Unsupported search mode: {criteria.mode}")

        search_elapsed_ms = (time.perf_counter() - start) * 1000

        if self._debug:
            debug_info["search_mode"] = criteria.mode
            debug_info["search_time_ms"] = search_elapsed_ms
            debug_info["candidates_retrieved"] = len(results)

        # Apply reranking if requested
        if criteria.rerank and results:
            rerank_start = time.perf_counter()
            results = self._apply_rerank(results, criteria.query, criteria.limit)
            rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000

            if self._debug:
                debug_info["rerank_time_ms"] = rerank_elapsed_ms
                debug_info["results_after_rerank"] = len(results)
        else:
            # Apply limit without reranking
            results = results[: criteria.limit]

        total_elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"SearchService.search({criteria.mode}) for '{criteria.query[:50]}...' "
            f"returned {len(results)} results in {total_elapsed_ms:.1f}ms"
        )

        return SearchResults(
            results=results,
            query=criteria.query,
            mode=criteria.mode,
            total_candidates=len(results),
            timing_ms=total_elapsed_ms,
        )

    def _search_fts(
        self, criteria: SearchCriteria, limit: int
    ) -> list[SearchResult]:
        """Execute FTS search.

        Args:
            criteria: Search criteria.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects from FTS search.
        """
        fts = self._ensure_fts_search()

        # Create modified criteria with effective limit
        fts_criteria = SearchCriteria(
            query=criteria.query,
            mode="fts",
            dataset_name=criteria.dataset_name,
            limit=limit,
            rerank=False,
        )

        fts_results = fts.search(fts_criteria)
        return fts_results.results

    def _search_vector(
        self, criteria: SearchCriteria, limit: int
    ) -> list[SearchResult]:
        """Execute vector search.

        Args:
            criteria: Search criteria.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects from vector search.
        """
        vector = self._ensure_vector_search()

        results = vector.search(
            query=criteria.query,
            top_k=limit,
            dataset_name=criteria.dataset_name,
        )
        return results

    def _search_hybrid(
        self, criteria: SearchCriteria, limit: int
    ) -> list[SearchResult]:
        """Execute hybrid search (RRF fusion of FTS and vector).

        Args:
            criteria: Search criteria.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects from hybrid search.
        """
        hybrid = self._ensure_hybrid_search()

        results = hybrid.search(
            query=criteria.query,
            top_k=limit,
            dataset_name=criteria.dataset_name,
        )
        return results

    def _apply_rerank(
        self,
        results: list[SearchResult],
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Apply LLM-as-judge reranking to search results.

        Converts SearchResult objects to LlamaIndex NodeWithScore format,
        applies reranking, and converts back to SearchResult format.

        Args:
            results: List of SearchResult objects to rerank.
            query: The search query for relevance evaluation.
            limit: Maximum number of results to return after reranking.

        Returns:
            Reranked list of SearchResult objects, limited to `limit`.
        """
        from llama_index.core.schema import NodeWithScore, TextNode

        reranker = self._ensure_reranker()

        # Convert SearchResult to NodeWithScore
        nodes: list[NodeWithScore] = []
        result_by_node_id: dict[str, SearchResult] = {}

        for i, result in enumerate(results):
            # Create unique node ID for mapping back
            node_id = f"result_{i}"

            # Build metadata for the node
            metadata = {
                "source_doc_id": f"{result.dataset_name}:{result.path}",
                **(result.metadata or {}),
            }
            if result.chunk_seq is not None:
                metadata["chunk_seq"] = result.chunk_seq
            if result.chunk_pos is not None:
                metadata["chunk_pos"] = result.chunk_pos

            node = TextNode(
                id_=node_id,
                text=result.chunk_text or "",
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=node, score=result.score))
            result_by_node_id[node_id] = result

        logger.debug(f"Reranking {len(nodes)} candidates for query '{query[:50]}...'")

        # Apply reranking
        reranked_nodes = reranker.rerank(nodes=nodes, query=query, top_n=limit)

        # Convert back to SearchResult, preserving rerank scores
        reranked_results: list[SearchResult] = []
        for node_with_score in reranked_nodes:
            node_id = node_with_score.node.id_
            original_result = result_by_node_id.get(node_id)

            if original_result is None:
                logger.warning(f"Could not find original result for node {node_id}")
                continue

            # Update scores to include rerank score
            new_scores = {**original_result.scores, "rerank": node_with_score.score or 0.0}

            reranked_results.append(
                SearchResult(
                    path=original_result.path,
                    dataset_name=original_result.dataset_name,
                    score=node_with_score.score or 0.0,  # Use rerank score as primary
                    chunk_text=original_result.chunk_text,
                    chunk_seq=original_result.chunk_seq,
                    chunk_pos=original_result.chunk_pos,
                    metadata=original_result.metadata,
                    scores=new_scores,
                )
            )

        logger.debug(f"Reranking complete: {len(results)} -> {len(reranked_results)} results")

        return reranked_results
