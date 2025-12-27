"""Hybrid search pipeline for PMD."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.types import RankedResult, SearchResult
from ..store.search import SearchRepository
from .fusion import reciprocal_rank_fusion
from .scoring import blend_scores

if TYPE_CHECKING:
    from ..llm.query_expansion import QueryExpander
    from ..llm.reranker import DocumentReranker


@dataclass
class SearchPipelineConfig:
    """Configuration for the hybrid search pipeline."""

    fts_weight: float = 1.0
    vec_weight: float = 1.0
    rrf_k: int = 60
    top_rank_bonus: float = 0.05
    expansion_weight: float = 0.5
    rerank_candidates: int = 30
    enable_query_expansion: bool = False
    enable_reranking: bool = False


class HybridSearchPipeline:
    """Orchestrates hybrid search with FTS, vector, and optional reranking."""

    def __init__(
        self,
        search_repo: SearchRepository,
        config: SearchPipelineConfig | None = None,
        query_expander: "QueryExpander | None" = None,
        reranker: "DocumentReranker | None" = None,
    ):
        """Initialize the pipeline.

        Args:
            search_repo: SearchRepository instance for queries.
            config: Optional SearchPipelineConfig (uses defaults if None).
            query_expander: Optional QueryExpander for query variations.
            reranker: Optional DocumentReranker for relevance scoring.
        """
        self.search_repo = search_repo
        self.config = config or SearchPipelineConfig()
        self.query_expander = query_expander
        self.reranker = reranker

        # Ensure expander and reranker are only used if enabled
        if not self.config.enable_query_expansion:
            self.query_expander = None
        if not self.config.enable_reranking:
            self.reranker = None

    async def search(
        self,
        query: str,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[RankedResult]:
        """Execute full hybrid search pipeline asynchronously.

        Pipeline steps:
        1. Query expansion (Phase 3)
        2. Parallel FTS5 and vector search for all query variants
        3. Reciprocal Rank Fusion
        4. LLM Reranking (Phase 3)
        5. Position-aware score blending
        6. Filter and limit results

        Args:
            query: Search query string.
            limit: Maximum results to return.
            collection_id: Optional collection ID to limit scope.
            min_score: Minimum score threshold for results.

        Returns:
            List of RankedResult objects sorted by relevance.
        """
        # Step 1: Query expansion (Phase 3)
        queries = [query]
        if self.config.enable_query_expansion:
            expansions = await self._expand_query(query)
            queries.extend(expansions)

        # Step 2: Parallel FTS5 and vector search for all query variants
        all_results = self._parallel_search(queries, limit * 3, collection_id)

        # Step 3: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            all_results,
            k=self.config.rrf_k,
            original_query_weight=2.0,
        )

        # Take top candidates for reranking
        candidates = fused[: self.config.rerank_candidates]

        # Step 4: LLM Reranking (Phase 3)
        if self.config.enable_reranking and candidates:
            reranked = await self._rerank(query, candidates)
            final = reranked
        else:
            final = candidates

        # Step 5: Filter and limit
        final = [r for r in final if r.score >= min_score]
        return final[:limit]

    async def _expand_query(self, query: str) -> list[str]:
        """Generate query variations using LLM.

        Args:
            query: Original query string.

        Returns:
            List of query variations [original, var1, var2, ...].
        """
        if not self.query_expander:
            return []

        try:
            variations = await self.query_expander.expand(query, num_variations=2)
            return variations
        except Exception:
            return []

    def _parallel_search(
        self,
        queries: list[str],
        limit: int,
        collection_id: int | None,
    ) -> list[list[SearchResult]]:
        """Run FTS5 and vector search in parallel for all queries.

        Args:
            queries: List of query strings to search.
            limit: Results per query.
            collection_id: Optional collection ID.

        Returns:
            List of result lists [fts1, vec1, fts2, vec2, ...].
        """
        results = []

        for query in queries:
            # FTS5 search
            fts_results = self.search_repo.search_fts(
                query,
                limit,
                collection_id,
            )
            results.append(fts_results)

            # Vector search (placeholder in Phase 2)
            vec_results = self.search_repo.search_vec(
                [],  # Empty embedding in Phase 2
                limit,
                collection_id,
            )
            results.append(vec_results)

        return results

    async def _rerank(
        self,
        query: str,
        candidates: list[RankedResult],
    ) -> list[RankedResult]:
        """Rerank candidates using LLM.

        Args:
            query: Original search query.
            candidates: Candidate results to rerank.

        Returns:
            Reranked results.
        """
        if not self.reranker:
            return candidates

        try:
            return await self.reranker.rerank(query, candidates)
        except Exception:
            return candidates
