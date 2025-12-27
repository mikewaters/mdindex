"""Hybrid search pipeline for PMD."""

from dataclasses import dataclass

from ..core.types import RankedResult, SearchResult
from ..store.search import SearchRepository
from .fusion import reciprocal_rank_fusion
from .scoring import blend_scores


@dataclass
class SearchPipelineConfig:
    """Configuration for the hybrid search pipeline."""

    fts_weight: float = 1.0
    vec_weight: float = 1.0
    rrf_k: int = 60
    top_rank_bonus: float = 0.05
    expansion_weight: float = 0.5
    rerank_candidates: int = 30
    enable_query_expansion: bool = False  # Phase 3
    enable_reranking: bool = False  # Phase 3


class HybridSearchPipeline:
    """Orchestrates hybrid search with FTS, vector, and optional reranking."""

    def __init__(
        self,
        search_repo: SearchRepository,
        config: SearchPipelineConfig | None = None,
    ):
        """Initialize the pipeline.

        Args:
            search_repo: SearchRepository instance for queries.
            config: Optional SearchPipelineConfig (uses defaults if None).
        """
        self.search_repo = search_repo
        self.config = config or SearchPipelineConfig()

    def search(
        self,
        query: str,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[RankedResult]:
        """Execute full hybrid search pipeline.

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
            expansions = self._expand_query(query)
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
            reranked = self._rerank(query, candidates)
            final = blend_scores(candidates, reranked)
        else:
            final = candidates

        # Step 5: Filter and limit
        final = [r for r in final if r.score >= min_score]
        return final[:limit]

    def _expand_query(self, query: str) -> list[str]:
        """Generate query variations.

        Phase 3: Will use Ollama to generate semantic variations.

        Args:
            query: Original query string.

        Returns:
            List of query variations.
        """
        # Phase 2: Return empty list (no expansion yet)
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

    def _rerank(
        self,
        query: str,
        candidates: list[RankedResult],
    ) -> list:
        """Rerank candidates using LLM.

        Phase 3: Will use Ollama reranker.

        Args:
            query: Original search query.
            candidates: Candidate results to rerank.

        Returns:
            List of RerankDocumentResult objects.
        """
        # Phase 2: Return empty list (no reranking yet)
        return []
