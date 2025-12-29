"""Hybrid search pipeline for PMD.

This module implements the core hybrid search pipeline that combines:
- FTS5 BM25 full-text search
- Vector semantic search via embeddings
- LLM-based query expansion
- LLM-based reranking with position-aware score blending

The pipeline uses Reciprocal Rank Fusion (RRF) to combine results from
multiple retrieval methods, then optionally applies LLM reranking with
position-aware blending via `pmd.search.scoring.blend_scores`.

Typical usage:

    from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig
    from pmd.store.search import FTS5SearchRepository

    fts_repo = FTS5SearchRepository(db)

    config = SearchPipelineConfig(
        enable_query_expansion=True,
        enable_reranking=True,
    )
    pipeline = HybridSearchPipeline(
        fts_repo,
        config=config,
        query_expander=expander,
        reranker=reranker,
        embedding_generator=embedder,  # provides vector search via embedding_repo
    )

    results = await pipeline.search("machine learning clustering", limit=10)

See Also:
    - `pmd.search.scoring`: Score normalization and blending functions
    - `pmd.search.fusion`: Reciprocal Rank Fusion implementation
    - `pmd.llm.reranker`: LLM-based document reranking
    - `pmd.store.search`: FTS5SearchRepository
    - `pmd.store.embeddings`: EmbeddingRepository for vector search
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.types import RankedResult, SearchResult
from ..store.search import FTS5SearchRepository
from .fusion import reciprocal_rank_fusion
from .scoring import blend_scores, normalize_scores

if TYPE_CHECKING:
    from ..llm.embeddings import EmbeddingGenerator
    from ..llm.query_expansion import QueryExpander
    from ..llm.reranker import DocumentReranker


@dataclass
class SearchPipelineConfig:
    """Configuration for the hybrid search pipeline.

    Attributes:
        fts_weight: Weight for FTS5 BM25 results in RRF (default: 1.0).
        vec_weight: Weight for vector search results in RRF (default: 1.0).
        rrf_k: Smoothing constant for RRF formula (default: 60).
        top_rank_bonus: Bonus score for top-ranked results (default: 0.05).
        expansion_weight: Weight for expanded query results (default: 0.5).
        rerank_candidates: Number of candidates to send to reranker (default: 30).
        enable_query_expansion: Enable LLM query expansion (default: False).
        enable_reranking: Enable LLM reranking with position-aware blending (default: False).
        normalize_final_scores: Normalize final scores to 0-1 range (default: True).
    """

    fts_weight: float = 1.0
    vec_weight: float = 1.0
    rrf_k: int = 60
    top_rank_bonus: float = 0.05
    expansion_weight: float = 0.5
    rerank_candidates: int = 30
    enable_query_expansion: bool = False
    enable_reranking: bool = False
    normalize_final_scores: bool = True


class HybridSearchPipeline:
    """Orchestrates hybrid search with FTS, vector, and optional reranking.

    This pipeline combines multiple retrieval strategies using Reciprocal Rank
    Fusion (RRF), with optional LLM-based enhancements:

    1. **Query Expansion**: Generate semantic query variations to improve recall
    2. **Parallel Search**: Run FTS5 and vector search for all query variants
    3. **RRF Fusion**: Combine ranked lists using reciprocal rank fusion
    4. **LLM Reranking**: Score candidates with LLM relevance judgment
    5. **Position-Aware Blending**: Blend RRF and rerank scores by position

    The position-aware blending strategy (from `pmd.search.scoring.blend_scores`):
    - Rank 1-3: 75% RRF + 25% reranker (trust initial retrieval for top results)
    - Rank 4-10: 60% RRF + 40% reranker (balanced weighting)
    - Rank 11+: 40% RRF + 60% reranker (trust reranker for borderline cases)

    Example:
        >>> from pmd.store.search import FTS5SearchRepository
        >>> fts_repo = FTS5SearchRepository(db)
        >>> pipeline = HybridSearchPipeline(fts_repo, config, embedding_generator=embedder)
        >>> results = await pipeline.search("graph database neo4j", limit=5)
        >>> for r in results:
        ...     print(f"{r.title}: {r.score:.3f} (rerank: {r.rerank_score})")

    See Also:
        - `pmd.search.scoring.blend_scores`: Position-aware score blending
        - `pmd.search.scoring.normalize_scores`: Score normalization
        - `pmd.llm.reranker.DocumentReranker`: LLM reranking
        - `pmd.store.search.FTS5SearchRepository`: Full-text search
        - `pmd.store.embeddings.EmbeddingRepository`: Vector similarity search
    """

    def __init__(
        self,
        fts_repo: FTS5SearchRepository,
        config: SearchPipelineConfig | None = None,
        query_expander: "QueryExpander | None" = None,
        reranker: "DocumentReranker | None" = None,
        embedding_generator: "EmbeddingGenerator | None" = None,
    ):
        """Initialize the pipeline.

        Args:
            fts_repo: FTS5SearchRepository for full-text search.
            config: Optional SearchPipelineConfig (uses defaults if None).
            query_expander: Optional QueryExpander for query variations.
            reranker: Optional DocumentReranker for relevance scoring.
            embedding_generator: Optional EmbeddingGenerator for query embeddings.
                                Provides vector search via its embedding_repo.
        """
        self.fts_repo = fts_repo
        self.config = config or SearchPipelineConfig()
        self.query_expander = query_expander
        self.reranker = reranker
        self.embedding_generator = embedding_generator

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
        1. Query expansion - Generate semantic variations (if enabled)
        2. Parallel search - FTS5 and vector search for all query variants
        3. RRF fusion - Combine results using Reciprocal Rank Fusion
        4. LLM reranking - Get relevance scores from LLM (if enabled)
        5. Position-aware blending - Blend RRF and rerank scores
        6. Normalization - Normalize scores to 0-1 range (if enabled)
        7. Filter and limit - Apply score threshold and result limit

        The position-aware blending uses `pmd.search.scoring.blend_scores`
        which applies different weights based on result position:
        - Top 3 results: Trust RRF more (75% RRF, 25% reranker)
        - Rank 4-10: Balanced weighting (60% RRF, 40% reranker)
        - Rank 11+: Trust reranker more (40% RRF, 60% reranker)

        Args:
            query: Search query string.
            limit: Maximum results to return.
            collection_id: Optional collection ID to limit scope.
            min_score: Minimum score threshold for results.

        Returns:
            List of RankedResult objects sorted by relevance.

        See Also:
            - `pmd.search.scoring.blend_scores`: Position-aware blending
            - `pmd.search.scoring.normalize_scores`: Score normalization
        """
        # Step 1: Query expansion
        queries = [query]
        if self.config.enable_query_expansion:
            expansions = await self._expand_query(query)
            queries.extend(expansions)

        # Step 2: Parallel FTS5 and vector search for all query variants
        all_results = await self._parallel_search(queries, limit * 3, collection_id)

        # Step 3: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            all_results,
            k=self.config.rrf_k,
            original_query_weight=2.0,
        )

        # Take top candidates for reranking
        candidates = fused[: self.config.rerank_candidates]

        # Step 4 & 5: LLM Reranking with position-aware blending
        if self.config.enable_reranking and candidates and self.reranker:
            final = await self._rerank_with_blending(query, candidates)
        else:
            final = candidates

        # Step 6: Normalize scores to 0-1 range
        if self.config.normalize_final_scores and final:
            final = normalize_scores(final)

        # Step 7: Filter and limit
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

    async def _parallel_search(
        self,
        queries: list[str],
        limit: int,
        collection_id: int | None,
    ) -> list[list[SearchResult]]:
        """Run FTS5 and vector search in parallel for all queries.

        Uses FTS5SearchRepository for text search and EmbeddingRepository
        for vector similarity search.

        Args:
            queries: List of query strings to search.
            limit: Results per query.
            collection_id: Optional collection ID.

        Returns:
            List of result lists [fts1, vec1, fts2, vec2, ...].
        """
        results = []

        for query in queries:
            # FTS5 search using dedicated repository
            fts_results = self.fts_repo.search(
                query,
                limit,
                collection_id,
            )
            results.append(fts_results)

            # Vector search with query embedding via EmbeddingRepository
            vec_results: list[SearchResult] = []
            if self.embedding_generator:
                try:
                    query_embedding = await self.embedding_generator.embed_query(query)
                    if query_embedding:
                        vec_results = self.embedding_generator.embedding_repo.search_vectors(
                            query_embedding,
                            limit,
                            collection_id,
                        )
                except Exception:
                    pass  # Fall back to empty results on error

            results.append(vec_results)

        return results

    async def _rerank_with_blending(
        self,
        query: str,
        candidates: list[RankedResult],
    ) -> list[RankedResult]:
        """Rerank candidates using LLM with position-aware score blending.

        This method:
        1. Gets raw relevance scores from the LLM via the reranker
        2. Applies position-aware blending using `pmd.search.scoring.blend_scores`

        The blending strategy trusts top-ranked results from initial retrieval
        more, while relying on the reranker to distinguish borderline cases:
        - Rank 1-3: 75% RRF + 25% reranker
        - Rank 4-10: 60% RRF + 40% reranker
        - Rank 11+: 40% RRF + 60% reranker

        Args:
            query: Original search query.
            candidates: Candidate results to rerank (with RRF scores).

        Returns:
            Reranked results with blended scores.

        See Also:
            - `pmd.search.scoring.blend_scores`: Position-aware blending logic
            - `pmd.llm.reranker.DocumentReranker.get_rerank_scores`: Raw LLM scores
        """
        if not self.reranker:
            return candidates

        try:
            # Get raw rerank scores from LLM
            rerank_results = await self.reranker.get_rerank_scores(query, candidates)

            # Apply position-aware blending
            return blend_scores(candidates, rerank_results)
        except Exception:
            return candidates
