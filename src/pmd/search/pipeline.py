"""Hybrid search pipeline for PMD.

This module implements the core hybrid search pipeline that combines:
- FTS5 BM25 full-text search
- Vector semantic search via embeddings
- Tag-based document retrieval
- LLM-based query expansion
- LLM-based reranking with position-aware score blending
- Metadata-based score boosting

The pipeline uses Reciprocal Rank Fusion (RRF) to combine results from
multiple retrieval methods, then optionally applies LLM reranking with
position-aware blending via `pmd.search.scoring.blend_scores`.

The pipeline depends on abstract ports (defined in `pmd.search.ports`) rather
than concrete implementations, making it testable with in-memory fakes.

Typical usage:

    from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig
    from pmd.search.adapters import FTS5TextSearcher, EmbeddingVectorSearcher

    config = SearchPipelineConfig(
        enable_query_expansion=True,
        enable_reranking=True,
    )
    pipeline = HybridSearchPipeline(
        text_searcher=FTS5TextSearcher(fts_repo),
        vector_searcher=EmbeddingVectorSearcher(embedding_gen),
        query_expander=LLMQueryExpanderAdapter(expander),
        reranker=LLMRerankerAdapter(reranker),
        config=config,
    )

    results = await pipeline.search("machine learning clustering", limit=10)

See Also:
    - `pmd.search.ports`: Port protocol definitions
    - `pmd.search.adapters`: Adapter implementations
    - `pmd.search.scoring`: Score normalization and blending functions
    - `pmd.search.fusion`: Reciprocal Rank Fusion implementation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from pmd.core.types import RankedResult, SearchResult
from pmd.search.fusion import reciprocal_rank_fusion
from pmd.search.scoring import blend_scores, normalize_scores

if TYPE_CHECKING:
    from pmd.search.ports import (
        TextSearcher,
        VectorSearcher,
        TagSearcher,
        QueryExpander,
        Reranker,
        MetadataBooster,
        TagInferencer,
    )


@dataclass
class SearchPipelineConfig:
    """Configuration for the hybrid search pipeline.

    Weights:
        fts_weight: Weight for FTS5 BM25 results in RRF.
            Higher values give more influence to lexical matches.
            Default: 1.0
        vec_weight: Weight for vector search results in RRF.
            Higher values give more influence to semantic similarity.
            Default: 1.0
        tag_weight: Weight for tag-based results in RRF.
            Typically lower than FTS/vec since tags are sparse signals.
            Default: 0.8
        rrf_k: Smoothing constant for RRF formula.
            Higher values produce more uniform score distributions.
            Lower values emphasize top-ranked documents more strongly.
            Default: 60

    Scoring:
        top_rank_bonus: Bonus score added to top-ranked results.
            Default: 0.05
        expansion_weight: Weight multiplier for expanded query results.
            Applied to FTS/vec/tag weights for non-original queries.
            Default: 0.5

    Reranking:
        rerank_candidates: Number of candidates to send to reranker.
            More candidates = better recall but slower and more expensive.
            Default: 30

    Feature Flags:
        enable_query_expansion: Generate semantic query variations via LLM.
            Default: False
        enable_reranking: Apply LLM-based relevance reranking.
            Default: False
        enable_tag_retrieval: Include tag-based results in RRF fusion.
            Default: False
        enable_metadata_boost: Boost scores based on tag matches.
            Default: False

    Metadata Boosting:
        metadata_boost_factor: Base boost multiplier for tag matches.
            Applied exponentially: boost = factor ** total_match_weight.
            Default: 1.15
        metadata_max_boost: Maximum allowed boost multiplier.
            Caps the total boost to prevent runaway scores.
            Default: 2.0

    Output:
        normalize_final_scores: Normalize final scores to 0-1 range.
            Default: True
    """

    fts_weight: float = 1.0
    vec_weight: float = 1.0
    tag_weight: float = 0.8
    rrf_k: int = 60
    top_rank_bonus: float = 0.05
    expansion_weight: float = 0.5
    rerank_candidates: int = 30
    enable_query_expansion: bool = False
    enable_reranking: bool = False
    enable_tag_retrieval: bool = False
    enable_metadata_boost: bool = False
    metadata_boost_factor: float = 1.15
    metadata_max_boost: float = 2.0
    normalize_final_scores: bool = True


class HybridSearchPipeline:
    """Orchestrates hybrid search with FTS, vector, and optional reranking.

    This pipeline combines multiple retrieval strategies using Reciprocal Rank
    Fusion (RRF), with optional LLM-based and metadata-based enhancements:

    1. **Query Expansion**: Generate semantic query variations to improve recall
    2. **Parallel Search**: Run FTS5, vector, and tag search for all query variants
    3. **RRF Fusion**: Combine ranked lists using reciprocal rank fusion
    4. **Metadata Boost**: Boost documents with matching tags (optional)
    5. **LLM Reranking**: Score candidates with LLM relevance judgment
    6. **Position-Aware Blending**: Blend RRF and rerank scores by position

    The pipeline depends on abstract ports rather than concrete implementations:
    - TextSearcher: Full-text search (FTS5, BM25)
    - VectorSearcher: Vector similarity search (embeddings)
    - TagSearcher: Tag-based document retrieval
    - QueryExpander: Query expansion via LLM
    - Reranker: Document reranking via LLM
    - MetadataBooster: Score boosting based on tag matches
    - TagInferencer: Tag inference from query text

    This design allows the pipeline to be tested with in-memory fakes without
    requiring database or LLM infrastructure.

    The position-aware blending strategy (from `pmd.search.scoring.blend_scores`):
    - Rank 1-3: 75% RRF + 25% reranker (trust initial retrieval for top results)
    - Rank 4-10: 60% RRF + 40% reranker (balanced weighting)
    - Rank 11+: 40% RRF + 60% reranker (trust reranker for borderline cases)

    Example:
        >>> from pmd.search.adapters import FTS5TextSearcher
        >>> text_searcher = FTS5TextSearcher(fts_repo)
        >>> pipeline = HybridSearchPipeline(text_searcher=text_searcher)
        >>> results = await pipeline.search("graph database neo4j", limit=5)

    See Also:
        - `pmd.search.ports`: Port protocol definitions
        - `pmd.search.adapters`: Adapter implementations
        - `pmd.search.scoring.blend_scores`: Position-aware score blending
    """

    def __init__(
        self,
        text_searcher: "TextSearcher",
        vector_searcher: "VectorSearcher | None" = None,
        tag_searcher: "TagSearcher | None" = None,
        query_expander: "QueryExpander | None" = None,
        reranker: "Reranker | None" = None,
        metadata_booster: "MetadataBooster | None" = None,
        tag_inferencer: "TagInferencer | None" = None,
        config: SearchPipelineConfig | None = None,
    ):
        """Initialize the pipeline with port implementations.

        Args:
            text_searcher: TextSearcher implementation for full-text search.
                Required - the pipeline always performs FTS search.
            vector_searcher: Optional VectorSearcher for semantic search.
                When provided, vector results are included in RRF fusion.
            tag_searcher: Optional TagSearcher for tag-based retrieval.
                Used when enable_tag_retrieval is True.
            query_expander: Optional QueryExpander for query variations.
                Used when enable_query_expansion is True.
            reranker: Optional Reranker for LLM-based relevance scoring.
                Used when enable_reranking is True.
            metadata_booster: Optional MetadataBooster for tag-based score boosting.
                Used when enable_metadata_boost is True.
            tag_inferencer: Optional TagInferencer for query tag inference.
                Used by both tag_searcher and metadata_booster.
            config: Pipeline configuration (uses defaults if None).
        """
        self.text_searcher = text_searcher
        self.vector_searcher = vector_searcher
        self.tag_searcher = tag_searcher
        self.query_expander = query_expander
        self.reranker = reranker
        self.metadata_booster = metadata_booster
        self.tag_inferencer = tag_inferencer
        self.config = config or SearchPipelineConfig()

        # Clear components that are disabled by config
        if not self.config.enable_query_expansion:
            self.query_expander = None
        if not self.config.enable_reranking:
            self.reranker = None
        if not self.config.enable_tag_retrieval:
            self.tag_searcher = None
        if not self.config.enable_metadata_boost:
            self.metadata_booster = None

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
        2. Parallel search - FTS5, vector, and tag search for all query variants
        3. RRF fusion - Combine results using Reciprocal Rank Fusion
        4. Metadata boost - Boost documents with matching tags (if enabled)
        5. LLM reranking - Get relevance scores from LLM (if enabled)
        6. Position-aware blending - Blend RRF and rerank scores
        7. Normalization - Normalize scores to 0-1 range (if enabled)
        8. Filter and limit - Apply score threshold and result limit

        Args:
            query: Search query string.
            limit: Maximum results to return.
            collection_id: Optional collection ID to limit scope.
            min_score: Minimum score threshold for results.

        Returns:
            List of RankedResult objects sorted by relevance.
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Hybrid search: {query_preview!r}, limit={limit}, collection_id={collection_id}")
        start_time = time.perf_counter()

        # Step 1: Query expansion
        queries = [query]
        if self.config.enable_query_expansion and self.query_expander:
            expansions = await self._expand_query(query)
            queries.extend(expansions)
            logger.debug(f"Query expansion: {len(queries)} queries (original + {len(expansions)} expansions)")

        # Step 2: Parallel FTS5, vector, and tag search for all query variants
        all_results, weights = await self._parallel_search(queries, limit * 3, collection_id)

        # Step 3: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            all_results,
            k=self.config.rrf_k,
            weights=weights,
        )
        logger.debug(f"RRF fusion: {len(fused)} candidates from {len(all_results)} result lists")

        # Take top candidates for reranking
        candidates = fused[: self.config.rerank_candidates]

        # Step 4: Metadata boost (before reranking so reranker can consider boosted order)
        if self.config.enable_metadata_boost and candidates and self.metadata_booster:
            candidates = self._apply_metadata_boost(query, candidates)

        # Step 5 & 6: LLM Reranking with position-aware blending
        if self.config.enable_reranking and candidates and self.reranker:
            final = await self._rerank_with_blending(query, candidates)
        else:
            final = candidates

        # Step 7: Normalize scores to 0-1 range
        if self.config.normalize_final_scores and final:
            final = normalize_scores(final)

        # Step 8: Filter and limit
        final = [r for r in final if r.score >= min_score]
        final = final[:limit]

        elapsed = (time.perf_counter() - start_time) * 1000
        if final:
            top_score = final[0].score if final else 0
            logger.info(f"Search complete: {len(final)} results, top_score={top_score:.3f}, {elapsed:.1f}ms")
        else:
            logger.info(f"Search complete: no results, {elapsed:.1f}ms")

        return final

    async def _expand_query(self, query: str) -> list[str]:
        """Generate query variations using QueryExpander port.

        Args:
            query: Original query string.

        Returns:
            List of query variations (excluding original).
        """
        if not self.query_expander:
            return []

        logger.debug(f"Expanding query: {query[:50]!r}...")
        start_time = time.perf_counter()

        try:
            variations = await self.query_expander.expand(query, num_variations=2)
            # Remove original query if included
            variations = [v for v in variations if v != query]
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Query expanded to {len(variations)} variations in {elapsed:.1f}ms")
            return variations
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Query expansion failed after {elapsed:.1f}ms: {e}")
            return []

    async def _parallel_search(
        self,
        queries: list[str],
        limit: int,
        collection_id: int | None,
    ) -> tuple[list[list[SearchResult]], list[float]]:
        """Run FTS5, vector, and tag search in parallel for all queries.

        Uses the port interfaces: TextSearcher, VectorSearcher, TagSearcher.

        Args:
            queries: List of query strings to search.
            limit: Results per query.
            collection_id: Optional collection ID.

        Returns:
            Tuple of (result_lists, weights) where:
            - result_lists: [fts1, vec1, tag1?, fts2, vec2, tag2?, ...]
            - weights: Corresponding weights for each list
        """
        logger.debug(f"Parallel search: {len(queries)} queries, limit={limit}")
        start_time = time.perf_counter()
        results: list[list[SearchResult]] = []
        weights: list[float] = []
        total_fts = 0
        total_vec = 0
        total_tag = 0

        for i, query in enumerate(queries):
            # Weight factor for original query vs expansions
            is_original = i == 0
            weight_factor = self.config.expansion_weight if not is_original else 1.0

            # FTS search using TextSearcher port
            fts_results = self.text_searcher.search(query, limit, collection_id)
            results.append(fts_results)
            weights.append(self.config.fts_weight * weight_factor)
            total_fts += len(fts_results)

            # Vector search using VectorSearcher port
            vec_results: list[SearchResult] = []
            if self.vector_searcher:
                try:
                    vec_results = await self.vector_searcher.search(query, limit, collection_id)
                except Exception as e:
                    logger.debug(f"Vector search failed for query: {e}")

            results.append(vec_results)
            weights.append(self.config.vec_weight * weight_factor)
            total_vec += len(vec_results)

            # Tag-based search using TagSearcher port
            if self.tag_searcher and self.tag_inferencer:
                tag_results: list[SearchResult] = []
                try:
                    # Infer and expand tags
                    query_tags = self.tag_inferencer.infer_tags(query)
                    if query_tags:
                        expanded_tags = self.tag_inferencer.expand_tags(query_tags)
                        tag_results = self.tag_searcher.search(expanded_tags, limit, collection_id)
                except Exception as e:
                    logger.debug(f"Tag search failed for query: {e}")

                results.append(tag_results)
                weights.append(self.config.tag_weight * weight_factor)
                total_tag += len(tag_results)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Parallel search complete: FTS={total_fts}, VEC={total_vec}, TAG={total_tag}, {elapsed:.1f}ms"
        )

        return results, weights

    async def _rerank_with_blending(
        self,
        query: str,
        candidates: list[RankedResult],
    ) -> list[RankedResult]:
        """Rerank candidates using Reranker port with position-aware score blending.

        Uses the Reranker port to get relevance scores, then applies
        position-aware blending via `pmd.search.scoring.blend_scores`.

        Args:
            query: Original search query.
            candidates: Candidate results to rerank (with RRF scores).

        Returns:
            Reranked results with blended scores.
        """
        if not self.reranker:
            return candidates

        logger.debug(f"Reranking {len(candidates)} candidates with LLM")
        start_time = time.perf_counter()

        try:
            # Get rerank scores from Reranker port
            rerank_scores = await self.reranker.rerank(query, candidates)

            # Convert RerankScore to RerankDocumentResult for blend_scores
            # This maintains compatibility with the scoring module
            from pmd.core.types import RerankDocumentResult
            rerank_results = [
                RerankDocumentResult(
                    file=score.file,
                    relevant=score.relevant,
                    confidence=score.confidence,
                    score=score.score,
                    raw_token="Yes" if score.relevant else "No",
                )
                for score in rerank_scores
            ]

            # Apply position-aware blending
            blended = blend_scores(candidates, rerank_results)
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Reranking complete: {len(blended)} results, {elapsed:.1f}ms")
            return blended
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Reranking failed after {elapsed:.1f}ms: {e}")
            return candidates

    def _apply_metadata_boost(
        self,
        query: str,
        candidates: list[RankedResult],
    ) -> list[RankedResult]:
        """Apply metadata-based score boosting using MetadataBooster port.

        Uses TagInferencer to get query tags, then delegates to MetadataBooster
        for the actual boosting logic.

        Args:
            query: Original search query.
            candidates: Candidate results to boost.

        Returns:
            Boosted and re-sorted results.
        """
        if not self.metadata_booster or not self.tag_inferencer:
            return candidates

        logger.debug(f"Applying metadata boost to {len(candidates)} candidates")
        start_time = time.perf_counter()

        try:
            # Infer and expand tags
            query_tags = self.tag_inferencer.infer_tags(query)
            if not query_tags:
                logger.debug("No tags inferred from query, skipping metadata boost")
                return candidates

            expanded_tags = self.tag_inferencer.expand_tags(query_tags)
            logger.debug(f"Inferred {len(query_tags)} tags, expanded to {len(expanded_tags)}")

            # Apply boost via MetadataBooster port
            boosted = self.metadata_booster.boost(candidates, expanded_tags)

            # Extract results (discard boost info)
            results = [r for r, _boost_info in boosted]
            boosted_count = sum(1 for _, info in boosted if info.boost_applied > 1.0)

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Metadata boost complete: {boosted_count} boosted, {elapsed:.1f}ms")

            return results

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Metadata boost failed after {elapsed:.1f}ms: {e}")
            return candidates
