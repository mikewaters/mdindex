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

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from ..core.types import RankedResult, SearchResult
from ..store.search import FTS5SearchRepository
from .fusion import reciprocal_rank_fusion
from pmd.metadata import Ontology
from .metadata import (
    LexicalTagMatcher,
    MetadataBoostConfig,
    TagRetriever,
    apply_metadata_boost,
    apply_metadata_boost_v2,
    build_path_to_id_map,
    get_document_tags_batch,
)
from .scoring import blend_scores, normalize_scores

if TYPE_CHECKING:
    from ..llm.embeddings import EmbeddingGenerator
    from ..llm.query_expansion import QueryExpander
    from ..llm.reranker import DocumentReranker
    from ..store.document_metadata import DocumentMetadataRepository


@dataclass
class SearchPipelineConfig:
    """Configuration for the hybrid search pipeline.

    Attributes:
        fts_weight: Weight for FTS5 BM25 results in RRF (default: 1.0).
        vec_weight: Weight for vector search results in RRF (default: 1.0).
        tag_weight: Weight for tag-based results in RRF (default: 0.8).
        rrf_k: Smoothing constant for RRF formula (default: 60).
        top_rank_bonus: Bonus score for top-ranked results (default: 0.05).
        expansion_weight: Weight for expanded query results (default: 0.5).
        rerank_candidates: Number of candidates to send to reranker (default: 30).
        enable_query_expansion: Enable LLM query expansion (default: False).
        enable_reranking: Enable LLM reranking with position-aware blending (default: False).
        enable_tag_retrieval: Enable tag-based retrieval in RRF (default: False).
        enable_metadata_boost: Enable metadata-based score boosting (default: False).
        metadata_boost: Configuration for metadata boosting (uses defaults if None).
        normalize_final_scores: Normalize final scores to 0-1 range (default: True).
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
    metadata_boost: MetadataBoostConfig | None = None
    normalize_final_scores: bool = True


class HybridSearchPipeline:
    """Orchestrates hybrid search with FTS, vector, and optional reranking.

    This pipeline combines multiple retrieval strategies using Reciprocal Rank
    Fusion (RRF), with optional LLM-based and metadata-based enhancements:

    1. **Query Expansion**: Generate semantic query variations to improve recall
    2. **Parallel Search**: Run FTS5 and vector search for all query variants
    3. **RRF Fusion**: Combine ranked lists using reciprocal rank fusion
    4. **Metadata Boost**: Boost documents with matching tags (optional)
    5. **LLM Reranking**: Score candidates with LLM relevance judgment
    6. **Position-Aware Blending**: Blend RRF and rerank scores by position

    The position-aware blending strategy (from `pmd.search.scoring.blend_scores`):
    - Rank 1-3: 75% RRF + 25% reranker (trust initial retrieval for top results)
    - Rank 4-10: 60% RRF + 40% reranker (balanced weighting)
    - Rank 11+: 40% RRF + 60% reranker (trust reranker for borderline cases)

    Metadata boosting (from `pmd.search.metadata.scoring.apply_metadata_boost`):
    - Uses LexicalTagMatcher to infer tags from the query
    - Boosts documents whose tags match query-inferred tags
    - Configurable boost factor and max boost cap

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
        - `pmd.search.metadata.scoring.apply_metadata_boost`: Metadata boosting
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
        tag_matcher: LexicalTagMatcher | None = None,
        metadata_repo: "DocumentMetadataRepository | None" = None,
        ontology: Ontology | None = None,
        tag_retriever: TagRetriever | None = None,
    ):
        """Initialize the pipeline.

        Args:
            fts_repo: FTS5SearchRepository for full-text search.
            config: Optional SearchPipelineConfig (uses defaults if None).
            query_expander: Optional QueryExpander for query variations.
            reranker: Optional DocumentReranker for relevance scoring.
            embedding_generator: Optional EmbeddingGenerator for query embeddings.
                                Provides vector search via its embedding_repo.
            tag_matcher: Optional LexicalTagMatcher for query tag inference.
            metadata_repo: Optional DocumentMetadataRepository for document tags.
            ontology: Optional Ontology for hierarchical tag matching.
                     When provided, enables weighted boosting with parent-child
                     tag relationships.
            tag_retriever: Optional TagRetriever for tag-based document retrieval.
                          When provided with enable_tag_retrieval, adds tag results
                          to RRF fusion.
        """
        self.fts_repo = fts_repo
        self.config = config or SearchPipelineConfig()
        self.query_expander = query_expander
        self.reranker = reranker
        self.embedding_generator = embedding_generator
        self.tag_matcher = tag_matcher
        self.metadata_repo = metadata_repo
        self.ontology = ontology
        self.tag_retriever = tag_retriever

        # Ensure expander and reranker are only used if enabled
        if not self.config.enable_query_expansion:
            self.query_expander = None
        if not self.config.enable_reranking:
            self.reranker = None
        # Tag retrieval requires tag_retriever or tag_matcher
        if not self.config.enable_tag_retrieval:
            self.tag_retriever = None
        # Metadata boost requires both matcher and repo
        # Ontology is used by both metadata boost and tag retrieval
        if not self.config.enable_metadata_boost:
            # Only clear these if tag retrieval doesn't need them
            if not self.config.enable_tag_retrieval:
                self.tag_matcher = None
                self.metadata_repo = None
                self.ontology = None

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
        4. Metadata boost - Boost documents with matching tags (if enabled)
        5. LLM reranking - Get relevance scores from LLM (if enabled)
        6. Position-aware blending - Blend RRF and rerank scores
        7. Normalization - Normalize scores to 0-1 range (if enabled)
        8. Filter and limit - Apply score threshold and result limit

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
            - `pmd.search.metadata.scoring.apply_metadata_boost`: Metadata boosting
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Hybrid search: {query_preview!r}, limit={limit}, collection_id={collection_id}")
        start_time = time.perf_counter()

        # Step 1: Query expansion
        queries = [query]
        if self.config.enable_query_expansion:
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
        if self.config.enable_metadata_boost and candidates:
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
        """Generate query variations using LLM.

        Args:
            query: Original query string.

        Returns:
            List of query variations [original, var1, var2, ...].
        """
        if not self.query_expander:
            return []

        logger.debug(f"Expanding query: {query[:50]!r}...")
        start_time = time.perf_counter()

        try:
            variations = await self.query_expander.expand(query, num_variations=2)
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

        Uses FTS5SearchRepository for text search, EmbeddingRepository
        for vector similarity search, and TagRetriever for tag-based search.

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

            # FTS5 search using dedicated repository
            fts_results = self.fts_repo.search(
                query,
                limit,
                collection_id,
            )
            results.append(fts_results)
            weights.append(self.config.fts_weight * weight_factor)
            total_fts += len(fts_results)

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
                except Exception as e:
                    logger.debug(f"Vector search failed for query: {e}")

            results.append(vec_results)
            weights.append(self.config.vec_weight * weight_factor)
            total_vec += len(vec_results)

            # Tag-based search using TagRetriever
            if self.tag_retriever and self.tag_matcher:
                tag_results: list[SearchResult] = []
                try:
                    # Infer tags from query
                    query_tags = self.tag_matcher.get_matching_tags(query)
                    if query_tags:
                        # Expand with ontology if available
                        if self.ontology:
                            expanded_tags = self.ontology.expand_for_matching(query_tags)
                        else:
                            expanded_tags = {tag: 1.0 for tag in query_tags}

                        tag_results = self.tag_retriever.search(
                            expanded_tags,
                            limit,
                            collection_id,
                        )
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

        logger.debug(f"Reranking {len(candidates)} candidates with LLM")
        start_time = time.perf_counter()

        try:
            # Get raw rerank scores from LLM
            rerank_results = await self.reranker.get_rerank_scores(query, candidates)

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
        """Apply metadata-based score boosting to candidates.

        Uses LexicalTagMatcher to infer tags from the query, then boosts
        documents whose tags overlap with query-inferred tags.

        When an ontology is provided, uses weighted boosting where parent tags
        get reduced weights (e.g., parent=0.7x, grandparent=0.49x). Without
        ontology, uses simple set-based matching (v1).

        Args:
            query: Original search query.
            candidates: Candidate results to boost.

        Returns:
            Boosted and re-sorted results.

        See Also:
            - `pmd.search.metadata.scoring.apply_metadata_boost`: v1 boosting
            - `pmd.search.metadata.scoring.apply_metadata_boost_v2`: Weighted boosting
            - `pmd.search.metadata.inference.LexicalTagMatcher`: Tag inference
            - `pmd.search.metadata.ontology.Ontology`: Hierarchical tag expansion
        """
        if not self.tag_matcher or not self.metadata_repo:
            return candidates

        logger.debug(f"Applying metadata boost to {len(candidates)} candidates")
        start_time = time.perf_counter()

        try:
            # Step 1: Infer tags from query
            query_tags = self.tag_matcher.get_matching_tags(query)
            if not query_tags:
                logger.debug("No tags inferred from query, skipping metadata boost")
                return candidates

            logger.debug(f"Inferred query tags: {query_tags}")

            # Step 2: Build path->id map for candidates
            paths = [r.file for r in candidates]
            path_to_id = build_path_to_id_map(self.fts_repo.db, paths)

            # Step 3: Get document tags for candidates
            doc_ids = list(path_to_id.values())
            doc_id_to_tags = get_document_tags_batch(self.metadata_repo, doc_ids)

            # Step 4: Apply boost (v2 with ontology, v1 without)
            if self.ontology:
                # Expand query tags to include ancestors with weights
                expanded_tags = self.ontology.expand_for_matching(query_tags)
                logger.debug(f"Expanded to {len(expanded_tags)} weighted tags: {expanded_tags}")

                # Get boost config values if set
                boost_factor = 1.15
                max_boost = 2.0
                if self.config.metadata_boost:
                    boost_factor = self.config.metadata_boost.boost_factor
                    max_boost = self.config.metadata_boost.max_boost

                boosted = apply_metadata_boost_v2(
                    candidates,
                    expanded_tags,
                    doc_id_to_tags,
                    path_to_id,
                    boost_factor=boost_factor,
                    max_boost=max_boost,
                )
                boosted_count = sum(1 for _, b in boosted if b.boost_applied > 1.0)
            else:
                # v1: Simple set-based matching
                boosted = apply_metadata_boost(
                    candidates,
                    query_tags,
                    doc_id_to_tags,
                    path_to_id,
                    self.config.metadata_boost,
                )
                boosted_count = sum(1 for _, b in boosted if b.boost_applied > 1.0)

            # Extract just the results (discard boost info for now)
            results = [r for r, _boost_info in boosted]

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Metadata boost complete: {boosted_count} boosted, {elapsed:.1f}ms")

            return results

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Metadata boost failed after {elapsed:.1f}ms: {e}")
            return candidates
