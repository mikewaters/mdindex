"""Score normalization and blending for hybrid search.

This module provides scoring utilities for the hybrid search pipeline:

Functions:
    normalize_scores: Normalize result scores to 0-1 range
    blend_scores: Position-aware blending of RRF and reranker scores
    weighted_score: Weighted combination of FTS and vector scores
    confidence_score: Score agreement-based confidence calculation

Usage in the Search Pipeline:
    The `HybridSearchPipeline` uses these functions as follows:

    1. **blend_scores**: Applied after LLM reranking to combine RRF fusion
       scores with LLM relevance scores. Uses position-aware weighting that
       trusts top results from initial retrieval while relying on the
       reranker for borderline cases.

    2. **normalize_scores**: Applied as a final step to normalize all scores
       to a 0-1 range for consistent thresholding and display.

Example:
    >>> from pmd.search.scoring import blend_scores, normalize_scores
    >>> from pmd.llm.reranker import DocumentReranker
    >>>
    >>> # Get raw rerank scores from LLM
    >>> rerank_scores = await reranker.get_rerank_scores(query, candidates)
    >>>
    >>> # Apply position-aware blending
    >>> blended = blend_scores(candidates, rerank_scores)
    >>>
    >>> # Normalize to 0-1 range
    >>> final = normalize_scores(blended)

Position-Aware Blending Strategy:
    The `blend_scores` function applies different weights based on result
    position, recognizing that:

    - **Top results (rank 1-3)**: Initial retrieval is usually correct for
      highly-ranked results, so we weight RRF at 75%.

    - **Middle results (rank 4-10)**: More balanced weighting at 60% RRF,
      as these could go either way.

    - **Borderline results (rank 11+)**: The reranker is more useful for
      distinguishing marginally relevant documents, so we weight the
      reranker at 60%.

See Also:
    - `pmd.search.pipeline.HybridSearchPipeline`: Main consumer of these functions
    - `pmd.llm.reranker.DocumentReranker`: Provides rerank scores for blending
    - `pmd.search.fusion.reciprocal_rank_fusion`: Produces RRF scores for blending
"""

from ..core.types import RankedResult, RerankDocumentResult


def normalize_scores(results: list[RankedResult]) -> list[RankedResult]:
    """Normalize scores to 0-1 range using max-normalization.

    Divides all scores by the maximum score, so the highest-scoring result
    gets a score of 1.0 and others are proportionally scaled.

    Used by `HybridSearchPipeline` as a final step to ensure consistent
    score ranges for thresholding (min_score) and display.

    Args:
        results: List of ranked results with arbitrary score ranges.

    Returns:
        New list of RankedResult objects with scores normalized to 0-1.
        Returns empty list if input is empty.
        Returns unchanged if max score is 0.

    Example:
        >>> results = [RankedResult(..., score=0.8), RankedResult(..., score=0.4)]
        >>> normalized = normalize_scores(results)
        >>> [r.score for r in normalized]
        [1.0, 0.5]

    See Also:
        - `HybridSearchPipeline.search`: Uses this for final normalization
    """
    if not results:
        return results

    max_score = max(r.score for r in results)
    if max_score == 0:
        return results

    normalized = []
    for result in results:
        normalized.append(
            RankedResult(
                file=result.file,
                display_path=result.display_path,
                title=result.title,
                body=result.body,
                score=result.score / max_score,
                fts_score=result.fts_score,
                vec_score=result.vec_score,
                rerank_score=result.rerank_score,
            )
        )

    return normalized


def blend_scores(
    rrf_results: list[RankedResult],
    rerank_results: list[RerankDocumentResult],
) -> list[RankedResult]:
    """Blend RRF scores with reranker scores using position-aware weighting.

    This is the core scoring function used by `HybridSearchPipeline` to combine
    initial retrieval scores (from RRF fusion) with LLM relevance judgments.

    Position-Aware Blending Strategy:
        - **Rank 1-3**: 75% RRF + 25% reranker
          Top results from initial retrieval are usually correct.

        - **Rank 4-10**: 60% RRF + 40% reranker
          Balanced weighting for middle-ranked results.

        - **Rank 11+**: 40% RRF + 60% reranker
          Trust the reranker more for borderline relevance decisions.

    This approach recognizes that:
    - Highly-ranked results from retrieval are likely relevant
    - The reranker is more useful for distinguishing borderline cases
    - Hybrid weighting gives best of both strategies

    Args:
        rrf_results: Results from RRF fusion with initial scores.
            These should be sorted by RRF score (highest first).
        rerank_results: Results from LLM reranking via
            `DocumentReranker.get_rerank_scores()`.

    Returns:
        New list of RankedResult objects with blended scores, sorted by
        the new blended score (highest first). Each result includes the
        rerank_score field populated from the reranker output.

    Example:
        >>> from pmd.llm.reranker import DocumentReranker
        >>>
        >>> # Get candidates from RRF fusion
        >>> candidates = reciprocal_rank_fusion(search_results)
        >>>
        >>> # Get LLM relevance scores
        >>> reranker = DocumentReranker(llm_provider)
        >>> rerank_scores = await reranker.get_rerank_scores(query, candidates)
        >>>
        >>> # Apply position-aware blending
        >>> final = blend_scores(candidates, rerank_scores)

    See Also:
        - `HybridSearchPipeline._rerank_with_blending`: Main caller
        - `DocumentReranker.get_rerank_scores`: Provides rerank_results
        - `reciprocal_rank_fusion`: Provides rrf_results
    """
    # Create mapping of filepath -> rerank score
    rerank_map = {r.file: r.score for r in rerank_results}

    blended = []
    for rank, result in enumerate(rrf_results):
        rrf_score = result.score
        rerank_score = rerank_map.get(result.file, 0.5)  # Default to neutral 0.5

        # Determine position-based weight
        if rank < 3:
            # Top 3: trust RRF more
            rrf_weight = 0.75
        elif rank < 10:
            # Rank 4-10: balanced
            rrf_weight = 0.60
        else:
            # Rank 11+: trust reranker more
            rrf_weight = 0.40

        # Calculate final blended score
        final_score = rrf_weight * rrf_score + (1 - rrf_weight) * rerank_score

        blended.append(
            RankedResult(
                file=result.file,
                display_path=result.display_path,
                title=result.title,
                body=result.body,
                score=final_score,
                fts_score=result.fts_score,
                vec_score=result.vec_score,
                rerank_score=rerank_score,
            )
        )

    # Re-sort by blended score
    blended.sort(key=lambda r: r.score, reverse=True)
    return blended


def weighted_score(
    fts_score: float,
    vec_score: float,
    fts_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> float:
    """Calculate weighted combination of FTS and vector scores.

    Utility function for combining scores from different retrieval methods.
    This can be used as an alternative to RRF when you have individual
    FTS and vector scores for a document and want a simple weighted average.

    Note:
        The main search pipeline uses `reciprocal_rank_fusion` instead of
        this function, as RRF operates on ranked lists rather than individual
        scores. This function is available for custom scoring scenarios.

    Args:
        fts_score: Normalized FTS5 BM25 score (0-1).
        vec_score: Normalized vector similarity score (0-1).
        fts_weight: Weight for FTS component (default 1.0).
        vec_weight: Weight for vector component (default 1.0).

    Returns:
        Weighted combination score. If both weights are equal (default),
        this is the simple average. Returns 0.0 if both weights are 0.

    Example:
        >>> # Equal weighting (simple average)
        >>> weighted_score(0.8, 0.6)
        0.7

        >>> # Favor FTS results
        >>> weighted_score(0.8, 0.6, fts_weight=2.0, vec_weight=1.0)
        0.733...

        >>> # Vector-only scoring
        >>> weighted_score(0.8, 0.6, fts_weight=0.0, vec_weight=1.0)
        0.6

    See Also:
        - `reciprocal_rank_fusion`: Rank-based fusion (used by pipeline)
    """
    total_weight = fts_weight + vec_weight
    if total_weight == 0:
        return 0.0

    weighted_sum = fts_score * fts_weight + vec_score * vec_weight
    return weighted_sum / total_weight


def confidence_score(
    primary_score: float,
    secondary_score: float,
    threshold: float = 0.2,
) -> float:
    """Calculate confidence based on score agreement between two signals.

    When multiple scoring signals agree (e.g., FTS and vector, or RRF and
    reranker), we can be more confident in the result. This function
    quantifies that confidence.

    The confidence calculation:
    - If scores agree (within threshold): Boost the primary score slightly
    - If scores disagree: Reduce confidence proportionally to disagreement

    This can be used to:
    - Add confidence metadata to search results
    - Filter results where signals strongly disagree
    - Weight results in downstream processing

    Note:
        This function is available for custom confidence calculations.
        The main pipeline does not currently add confidence metadata,
        but this could be integrated for result explanation features.

    Args:
        primary_score: Primary relevance score (0-1), e.g., RRF score.
        secondary_score: Secondary score (0-1), e.g., reranker score.
        threshold: Maximum difference to consider "agreement" (default 0.2).
            Scores within this threshold get a confidence boost.

    Returns:
        Confidence-adjusted score (0-1). Higher when signals agree,
        lower when they disagree.

    Example:
        >>> # Strong agreement - boost confidence
        >>> confidence_score(0.8, 0.85)
        0.85  # Boosted above primary

        >>> # Within threshold - still boosted
        >>> confidence_score(0.8, 0.65, threshold=0.2)
        0.835  # Slight boost

        >>> # Disagreement - reduced confidence
        >>> confidence_score(0.8, 0.3)
        0.4  # Significantly reduced

    See Also:
        - `blend_scores`: Position-aware score blending (used by pipeline)
        - `DocumentReranker.calculate_confidence`: Reranker-specific confidence
    """
    agreement = 1.0 - abs(primary_score - secondary_score)

    # If scores agree, boost confidence
    if abs(primary_score - secondary_score) <= threshold:
        return min(1.0, primary_score + agreement * 0.1)

    # If scores disagree significantly, reduce confidence
    return primary_score * agreement
