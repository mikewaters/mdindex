"""Score normalization and blending for hybrid search."""

from ..core.types import RankedResult, RerankDocumentResult


def normalize_scores(results: list[RankedResult]) -> list[RankedResult]:
    """Normalize scores to 0-1 range.

    Args:
        results: List of ranked results.

    Returns:
        Results with normalized scores.
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

    Position-aware blending strategy:
    - Rank 1-3:   75% RRF + 25% reranker (trust initial ranking)
    - Rank 4-10:  60% RRF + 40% reranker
    - Rank 11+:   40% RRF + 60% reranker (trust reranker more for borderline)

    This approach recognizes that:
    - Highly-ranked results are likely relevant
    - Reranker is more useful for distinguishing borderline cases
    - Hybrid weighting gives best of both strategies

    Args:
        rrf_results: Results from RRF fusion.
        rerank_results: Results from LLM reranking.

    Returns:
        Blended results with updated scores.
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

    Args:
        fts_score: Normalized FTS5 BM25 score (0-1).
        vec_score: Normalized vector similarity score (0-1).
        fts_weight: Weight for FTS component (default 1.0).
        vec_weight: Weight for vector component (default 1.0).

    Returns:
        Weighted combination score (0-1).
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
    """Calculate confidence based on score agreement.

    If primary and secondary agree (within threshold), boost confidence.

    Args:
        primary_score: Primary score (0-1).
        secondary_score: Secondary score (0-1).
        threshold: Maximum difference for "agreement" (default 0.2).

    Returns:
        Confidence score (0-1).
    """
    agreement = 1.0 - abs(primary_score - secondary_score)

    # If scores agree, boost confidence
    if abs(primary_score - secondary_score) <= threshold:
        return min(1.0, primary_score + agreement * 0.1)

    # If scores disagree significantly, reduce confidence
    return primary_score * agreement
