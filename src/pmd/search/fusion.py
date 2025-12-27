"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists."""

from collections import defaultdict

from ..core.types import RankedResult, SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    original_query_weight: float = 2.0,
    weights: list[float] | None = None,
) -> list[RankedResult]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF Score = Σ(weight / (k + rank + 1))

    The algorithm:
    1. For each result list, assign a rank (0-indexed)
    2. Calculate RRF score: weight / (k + rank + 1)
    3. Add bonuses for top ranks (+0.05 for #1, +0.02 for #2-3)
    4. Accumulate scores for duplicate items across lists
    5. Sort by final fused score

    Args:
        result_lists: List of ranked result lists to fuse.
        k: Smoothing constant (default 60, prevents score explosion).
        original_query_weight: Extra weight for original query results.
        weights: Optional per-list weights. If None, uses [original_query_weight,
                 original_query_weight] + [1.0] * (len(result_lists) - 2)

    Returns:
        List of RankedResult objects sorted by fused score (highest first).
    """
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, SearchResult] = {}

    if weights is None:
        # Default: first two lists (original query FTS and vector) get higher weight
        weights = [original_query_weight, original_query_weight] + [1.0] * (
            len(result_lists) - 2
        )

    # Process each result list
    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0

        for rank, result in enumerate(results):
            # Calculate RRF score using the formula
            rrf_score = weight / (k + rank + 1)

            # Apply top-rank bonuses for better ranking of highly relevant items
            if rank == 0:
                rrf_score += 0.05
            elif rank <= 2:
                rrf_score += 0.02

            # Accumulate score for this document (key by filepath)
            scores[result.filepath] += rrf_score

            # Keep the result with highest individual score per document
            if result.filepath not in docs or result.score > docs[result.filepath].score:
                docs[result.filepath] = result

    # Sort documents by fused score (highest first)
    sorted_files = sorted(scores.keys(), key=lambda f: scores[f], reverse=True)

    # Convert to RankedResult objects with fused scores
    ranked_results = []
    for filepath in sorted_files:
        doc = docs[filepath]
        ranked_results.append(
            RankedResult(
                file=doc.filepath,
                display_path=doc.display_path,
                title=doc.title,
                body=doc.body or "",
                score=scores[filepath],
                fts_score=(doc.score if doc.source.value == "fts" else None),
                vec_score=(doc.score if doc.source.value == "vec" else None),
                rerank_score=None,
            )
        )

    return ranked_results


def rrf_formula(rank: int, k: int = 60, weight: float = 1.0) -> float:
    """Calculate RRF score for a single rank.

    Args:
        rank: 0-indexed position in ranked list.
        k: Smoothing constant.
        weight: Weight multiplier for this list.

    Returns:
        RRF score component.
    """
    return weight / (k + rank + 1)


def calculate_reciprocal_rank(score: float, k: int = 60) -> float:
    """Calculate rank from RRF score (inverse of rrf_formula).

    Args:
        score: RRF score.
        k: Smoothing constant.

    Returns:
        Estimated rank (0-indexed).
    """
    if score <= 0:
        return float("inf")
    return (1.0 / score) - k - 1
