"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from ..core.types import RankedResult, SearchResult, SearchSource


@dataclass
class _DocProvenance:
    """Internal tracking of document provenance during fusion."""

    doc: SearchResult
    fts_score: Optional[float] = None
    vec_score: Optional[float] = None
    fts_rank: Optional[int] = None
    vec_rank: Optional[int] = None
    sources: set = None  # type: ignore

    def __post_init__(self):
        if self.sources is None:
            self.sources = set()


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
            Expected order: [fts1, vec1, fts2, vec2, ...] where odd indices are FTS,
            even indices are vector results.
        k: Smoothing constant (default 60, prevents score explosion).
        original_query_weight: Extra weight for original query results.
        weights: Optional per-list weights. If None, uses [original_query_weight,
                 original_query_weight] + [1.0] * (len(result_lists) - 2)

    Returns:
        List of RankedResult objects sorted by fused score (highest first).
        Includes provenance fields: fts_rank, vec_rank, sources_count.
    """
    scores: dict[str, float] = defaultdict(float)
    provenance: dict[str, _DocProvenance] = {}

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

            # Track provenance
            if result.filepath not in provenance:
                provenance[result.filepath] = _DocProvenance(doc=result)

            prov = provenance[result.filepath]

            # Determine source from the result's source attribute
            is_fts = result.source == SearchSource.FTS

            if is_fts:
                prov.sources.add("fts")
                # Keep best FTS score and first (best) rank
                if prov.fts_rank is None or rank < prov.fts_rank:
                    prov.fts_rank = rank
                if prov.fts_score is None or result.score > prov.fts_score:
                    prov.fts_score = result.score
            else:
                prov.sources.add("vec")
                # Keep best vector score and first (best) rank
                if prov.vec_rank is None or rank < prov.vec_rank:
                    prov.vec_rank = rank
                if prov.vec_score is None or result.score > prov.vec_score:
                    prov.vec_score = result.score

            # Keep the result with highest individual score per document
            if result.score > prov.doc.score:
                prov.doc = result

    # Sort documents by fused score (highest first)
    sorted_files = sorted(scores.keys(), key=lambda f: scores[f], reverse=True)

    # Convert to RankedResult objects with fused scores and provenance
    ranked_results = []
    for filepath in sorted_files:
        prov = provenance[filepath]
        doc = prov.doc

        result = RankedResult(
            file=doc.filepath,
            display_path=doc.display_path,
            title=doc.title,
            body=doc.body or "",
            score=scores[filepath],
            fts_score=prov.fts_score,
            vec_score=prov.vec_score,
            rerank_score=None,
            fts_rank=prov.fts_rank,
            vec_rank=prov.vec_rank,
            sources_count=len(prov.sources),
        )
        ranked_results.append(result)

        # Debug logging for full diagnostics
        sources_str = "+".join(sorted(prov.sources))
        ranks_str = []
        if prov.fts_rank is not None:
            ranks_str.append(f"FTS#{prov.fts_rank + 1}")
        if prov.vec_rank is not None:
            ranks_str.append(f"VEC#{prov.vec_rank + 1}")
        fts_str = f"{prov.fts_score:.3f}" if prov.fts_score is not None else "N/A"
        vec_str = f"{prov.vec_score:.3f}" if prov.vec_score is not None else "N/A"
        logger.debug(
            f"RRF: {doc.title[:40]!r} | rrf={scores[filepath]:.4f} | "
            f"sources={sources_str} | ranks={','.join(ranks_str)} | "
            f"fts={fts_str} | vec={vec_str}"
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
