"""Hybrid search with Reciprocal Rank Fusion (RRF).

Combines full-text search and vector search results using RRF,
then deduplicates by (dataset_name, path) keeping the best chunk.

Example usage:
    from idx.search.hybrid import HybridSearch

    search = HybridSearch(session, vector_search)
    results = search.search(SearchCriteria(query="auth", mode="hybrid"))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.search.fts import FTSSearch
from idx.search.models import SearchCriteria, SearchResult, SearchResults
from idx.search.vector import VectorSearch

if TYPE_CHECKING:
    pass

__all__ = [
    "HybridSearch",
    "rrf_fusion",
]

logger = get_logger(__name__)

# Standard RRF constant (higher k reduces influence of top ranks)
RRF_K = 60


def rrf_fusion(
    ranked_lists: list[list[tuple[str, str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, str, float]]:
    """Combine ranked lists using Reciprocal Rank Fusion.

    RRF computes a combined score for each item based on its rank in each
    list: RRF(d) = sum(1 / (k + rank)) for each list where d appears.

    Args:
        ranked_lists: List of ranked lists. Each inner list contains
            (path, dataset_name, score) tuples sorted by score descending.
        k: RRF constant, typically 60. Higher values reduce the influence
            of top-ranked items.

    Returns:
        Combined list of (path, dataset_name, rrf_score) tuples,
        sorted by RRF score descending.

    Example:
        fts_results = [("a.md", "vault", 0.9), ("b.md", "vault", 0.8)]
        vec_results = [("b.md", "vault", 0.95), ("c.md", "vault", 0.7)]
        combined = rrf_fusion([fts_results, vec_results])
        # b.md gets highest RRF score (appears in both lists with good ranks)
    """
    # Map (path, dataset_name) -> cumulative RRF score
    scores: dict[tuple[str, str], float] = {}

    for ranked_list in ranked_lists:
        for rank, (path, dataset_name, _score) in enumerate(ranked_list, start=1):
            key = (path, dataset_name)
            rrf_score = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_score

    # Sort by RRF score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [(path, ds, score) for (path, ds), score in sorted_items]


class HybridSearch:
    """Hybrid search combining FTS and vector search with RRF fusion.

    Executes both search modes, fuses results with RRF, and deduplicates
    by (dataset_name, path) keeping the best chunk from each document.

    Example:
        hybrid = HybridSearch(session)
        results = hybrid.search(
            SearchCriteria(query="authentication patterns", limit=10)
        )
    """

    def __init__(
        self,
        session: Session,
        vector_search: VectorSearch | None = None,
        persist_dir: Path | None = None,
    ) -> None:
        """Initialize hybrid search.

        Args:
            session: SQLAlchemy session for FTS operations.
            vector_search: Optional pre-initialized VectorSearch instance.
                If not provided, a new one will be created.
            persist_dir: Directory for vector store persistence.
                Only used when vector_search is not provided.
        """
        self._session = session
        self._fts = FTSSearch(session)
        self._vector = vector_search or VectorSearch(persist_dir=persist_dir)

    def search(self, criteria: SearchCriteria) -> SearchResults:
        """Execute hybrid search with RRF fusion.

        Runs both FTS and vector search, fuses results with RRF,
        normalizes scores, and returns deduplicated results.

        Args:
            criteria: Search criteria including query, limit, and dataset filter.

        Returns:
            SearchResults with RRF-fused scores.
        """
        import time

        start = time.perf_counter()

        # Get more candidates from each source for better fusion
        candidate_multiplier = 3
        candidate_limit = criteria.limit * candidate_multiplier

        # Create modified criteria for sub-searches
        fts_criteria = SearchCriteria(
            query=criteria.query,
            mode="fts",
            dataset_name=criteria.dataset_name,
            limit=candidate_limit,
        )

        # Run FTS search
        fts_results = self._fts.search(fts_criteria)
        fts_ranked = [
            (r.path, r.dataset_name, r.score)
            for r in fts_results.results
        ]

        # Run vector search
        vec_ranked = self._vector.search_with_scores(
            criteria.query,
            dataset_name=criteria.dataset_name,
            limit=candidate_limit,
        )

        # Fuse with RRF
        fused = rrf_fusion([fts_ranked, vec_ranked])

        # Build result objects with provenance
        results = self._build_results(fused, fts_results, vec_ranked)

        # Normalize RRF scores to 0-1
        results = self._normalize_scores(results)

        # Apply final limit
        results = results[: criteria.limit]

        elapsed_ms = (time.perf_counter() - start) * 1000
        total_candidates = len(fts_results.results) + len(vec_ranked)

        logger.debug(
            f"Hybrid search '{criteria.query}' returned {len(results)} results "
            f"(from {total_candidates} candidates) in {elapsed_ms:.1f}ms"
        )

        return SearchResults(
            results=results,
            query=criteria.query,
            mode="hybrid",
            total_candidates=total_candidates,
            timing_ms=elapsed_ms,
        )

    def _build_results(
        self,
        fused: list[tuple[str, str, float]],
        fts_results: SearchResults,
        vec_ranked: list[tuple[str, str, float]],
    ) -> list[SearchResult]:
        """Build SearchResult objects with component scores.

        Args:
            fused: RRF-fused (path, dataset_name, rrf_score) tuples.
            fts_results: Original FTS results for component scores.
            vec_ranked: Original vector results for component scores.

        Returns:
            List of SearchResult objects with all scores populated.
        """
        # Build lookup maps for component scores
        fts_scores: dict[tuple[str, str], float] = {
            (r.path, r.dataset_name): r.score
            for r in fts_results.results
        }
        vec_scores: dict[tuple[str, str], float] = {
            (path, ds): score
            for path, ds, score in vec_ranked
        }

        results = []
        for path, dataset_name, rrf_score in fused:
            key = (path, dataset_name)
            scores = {"rrf": rrf_score}

            if key in fts_scores:
                scores["fts"] = fts_scores[key]
            if key in vec_scores:
                scores["vector"] = vec_scores[key]

            results.append(
                SearchResult(
                    path=path,
                    dataset_name=dataset_name,
                    score=rrf_score,
                    scores=scores,
                )
            )

        return results

    def _normalize_scores(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Normalize RRF scores to 0-1 range.

        Uses max normalization: each score is divided by the maximum score.

        Args:
            results: List of SearchResult objects.

        Returns:
            Results with normalized scores.
        """
        if not results:
            return results

        max_score = max(r.score for r in results)
        if max_score <= 0:
            return results

        for result in results:
            result.score = result.score / max_score
            result.scores["rrf"] = result.score

        return results
