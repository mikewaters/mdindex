"""Metadata-based scoring for search results.

Provides functions to boost search result scores based on tag matches,
improving relevance when document tags match query-inferred tags.

Provides types and functions for metadata-based scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Import ScoredResult from canonical location
from pmd.ontology.booster_scoring import ScoredResult

if TYPE_CHECKING:
    from pmd.store.repositories.metadata import DocumentMetadataRepository


@dataclass
class MetadataBoostConfig:
    """Configuration for metadata-based score boosting.

    Attributes:
        boost_factor: Multiplicative boost for matching documents (e.g., 1.5 = 50% boost).
        max_boost: Maximum total boost to apply (caps cumulative boosting).
        min_tags_for_boost: Minimum number of matching tags required for boost.
    """

    boost_factor: float = 1.3
    max_boost: float = 2.0
    min_tags_for_boost: int = 1


@dataclass
class BoostResult:
    """Result of applying metadata boost to a document.

    Attributes:
        original_score: Score before boosting.
        boosted_score: Score after boosting.
        matching_tags: Tags that matched between query and document.
        boost_applied: The actual boost multiplier applied.
    """

    original_score: float
    boosted_score: float
    matching_tags: set[str]
    boost_applied: float


def apply_metadata_boost(
    results: list[ScoredResult],
    query_tags: set[str],
    doc_id_to_tags: dict[int, set[str]],
    doc_path_to_id: dict[str, int],
    config: MetadataBoostConfig | None = None,
) -> list[tuple[ScoredResult, BoostResult]]:
    """Apply metadata-based score boosting to search results.

    Boosts scores for documents whose tags overlap with query-inferred tags.
    Uses multiplicative boosting with a configurable cap.

    Args:
        results: List of search results with scores.
        query_tags: Tags inferred from the search query.
        doc_id_to_tags: Mapping of document ID to its tags.
        doc_path_to_id: Mapping of document path to document ID.
        config: Boost configuration (uses defaults if None).

    Returns:
        List of (result, boost_info) tuples with updated scores.
        Results are re-sorted by boosted score.

    Example:
        results = search_pipeline.search("python web api")
        query_tags = matcher.get_matching_tags("python web api")

        boosted = apply_metadata_boost(
            results,
            query_tags,
            doc_id_to_tags={1: {"python", "web"}, 2: {"rust"}},
            doc_path_to_id={"docs/py.md": 1, "docs/rs.md": 2},
        )
        # Results with python/web tags will be boosted
    """
    if config is None:
        config = MetadataBoostConfig()

    if not query_tags:
        # No tags to match, return unchanged
        return [
            (r, BoostResult(r.score, r.score, set(), 1.0))
            for r in results
        ]

    boosted_results: list[tuple[ScoredResult, BoostResult]] = []

    for result in results:
        doc_id = doc_path_to_id.get(result.file)
        if doc_id is None:
            # Document not found, no boost
            boosted_results.append((
                result,
                BoostResult(result.score, result.score, set(), 1.0),
            ))
            continue

        doc_tags = doc_id_to_tags.get(doc_id, set())
        matching_tags = query_tags & doc_tags

        if len(matching_tags) >= config.min_tags_for_boost:
            # Calculate boost based on number of matching tags
            boost = _calculate_boost(
                len(matching_tags),
                config.boost_factor,
                config.max_boost,
            )
            boosted_score = result.score * boost
        else:
            boost = 1.0
            boosted_score = result.score

        # Update the result's score in place
        result.score = boosted_score

        boosted_results.append((
            result,
            BoostResult(
                original_score=boosted_score / boost if boost > 0 else boosted_score,
                boosted_score=boosted_score,
                matching_tags=matching_tags,
                boost_applied=boost,
            ),
        ))

    # Re-sort by boosted score (highest first)
    boosted_results.sort(key=lambda x: x[0].score, reverse=True)

    return boosted_results


def _calculate_boost(
    num_matches: int,
    boost_factor: float,
    max_boost: float,
) -> float:
    """Calculate the boost multiplier for a number of matching tags.

    Uses a logarithmic formula to provide diminishing returns for
    additional tag matches, capped at max_boost.

    Args:
        num_matches: Number of matching tags.
        boost_factor: Base boost factor.
        max_boost: Maximum allowed boost.

    Returns:
        Boost multiplier (1.0 = no boost).
    """
    if num_matches <= 0:
        return 1.0

    # Linear boost per match, but capped
    # Each additional match adds progressively less boost
    boost = 1.0 + (boost_factor - 1.0) * (1.0 + 0.5 * (num_matches - 1))
    return min(boost, max_boost)


@dataclass
class WeightedBoostResult:
    """Result of applying weighted metadata boost.

    Attributes:
        original_score: Score before boosting.
        boosted_score: Score after boosting.
        matching_tags: Tags that matched with their weights.
        total_match_weight: Sum of matching tag weights.
        boost_applied: The actual boost multiplier applied.
    """

    original_score: float
    boosted_score: float
    matching_tags: dict[str, float]  # tag -> weight
    total_match_weight: float
    boost_applied: float


def apply_metadata_boost_v2(
    results: list[ScoredResult],
    query_tags: dict[str, float],
    doc_id_to_tags: dict[int, set[str]],
    doc_path_to_id: dict[str, int],
    boost_factor: float = 1.15,
    max_boost: float = 2.0,
) -> list[tuple[ScoredResult, WeightedBoostResult]]:
    """Apply metadata-based score boosting with weighted tag matches.

    This version supports ontology expansion where query tags have
    different weights (exact matches = 1.0, parent matches < 1.0).

    The boost is calculated as: boost_factor ** total_match_weight,
    where total_match_weight is the sum of weights for matching tags.

    Args:
        results: List of search results with scores.
        query_tags: Dictionary mapping tags to weights (from ontology expansion).
        doc_id_to_tags: Mapping of document ID to its tags.
        doc_path_to_id: Mapping of document path to document ID.
        boost_factor: Base boost factor for exponential boost.
        max_boost: Maximum allowed boost multiplier.

    Returns:
        List of (result, boost_info) tuples with updated scores.
        Results are re-sorted by boosted score.

    Example:
        # From ontology expansion
        query_tags = {
            "ml/supervised/regression": 1.0,
            "ml/supervised": 0.7,
            "ml": 0.49,
        }

        boosted = apply_metadata_boost_v2(
            results,
            query_tags,
            doc_id_to_tags={1: {"ml/supervised", "python"}},
            doc_path_to_id={"docs/ml.md": 1},
        )
        # Doc 1 matches "ml/supervised" (0.7 weight) -> boost = 1.15 ** 0.7 = 1.10
    """
    if not query_tags:
        return [
            (r, WeightedBoostResult(r.score, r.score, {}, 0.0, 1.0))
            for r in results
        ]

    boosted_results: list[tuple[ScoredResult, WeightedBoostResult]] = []

    for result in results:
        doc_id = doc_path_to_id.get(result.file)
        if doc_id is None:
            # Document not found, no boost
            boosted_results.append((
                result,
                WeightedBoostResult(result.score, result.score, {}, 0.0, 1.0),
            ))
            continue

        doc_tags = doc_id_to_tags.get(doc_id, set())

        # Calculate weighted matches
        matching_tags: dict[str, float] = {}
        for tag, weight in query_tags.items():
            if tag in doc_tags:
                matching_tags[tag] = weight

        total_match_weight = sum(matching_tags.values())

        if total_match_weight > 0:
            # Exponential boost scaled by match weight
            boost = boost_factor ** total_match_weight
            boost = min(boost, max_boost)
            boosted_score = result.score * boost
        else:
            boost = 1.0
            boosted_score = result.score

        # Update the result's score in place
        result.score = boosted_score

        boosted_results.append((
            result,
            WeightedBoostResult(
                original_score=boosted_score / boost if boost > 0 else boosted_score,
                boosted_score=boosted_score,
                matching_tags=matching_tags,
                total_match_weight=total_match_weight,
                boost_applied=boost,
            ),
        ))

    # Re-sort by boosted score (highest first)
    boosted_results.sort(key=lambda x: x[0].score, reverse=True)

    return boosted_results


def get_document_tags_batch(
    metadata_repo: "DocumentMetadataRepository",
    document_ids: list[int],
) -> dict[int, set[str]]:
    """Get tags for multiple documents efficiently.

    Args:
        metadata_repo: Document metadata repository.
        document_ids: List of document IDs to fetch tags for.

    Returns:
        Dictionary mapping document ID to set of tags.
    """
    result: dict[int, set[str]] = {}
    for doc_id in document_ids:
        result[doc_id] = metadata_repo.get_tags(doc_id)
    return result


def build_path_to_id_map(
    db,
    paths: list[str],
) -> dict[str, int]:
    """Build a mapping from document paths to document IDs.

    Args:
        db: Database instance.
        paths: List of document paths to look up.

    Returns:
        Dictionary mapping path to document ID.
    """
    if not paths:
        return {}

    placeholders = ", ".join("?" for _ in paths)
    cursor = db.execute(
        f"SELECT id, path FROM documents WHERE path IN ({placeholders}) AND active = 1",
        tuple(paths),
    )

    return {row["path"]: row["id"] for row in cursor.fetchall()}
