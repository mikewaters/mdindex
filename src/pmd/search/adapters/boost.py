"""Metadata boost adapter.

Encapsulates metadata repository lookups and score boosting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from pmd.app.protocols import BoostInfo

if TYPE_CHECKING:
    from pmd.core.types import RankedResult
    from pmd.store.database import Database
    from pmd.store.documents import DocumentRepository
    from pmd.metadata.store import DocumentMetadataRepository
    from pmd.metadata.model.ontology import Ontology


class OntologyMetadataBooster:
    """Adapter that encapsulates metadata-based score boosting.

    This adapter hides all the complexity of looking up document tags
    from the database and calculating boost factors. It implements the
    MetadataBooster protocol.

    The booster supports two modes:
    - With ontology: Uses weighted tag expansion for hierarchical matching
    - Without ontology: Uses simple set-based tag matching

    Example:
        >>> from pmd.metadata.store import DocumentMetadataRepository
        >>> metadata_repo = DocumentMetadataRepository(db)
        >>> booster = OntologyMetadataBooster(db, metadata_repo)
        >>> boosted = booster.boost(results, {"python": 1.0, "ml": 0.7})
    """

    def __init__(
        self,
        db: "Database",
        metadata_repo: "DocumentMetadataRepository",
        ontology: "Ontology | None" = None,
        boost_factor: float = 1.15,
        max_boost: float = 2.0,
        document_repo: "DocumentRepository | None" = None,
    ):
        """Initialize the metadata booster.

        Args:
            db: Database for path-to-ID lookups.
            metadata_repo: Repository for document tag lookups.
            ontology: Optional ontology for hierarchical tag matching.
            boost_factor: Base boost factor for exponential boost (default 1.15).
            max_boost: Maximum allowed boost multiplier (default 2.0).
            document_repo: Repository for document queries.
        """
        self._db = db
        self._metadata_repo = metadata_repo
        self._ontology = ontology
        self._boost_factor = boost_factor
        self._max_boost = max_boost
        self._document_repo = document_repo

    def boost(
        self,
        results: list["RankedResult"],
        query_tags: dict[str, float],
    ) -> list[tuple["RankedResult", BoostInfo]]:
        """Apply metadata-based score boosting.

        Args:
            results: Ranked results to boost.
            query_tags: Tags inferred from query with weights.

        Returns:
            List of (result, boost_info) tuples with updated scores.
            Results are re-sorted by boosted score.
        """
        if not results or not query_tags:
            return [
                (r, BoostInfo(r.score, r.score, {}, 1.0))
                for r in results
            ]

        # Build path -> document ID mapping
        paths = [r.file for r in results]
        path_to_id = self._build_path_to_id_map(paths)

        # Get document IDs we can look up
        doc_ids = list(path_to_id.values())

        # Get tags for all documents
        doc_id_to_tags = self._get_document_tags_batch(doc_ids)

        # Apply boosting
        boosted_results: list[tuple["RankedResult", BoostInfo]] = []

        for result in results:
            doc_id = path_to_id.get(result.file)

            if doc_id is None:
                # Document not found, no boost
                boosted_results.append((
                    result,
                    BoostInfo(result.score, result.score, {}, 1.0),
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
                boost = self._boost_factor ** total_match_weight
                boost = min(boost, self._max_boost)
                boosted_score = result.score * boost
            else:
                boost = 1.0
                boosted_score = result.score

            # Create updated result with new score
            # Note: We need to create a new result since RankedResult might be frozen
            from dataclasses import replace
            updated_result = replace(result, score=boosted_score)

            boosted_results.append((
                updated_result,
                BoostInfo(
                    original_score=result.score,
                    boosted_score=boosted_score,
                    matching_tags=matching_tags,
                    boost_applied=boost,
                ),
            ))

        # Re-sort by boosted score (highest first)
        boosted_results.sort(key=lambda x: x[0].score, reverse=True)

        logger.debug(
            f"Metadata boost applied: {len(results)} results, "
            f"{sum(1 for _, info in boosted_results if info.boost_applied > 1.0)} boosted"
        )

        return boosted_results

    def _build_path_to_id_map(self, paths: list[str]) -> dict[str, int]:
        """Build a mapping from document paths to document IDs.

        Args:
            paths: List of document paths to look up.

        Returns:
            Dictionary mapping path to document ID.
        """
        if not paths:
            return {}

        if self._document_repo:
            return self._document_repo.get_ids_by_paths(paths)

        # Fallback to direct SQL if document_repo not available
        placeholders = ", ".join("?" for _ in paths)
        cursor = self._db.execute(
            f"SELECT id, path FROM documents WHERE path IN ({placeholders}) AND active = 1",
            tuple(paths),
        )

        return {row["path"]: row["id"] for row in cursor.fetchall()}

    def _get_document_tags_batch(self, doc_ids: list[int]) -> dict[int, set[str]]:
        """Get tags for multiple documents.

        Args:
            doc_ids: List of document IDs.

        Returns:
            Dictionary mapping document ID to set of tags.
        """
        result: dict[int, set[str]] = {}
        for doc_id in doc_ids:
            result[doc_id] = self._metadata_repo.get_tags(doc_id)
        return result
