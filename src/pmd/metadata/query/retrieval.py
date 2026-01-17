"""Tag-based document retrieval for search pipeline.

Provides a retrieval channel that finds documents by matching tags,
enabling tag-based search to participate in RRF fusion alongside
FTS and vector search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from pmd.core.types import SearchResult, SearchSource

if TYPE_CHECKING:
    from pmd.store.database import Database
    from pmd.store.repositories.metadata import DocumentMetadataRepository


@dataclass
class TagSearchConfig:
    """Configuration for tag-based retrieval.

    Attributes:
        normalize_scores: Whether to normalize scores to 0-1 range.
        min_score: Minimum score threshold for results.
        max_results: Maximum results to return (before limit).
    """

    normalize_scores: bool = True
    min_score: float = 0.0
    max_results: int = 100


class TagRetriever:
    """Retrieve documents by tag matches with normalized scores.

    This class provides a retrieval channel for the hybrid search pipeline,
    finding documents based on tag matches. When integrated with RRF fusion,
    this allows tags to serve as a primary retrieval signal rather than
    just a post-hoc boosting factor.

    Scoring:
    - With weighted tags (dict): Sum of weights for matching tags
    - With simple tags (set): Count of matching tags
    - Scores are normalized to 0-1 range based on max score

    Example:
        # With weighted tags from ontology expansion
        expanded_tags = ontology.expand_for_matching(["ml/supervised"])
        # {"ml/supervised": 1.0, "ml": 0.7}

        retriever = TagRetriever(db, metadata_repo)
        results = retriever.search(expanded_tags, limit=10)

        # Results are SearchResult objects compatible with RRF fusion

    See Also:
        - `pmd.metadata.Ontology.expand_for_matching`
        - `pmd.search.pipeline.HybridSearchPipeline`
        - `pmd.search.fusion.reciprocal_rank_fusion`
    """

    def __init__(
        self,
        db: "Database",
        metadata_repo: "DocumentMetadataRepository",
        config: TagSearchConfig | None = None,
    ):
        """Initialize the tag retriever.

        Args:
            db: Database connection for document lookups.
            metadata_repo: Repository for document metadata/tags.
            config: Optional configuration for retrieval behavior.
        """
        self.db = db
        self.metadata_repo = metadata_repo
        self.config = config or TagSearchConfig()

    def search(
        self,
        query_tags: dict[str, float] | set[str],
        limit: int = 10,
        source_collection_id: int | None = None,
    ) -> list[SearchResult]:
        """Search for documents matching the given tags.

        Finds documents that have at least one matching tag, then scores
        them based on the overlap with query tags.

        Args:
            query_tags: Tags to search for. Can be:
                - dict[str, float]: Weighted tags (from ontology expansion)
                - set[str]: Simple tag set (all weight 1.0)
            limit: Maximum number of results to return.
            source_collection_id: Optional collection ID to limit scope.

        Returns:
            List of SearchResult objects sorted by score (highest first).

        Example:
            # Weighted search
            results = retriever.search({"python": 1.0, "ml": 0.7}, limit=5)

            # Simple search
            results = retriever.search({"python", "web"}, limit=5)
        """
        if not query_tags:
            return []

        # Convert set to dict if needed
        if isinstance(query_tags, set):
            tags_dict: dict[str, float] = {tag: 1.0 for tag in query_tags}
        else:
            tags_dict = query_tags

        logger.debug(f"Tag search: {len(tags_dict)} tags, limit={limit}")

        # Find documents with any matching tag
        matching_doc_ids = self.metadata_repo.find_documents_with_any_tag(
            set(tags_dict.keys())
        )

        if not matching_doc_ids:
            logger.debug("Tag search: no matching documents")
            return []

        logger.debug(f"Tag search: {len(matching_doc_ids)} documents with matching tags")

        # Get document details and calculate scores
        results = self._score_and_fetch(
            matching_doc_ids,
            tags_dict,
            source_collection_id,
        )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply min_score filter
        if self.config.min_score > 0:
            results = [r for r in results if r.score >= self.config.min_score]

        # Apply limit
        results = results[:limit]

        logger.debug(f"Tag search: returning {len(results)} results")
        return results

    def _score_and_fetch(
        self,
        doc_ids: list[int],
        query_tags: dict[str, float],
        source_collection_id: int | None,
    ) -> list[SearchResult]:
        """Calculate scores and fetch document details.

        Args:
            doc_ids: Document IDs to process.
            query_tags: Weighted tags for scoring.
            source_collection_id: Optional collection filter.

        Returns:
            List of SearchResult objects with scores.
        """
        results: list[SearchResult] = []
        max_score = 0.0

        # Get all tags for matching documents
        doc_id_to_tags: dict[int, set[str]] = {}
        for doc_id in doc_ids:
            doc_id_to_tags[doc_id] = self.metadata_repo.get_tags(doc_id)

        # Build query for document details
        placeholders = ", ".join("?" for _ in doc_ids)
        collection_filter = ""
        params: list = list(doc_ids)

        if source_collection_id is not None:
            collection_filter = " AND d.source_collection_id = ?"
            params.append(source_collection_id)

        cursor = self.db.execute(
            f"""
            SELECT
                d.id,
                d.path,
                d.title,
                d.hash,
                d.source_collection_id,
                d.modified_at,
                c.doc as body
            FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.id IN ({placeholders})
                AND d.active = 1
                {collection_filter}
            """,
            tuple(params),
        )

        rows = cursor.fetchall()

        # Calculate scores and build results
        for row in rows:
            doc_id = row["id"]
            doc_tags = doc_id_to_tags.get(doc_id, set())

            # Calculate weighted score based on tag overlap
            score = 0.0
            for tag, weight in query_tags.items():
                if tag in doc_tags:
                    score += weight

            if score <= 0:
                continue

            max_score = max(max_score, score)

            body = row["body"] or ""
            result = SearchResult(
                filepath=row["path"],
                display_path=row["path"],
                title=row["title"] or "",
                context=body[:200] if body else None,
                hash=row["hash"],
                source_collection_id=row["source_collection_id"],
                modified_at=row["modified_at"],
                body_length=len(body),
                body=body,
                score=score,
                source=SearchSource.TAG,
                chunk_pos=None,
                snippet=body[:200] if body else None,
            )
            results.append(result)

        # Normalize scores if configured
        if self.config.normalize_scores and max_score > 0:
            for result in results:
                result.score = result.score / max_score

        return results


def create_tag_retriever(
    db: "Database",
    metadata_repo: "DocumentMetadataRepository",
    config: TagSearchConfig | None = None,
) -> TagRetriever:
    """Factory function to create a TagRetriever.

    Args:
        db: Database connection.
        metadata_repo: Document metadata repository.
        config: Optional configuration.

    Returns:
        Configured TagRetriever instance.
    """
    return TagRetriever(db, metadata_repo, config)
