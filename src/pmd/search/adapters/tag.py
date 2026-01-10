"""Tag-based search adapters.

Provides adapters for tag-based document retrieval and tag inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmd.core.types import SearchResult
    from pmd.metadata.query.retrieval import TagRetriever
    from pmd.metadata.query.inference import LexicalTagMatcher
    from pmd.metadata.model.ontology import Ontology


class TagRetrieverAdapter:
    """Adapter that wraps TagRetriever for the TagSearcher protocol.

    This adapter provides a thin wrapper around TagRetriever,
    implementing the TagSearcher protocol for use in HybridSearchPipeline.

    Example:
        >>> from pmd.metadata.query.retrieval import TagRetriever
        >>> tag_retriever = TagRetriever(db, metadata_repo)
        >>> searcher = TagRetrieverAdapter(tag_retriever)
        >>> results = searcher.search({"python": 1.0, "ml": 0.7}, limit=10)
    """

    def __init__(self, tag_retriever: "TagRetriever"):
        """Initialize with tag retriever.

        Args:
            tag_retriever: TagRetriever instance.
        """
        self._retriever = tag_retriever

    def search(
        self,
        tags: dict[str, float] | set[str],
        limit: int,
        collection_id: int | None = None,
    ) -> list["SearchResult"]:
        """Search documents by tag matches.

        Args:
            tags: Tags to search for. Can be:
                - dict[str, float]: Weighted tags (from ontology expansion)
                - set[str]: Simple tag set (all weight 1.0)
            limit: Maximum number of results to return.
            collection_id: Optional collection to scope search.

        Returns:
            List of SearchResult objects sorted by tag match score.
        """
        return self._retriever.search(tags, limit=limit, collection_id=collection_id)


class LexicalTagInferencer:
    """Adapter that combines LexicalTagMatcher and Ontology for TagInferencer protocol.

    This adapter encapsulates tag inference from query text and optional
    ontology-based expansion, hiding the complexity from the pipeline.

    Example:
        >>> from pmd.metadata.query.inference import LexicalTagMatcher
        >>> from pmd.metadata.model.ontology import Ontology
        >>> matcher = LexicalTagMatcher(known_tags, aliases)
        >>> ontology = Ontology(adjacency)
        >>> inferencer = LexicalTagInferencer(matcher, ontology)
        >>> tags = inferencer.infer_tags("python ml tutorial")
        >>> expanded = inferencer.expand_tags(tags)
    """

    def __init__(
        self,
        matcher: "LexicalTagMatcher",
        ontology: "Ontology | None" = None,
    ):
        """Initialize with matcher and optional ontology.

        Args:
            matcher: LexicalTagMatcher for inferring tags from text.
            ontology: Optional Ontology for expanding tags to ancestors.
        """
        self._matcher = matcher
        self._ontology = ontology

    def infer_tags(self, query: str) -> set[str]:
        """Infer tags from query text.

        Args:
            query: Search query string.

        Returns:
            Set of inferred tags.
        """
        return self._matcher.get_matching_tags(query)

    def expand_tags(self, tags: set[str]) -> dict[str, float]:
        """Expand tags using ontology relationships.

        If no ontology is configured, returns tags with weight 1.0.

        Args:
            tags: Base tags to expand.

        Returns:
            Dictionary mapping expanded tags to weights.
            Original tags have weight 1.0, ancestors have reduced weight.
        """
        if self._ontology is None:
            return {tag: 1.0 for tag in tags}

        return self._ontology.expand_for_matching(tags)
