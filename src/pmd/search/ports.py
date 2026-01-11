"""Port definitions for search pipeline.

This module defines the abstract interfaces (ports) that the HybridSearchPipeline
depends on. By programming against these protocols rather than concrete implementations,
the pipeline becomes testable with in-memory fakes and swappable with alternative
implementations.

Protocols defined:
    - TextSearcher: Full-text search capability (FTS/BM25)
    - VectorSearcher: Vector similarity search capability
    - TagSearcher: Tag-based document retrieval
    - QueryExpander: Query expansion via LLM or other means
    - Reranker: Document reranking capability
    - MetadataBooster: Score boosting based on metadata/tag matches
    - TagInferencer: Tag inference from query text

Usage:
    from pmd.search.ports import TextSearcher, VectorSearcher

    class MyCustomSearcher:
        def search(self, query: str, limit: int, source_collection_id: int | None = None):
            # Custom implementation
            ...

    # Type checking will verify protocol compliance
    searcher: TextSearcher = MyCustomSearcher()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pmd.core.types import RankedResult, SearchResult


@dataclass
class BoostInfo:
    """Information about score boosting applied to a result.

    Attributes:
        original_score: Score before boosting.
        boosted_score: Score after boosting.
        matching_tags: Tags that matched with their weights.
        boost_applied: The multiplier applied (1.0 = no boost).
    """

    original_score: float
    boosted_score: float
    matching_tags: dict[str, float]
    boost_applied: float


@dataclass
class RerankScore:
    """Reranking score for a single document.

    Attributes:
        file: Document filepath/identifier.
        score: Relevance score from reranker (0-1).
        relevant: Binary relevance judgment.
        confidence: Confidence in the judgment (0-1).
    """

    file: str
    score: float
    relevant: bool
    confidence: float


@runtime_checkable
class TextSearcher(Protocol):
    """Full-text search capability.

    Implementations provide BM25 or similar lexical search over document content.
    Results are returned as SearchResult objects with scores.

    Example implementation: FTS5TextSearcher wrapping FTS5SearchRepository.
    """

    def search(
        self,
        query: str,
        limit: int,
        source_collection_id: int | None = None,
    ) -> list["SearchResult"]:
        """Search documents using full-text search.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            source_collection_id: Optional collection to scope search.

        Returns:
            List of SearchResult objects sorted by relevance score.
        """
        ...


@runtime_checkable
class VectorSearcher(Protocol):
    """Vector similarity search capability.

    Implementations handle query embedding and similarity search internally.
    The caller provides a raw query string; the implementation handles embedding.

    Example implementation: EmbeddingVectorSearcher wrapping EmbeddingGenerator.
    """

    async def search(
        self,
        query: str,
        limit: int,
        source_collection_id: int | None = None,
    ) -> list["SearchResult"]:
        """Search documents using vector similarity.

        Args:
            query: Search query string (will be embedded internally).
            limit: Maximum number of results to return.
            source_collection_id: Optional collection to scope search.

        Returns:
            List of SearchResult objects sorted by similarity score.
        """
        ...


@runtime_checkable
class TagSearcher(Protocol):
    """Tag-based document retrieval.

    Implementations find documents by matching tags, with optional weighting.
    Supports both simple tag sets and weighted tags from ontology expansion.

    Example implementation: TagRetrieverSearcher wrapping TagRetriever.
    """

    def search(
        self,
        tags: dict[str, float] | set[str],
        limit: int,
        source_collection_id: int | None = None,
    ) -> list["SearchResult"]:
        """Search documents by tag matches.

        Args:
            tags: Tags to search for. Can be:
                - dict[str, float]: Weighted tags (from ontology expansion)
                - set[str]: Simple tag set (all weight 1.0)
            limit: Maximum number of results to return.
            source_collection_id: Optional collection to scope search.

        Returns:
            List of SearchResult objects sorted by tag match score.
        """
        ...


@runtime_checkable
class QueryExpander(Protocol):
    """Query expansion capability.

    Implementations generate query variations to improve recall.
    Typically uses LLM to create semantically similar queries.

    Example implementation: LLMQueryExpander wrapping LLM provider.
    """

    async def expand(
        self,
        query: str,
        num_variations: int = 2,
    ) -> list[str]:
        """Expand query into variations.

        Args:
            query: Original search query.
            num_variations: Number of variations to generate.

        Returns:
            List of query variations (includes original query first).
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Document reranking capability.

    Implementations score documents for relevance to query, typically using LLM.
    Returns scores that can be blended with retrieval scores.

    Example implementation: LLMReranker wrapping DocumentReranker.
    """

    async def rerank(
        self,
        query: str,
        candidates: list["RankedResult"],
    ) -> list[RerankScore]:
        """Rerank candidate documents by relevance.

        Args:
            query: Search query.
            candidates: Candidate documents from retrieval/fusion.

        Returns:
            List of RerankScore objects in same order as candidates.
        """
        ...


@runtime_checkable
class MetadataBooster(Protocol):
    """Score boosting based on metadata/tag matches.

    Implementations encapsulate all logic for looking up document tags
    and calculating boost factors. The pipeline doesn't need to know
    about the underlying metadata storage.

    Example implementation: OntologyMetadataBooster with metadata repo.
    """

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
        ...


@runtime_checkable
class TagInferencer(Protocol):
    """Tag inference from query text.

    Implementations extract likely tags from natural language queries
    and optionally expand them using ontology relationships.

    Example implementation: LexicalTagInferencer with matcher + ontology.
    """

    def infer_tags(self, query: str) -> set[str]:
        """Infer tags from query text.

        Args:
            query: Search query string.

        Returns:
            Set of inferred tags.
        """
        ...

    def expand_tags(self, tags: set[str]) -> dict[str, float]:
        """Expand tags using ontology relationships.

        Args:
            tags: Base tags to expand.

        Returns:
            Dictionary mapping expanded tags to weights.
            Original tags have weight 1.0, ancestors have reduced weight.
        """
        ...
