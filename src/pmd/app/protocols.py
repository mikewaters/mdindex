"""Protocol definitions for the pmd application.

This module defines Protocol types for all injectable dependencies,
allowing services to depend on interfaces rather than concrete implementations.

Protocol types enable:
- Explicit dependency injection in service constructors
- Easy testing with mock/fake implementations
- Loose coupling between components
- Clear contract definitions

Protocols are organized by domain:
- Storage: Repository protocols for data persistence
- LLM: Language model provider protocols
- Metadata: Document metadata extraction protocols
- Config: Application configuration protocols
- Search Pipeline: Ports for search pipeline components

Example:
    class MyService:
        def __init__(
            self,
            source_collection_repo: SourceCollectionRepositoryProtocol,
            document_repo: DocumentRepositoryProtocol,
        ):
            self._source_collection_repo = source_collection_repo
            self._document_repo = document_repo
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    import sqlite3
    from ..core.types import (
        DocumentResult,
        EmbeddingResult,
        RankedResult,
        RerankResult,
        SearchResult,
        SourceCollection,
    )
    from ..sources.content.base import DocumentSource
    from ..services.loading import EagerLoadResult, LoadResult


# =============================================================================
# Search Pipeline Data Types
# =============================================================================


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


# =============================================================================
# Database Protocol
# =============================================================================


@runtime_checkable
class DatabaseProtocol(Protocol):
    """Protocol for database operations.

    Services that need direct SQL access should depend on this protocol.
    Most services should prefer repository protocols instead.
    """

    path: Path
    """Path to the database file."""

    @property
    def vec_available(self) -> bool:
        """Check if sqlite-vec extension is loaded."""
        ...

    def connect(self) -> None:
        """Initialize database connection."""
        ...

    def close(self) -> None:
        """Close database connection."""
        ...

    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> "sqlite3.Cursor":
        """Execute SQL statement and return cursor."""
        ...

    @contextmanager
    def transaction(self) -> Iterator["sqlite3.Cursor"]:
        """Context manager for database transactions."""
        ...


# =============================================================================
# Repository Protocols
# =============================================================================


@runtime_checkable
class SourceCollectionRepositoryProtocol(Protocol):
    """Protocol for source collection operations."""

    def list_all(self) -> list["SourceCollection"]:
        """Get all source collections."""
        ...

    def get_by_name(self, name: str) -> "SourceCollection | None":
        """Get source collection by name."""
        ...

    def get_by_id(self, source_collection_id: int) -> "SourceCollection | None":
        """Get source collection by ID."""
        ...

    def create(
        self,
        name: str,
        pwd: str,
        glob_pattern: str = "**/*.md",
        source_type: str = "filesystem",
        source_config: dict[str, Any] | None = None,
    ) -> "SourceCollection":
        """Create a new source collection."""
        ...

    def remove(self, source_collection_id: int) -> tuple[int, int]:
        """Remove a source collection and return (docs_deleted, orphans_cleaned)."""
        ...

    def rename(self, source_collection_id: int, new_name: str) -> None:
        """Rename a source collection."""
        ...


@runtime_checkable
class DocumentRepositoryProtocol(Protocol):
    """Protocol for document operations."""

    def add_or_update(
        self,
        source_collection_id: int,
        path: str,
        title: str,
        content: str,
    ) -> tuple["DocumentResult", bool]:
        """Add or update a document. Returns (result, is_new)."""
        ...

    def get(self, source_collection_id: int, path: str) -> "DocumentResult | None":
        """Retrieve a document by path."""
        ...

    def get_by_hash(self, hash_value: str) -> str | None:
        """Retrieve content by content hash."""
        ...

    def list_by_collection(
        self,
        source_collection_id: int,
        active_only: bool = True,
    ) -> list["DocumentResult"]:
        """List all documents in a collection."""
        ...

    def delete(self, source_collection_id: int, path: str) -> bool:
        """Soft-delete a document."""
        ...

    def get_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID by collection and path."""
        ...


@runtime_checkable
class FTSRepositoryProtocol(Protocol):
    """Protocol for full-text search operations."""

    def search(
        self,
        query: str,
        limit: int = 5,
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list["SearchResult"]:
        """Execute FTS5 full-text search."""
        ...

    def index_document(self, doc_id: int, path: str, body: str) -> None:
        """Index a document in FTS5."""
        ...

    def remove_from_index(self, doc_id: int) -> None:
        """Remove a document from FTS5 index."""
        ...

    def reindex_collection(self, source_collection_id: int) -> int:
        """Reindex all documents in a collection."""
        ...


@runtime_checkable
class EmbeddingRepositoryProtocol(Protocol):
    """Protocol for embedding storage and vector search."""

    def store_embedding(
        self,
        hash_value: str,
        seq: int,
        pos: int,
        embedding: list[float],
        model: str,
    ) -> None:
        """Store embedding vector for a content chunk."""
        ...

    def has_embeddings(self, hash_value: str, model: str | None = None) -> bool:
        """Check if content has embeddings."""
        ...

    def delete_embeddings(self, hash_value: str) -> int:
        """Delete all embeddings for content."""
        ...

    def search_vectors(
        self,
        query_embedding: list[float],
        limit: int = 5,
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list["SearchResult"]:
        """Search for documents by vector similarity."""
        ...

    def delete_orphaned(self) -> int:
        """Delete embedding records not referenced by any active document."""
        ...


# =============================================================================
# LLM Provider Protocol
# =============================================================================


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider operations.

    Note: LLMProvider is already an ABC in pmd.llm.base.
    This protocol allows services to depend on the interface without
    importing the concrete base class.
    """

    async def embed(
        self,
        text: str,
        model: str | None = None,
        is_query: bool = False,
    ) -> "EmbeddingResult | None":
        """Generate embeddings for text."""
        ...

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """Generate text completion."""
        ...

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        model: str | None = None,
    ) -> "RerankResult":
        """Rerank documents by relevance to query."""
        ...

    async def is_available(self) -> bool:
        """Check if the LLM service is available."""
        ...

    async def close(self) -> None:
        """Close the provider and release resources."""
        ...

    def get_default_embedding_model(self) -> str:
        """Get default embedding model name."""
        ...


# =============================================================================
# Service Component Protocols
# =============================================================================


@runtime_checkable
class EmbeddingGeneratorProtocol(Protocol):
    """Protocol for embedding generation operations."""

    async def embed_document(
        self,
        hash_value: str,
        content: str,
        force: bool = False,
    ) -> int:
        """Generate and store embeddings for a document.

        Returns number of chunks embedded.
        """
        ...

    async def embed_query(self, query: str) -> list[float] | None:
        """Generate embedding for a search query."""
        ...


@runtime_checkable
class QueryExpanderProtocol(Protocol):
    """Protocol for query expansion operations."""

    async def expand(self, query: str, num_variants: int = 3) -> list[str]:
        """Expand query into variations."""
        ...


@runtime_checkable
class DocumentRerankerProtocol(Protocol):
    """Protocol for document reranking operations."""

    async def rerank(
        self,
        query: str,
        documents: list[dict],
    ) -> list[tuple[str, float]]:
        """Rerank documents by relevance.

        Returns list of (file, score) tuples.
        """
        ...


# =============================================================================
# Metadata Component Protocols
# =============================================================================


@runtime_checkable
class TagMatcherProtocol(Protocol):
    """Protocol for tag matching operations."""

    def match(self, text: str) -> set[str]:
        """Extract tags from text."""
        ...


@runtime_checkable
class OntologyProtocol(Protocol):
    """Protocol for ontology operations."""

    def expand_tag(self, tag: str) -> dict[str, float]:
        """Expand a tag to related tags with weights."""
        ...

    def get_parent(self, tag: str) -> str | None:
        """Get parent tag in hierarchy."""
        ...

    def get_children(self, tag: str) -> list[str]:
        """Get child tags in hierarchy."""
        ...


@runtime_checkable
class TagRetrieverProtocol(Protocol):
    """Protocol for tag-based document retrieval."""

    def retrieve_by_tags(
        self,
        tags: dict[str, float],
        limit: int = 10,
        source_collection_id: int | None = None,
    ) -> list["SearchResult"]:
        """Retrieve documents matching tags."""
        ...


@runtime_checkable
class DocumentMetadataRepositoryProtocol(Protocol):
    """Protocol for document metadata storage."""

    def get_by_document_id(self, document_id: int) -> Any | None:
        """Get metadata for a document."""
        ...

    def get_tags_for_document(self, document_id: int) -> set[str]:
        """Get tags for a document."""
        ...

    def upsert(self, metadata: Any) -> None:
        """Insert or update document metadata."""
        ...


# =============================================================================
# Configuration Protocols
# =============================================================================


@runtime_checkable
class SearchConfigProtocol(Protocol):
    """Protocol for search configuration."""

    fts_weight: float
    vec_weight: float
    rrf_k: int
    rerank_candidates: int


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol for application configuration.

    Note: Full Config class has many more fields.
    This protocol defines the minimal interface services need.
    """

    db_path: Path
    llm_provider: str

    @property
    def search(self) -> SearchConfigProtocol:
        """Get search configuration."""
        ...


# =============================================================================
# Loading Service Protocol
# =============================================================================


@runtime_checkable
class LoadingServiceProtocol(Protocol):
    """Protocol for document loading service.

    The loading service abstracts retrieval and preparation of source data
    for persistence, keeping IndexingService focused on persistence and
    indexing responsibilities.
    """

    async def load_collection_eager(
        self,
        collection_name: str,
        source: "DocumentSource | None" = None,
        force: bool = False,
    ) -> "EagerLoadResult":
        """Load all documents from a collection (materialized).

        Args:
            collection_name: Name of the collection to load.
            source: Optional source override; resolved from collection if None.
            force: If True, reload all documents regardless of change detection.

        Returns:
            EagerLoadResult with all documents, enumerated paths, and errors.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            SourceListError: If the source cannot enumerate documents.
        """
        ...

    async def load_collection_stream(
        self,
        collection_name: str,
        source: "DocumentSource | None" = None,
        force: bool = False,
    ) -> "LoadResult":
        """Load documents from a collection as a stream.

        Args:
            collection_name: Name of the collection to load.
            source: Optional source override; resolved from collection if None.
            force: If True, reload all documents regardless of change detection.

        Returns:
            LoadResult with async iterator, enumerated paths, and errors.
            Note: enumerated_paths is populated during enumeration, before
            documents are yielded. Errors accumulate as iteration proceeds.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            SourceListError: If the source cannot enumerate documents.
        """
        ...


# =============================================================================
# Search Pipeline Protocols (Ports)
# =============================================================================


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


# =============================================================================
# Type Aliases for Convenience
# =============================================================================

# These can be used for type hints when the full protocol isn't needed

SourceCollectionRepo = SourceCollectionRepositoryProtocol
DocumentRepo = DocumentRepositoryProtocol
FTSRepo = FTSRepositoryProtocol
EmbeddingRepo = EmbeddingRepositoryProtocol
LoadingService = LoadingServiceProtocol
