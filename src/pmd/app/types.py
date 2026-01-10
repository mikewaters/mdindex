"""Protocol definitions for dependency injection.

This module defines Protocol types for all injectable dependencies,
allowing services to depend on interfaces rather than concrete implementations.

Protocol types enable:
- Explicit dependency injection in service constructors
- Easy testing with mock/fake implementations
- Loose coupling between components
- Clear contract definitions

Example:
    class MyService:
        def __init__(
            self,
            collection_repo: CollectionRepositoryProtocol,
            document_repo: DocumentRepositoryProtocol,
        ):
            self._collection_repo = collection_repo
            self._document_repo = document_repo
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    import sqlite3
    from ..core.types import (
        Collection,
        DocumentResult,
        EmbeddingResult,
        RerankResult,
        SearchResult,
    )


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
class CollectionRepositoryProtocol(Protocol):
    """Protocol for collection operations."""

    def list_all(self) -> list["Collection"]:
        """Get all collections."""
        ...

    def get_by_name(self, name: str) -> "Collection | None":
        """Get collection by name."""
        ...

    def get_by_id(self, collection_id: int) -> "Collection | None":
        """Get collection by ID."""
        ...

    def create(
        self,
        name: str,
        pwd: str,
        glob_pattern: str = "**/*.md",
        source_type: str = "filesystem",
        source_config: dict[str, Any] | None = None,
    ) -> "Collection":
        """Create a new collection."""
        ...

    def remove(self, collection_id: int) -> tuple[int, int]:
        """Remove a collection and return (docs_deleted, orphans_cleaned)."""
        ...

    def rename(self, collection_id: int, new_name: str) -> None:
        """Rename a collection."""
        ...


@runtime_checkable
class DocumentRepositoryProtocol(Protocol):
    """Protocol for document operations."""

    def add_or_update(
        self,
        collection_id: int,
        path: str,
        title: str,
        content: str,
    ) -> tuple["DocumentResult", bool]:
        """Add or update a document. Returns (result, is_new)."""
        ...

    def get(self, collection_id: int, path: str) -> "DocumentResult | None":
        """Retrieve a document by path."""
        ...

    def get_by_hash(self, hash_value: str) -> str | None:
        """Retrieve content by content hash."""
        ...

    def list_by_collection(
        self,
        collection_id: int,
        active_only: bool = True,
    ) -> list["DocumentResult"]:
        """List all documents in a collection."""
        ...

    def delete(self, collection_id: int, path: str) -> bool:
        """Soft-delete a document."""
        ...


@runtime_checkable
class FTSRepositoryProtocol(Protocol):
    """Protocol for full-text search operations."""

    def search(
        self,
        query: str,
        limit: int = 5,
        collection_id: int | None = None,
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

    def reindex_collection(self, collection_id: int) -> int:
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
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list["SearchResult"]:
        """Search for documents by vector similarity."""
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
        collection_id: int | None = None,
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
# Type Aliases for Convenience
# =============================================================================

# These can be used for type hints when the full protocol isn't needed

CollectionRepo = CollectionRepositoryProtocol
DocumentRepo = DocumentRepositoryProtocol
FTSRepo = FTSRepositoryProtocol
EmbeddingRepo = EmbeddingRepositoryProtocol
