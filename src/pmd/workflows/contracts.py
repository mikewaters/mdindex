"""Workflow contracts and data types.

This module defines the dataclasses used as inputs and outputs for workflow
pipeline stages. These contracts establish clear boundaries between nodes
and enable type-safe pipeline composition.

Design principles:
- Reuse existing types when they fit (LoadedDocument, IndexResult, EmbedResult)
- Keep contracts immutable where possible
- Include sufficient context for downstream nodes without over-fetching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Callable, Any, runtime_checkable
import uuid

if TYPE_CHECKING:
    from pmd.core.types import SourceCollection
    from pmd.sources.content.base import DocumentReference, DocumentSource


# =============================================================================
# Re-exports from existing modules
# =============================================================================
# These types are used by workflows but defined elsewhere. We re-export them
# here for convenience and to establish the workflow API surface.

from pmd.services.indexing import IndexResult, EmbedResult
from pmd.services.loading import LoadedDocument

__all__ = [
    # Re-exports from existing modules
    "IndexResult",
    "EmbedResult",
    "LoadedDocument",
    # Workflow context
    "WorkflowContext",
    "ProgressCallback",
    # Ingestion contracts
    "IngestionRequest",
    "ResolvedCollection",
    "EnumeratedRefs",
    "LoadInput",
    "PersistedDocument",
    # Embedding contracts
    "EmbedRequest",
    "EmbedTarget",
    "Chunk",
    "DocumentChunks",
    "EmbeddingVector",
    "EmbeddedChunks",
    "StoreResult",
    # Protocols
    "ChunkerProtocol",
]


# =============================================================================
# Workflow Context
# =============================================================================


@dataclass
class WorkflowContext:
    """Context passed through workflow execution for tracing and coordination.

    Attributes:
        trace_id: Unique identifier for this workflow execution.
        span_id: Current span within the trace (updated per node).
        metadata: Arbitrary metadata for extension.
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def child_span(self) -> "WorkflowContext":
        """Create a child context with new span_id but same trace_id."""
        return WorkflowContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            metadata=self.metadata.copy(),
        )


# =============================================================================
# Progress Callback
# =============================================================================


ProgressCallback = Callable[[int, int, str], None]
"""Callback for reporting progress: (processed, total, current_item)."""


# =============================================================================
# Ingestion Contracts
# =============================================================================


@dataclass
class IngestionRequest:
    """Request to ingest documents from a collection.

    Attributes:
        collection_name: Name of the source collection to ingest.
        force: If True, reload all documents regardless of change detection.
        context: Optional workflow context for tracing.
        progress_callback: Optional callback for progress reporting.
    """

    collection_name: str
    force: bool = False
    context: WorkflowContext | None = None
    progress_callback: ProgressCallback | None = None


@dataclass
class ResolvedCollection:
    """A collection resolved from storage with its configuration.

    Attributes:
        collection: The source collection entity.
        source_config: Parsed source configuration dict.
    """

    collection: "SourceCollection"
    source_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnumeratedRefs:
    """Result of enumerating document references from a source.

    Attributes:
        paths: Set of all enumerated paths (for stale detection).
        count: Total number of references enumerated.
    """

    paths: set[str]
    count: int


@dataclass
class LoadInput:
    """Input for loading a single document.

    Wraps the context needed by the load_document node.

    Attributes:
        ref: Document reference from enumeration.
        collection: Source collection being indexed.
        source: Document source for fetching content.
        force: If True, reload regardless of change detection.
    """

    ref: "DocumentReference"
    collection: "SourceCollection"
    source: "DocumentSource"
    force: bool = False


@dataclass
class PersistedDocument:
    """Result of persisting a single document.

    Attributes:
        doc_id: Database ID of the persisted document.
        path: Document path (identity).
        hash: Content hash.
        indexed_fts: Whether the document was indexed in FTS.
        is_new: Whether this was a new document (vs update).
    """

    doc_id: int
    path: str
    hash: str
    indexed_fts: bool
    is_new: bool


# =============================================================================
# Embedding Contracts
# =============================================================================


@dataclass
class EmbedRequest:
    """Request to generate embeddings for a collection.

    Attributes:
        collection_name: Name of the source collection.
        force: If True, regenerate embeddings even if they exist.
        context: Optional workflow context for tracing.
        progress_callback: Optional callback for progress reporting.
    """

    collection_name: str
    force: bool = False
    context: WorkflowContext | None = None
    progress_callback: ProgressCallback | None = None


@dataclass
class EmbedTarget:
    """A document targeted for embedding generation.

    Attributes:
        doc_hash: Content hash (used as embedding key).
        path: Document path for logging/progress.
        content: Document content to chunk and embed.
    """

    doc_hash: str
    path: str
    content: str


@dataclass
class Chunk:
    """A chunk of document content for embedding.

    Attributes:
        chunk_id: Unique identifier within the document (0-indexed).
        text: The chunk text content.
        start_char: Starting character offset in source document.
        end_char: Ending character offset in source document.
        metadata: Optional metadata (headings, section info, etc.).
    """

    chunk_id: int
    text: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class DocumentChunks:
    """Chunks produced from a single document.

    Attributes:
        doc_hash: Content hash of the source document.
        chunks: List of chunks extracted from the document.
    """

    doc_hash: str
    chunks: list[Chunk]


@dataclass
class EmbeddingVector:
    """A single embedding vector for a chunk.

    Attributes:
        doc_hash: Content hash of the source document.
        chunk_id: Chunk identifier within the document.
        vector: The embedding vector.
        metadata: Optional metadata stored with the vector.
    """

    doc_hash: str
    chunk_id: int
    vector: list[float]
    metadata: dict[str, Any] | None = None


@dataclass
class EmbeddedChunks:
    """Embeddings produced for a document's chunks.

    Attributes:
        doc_hash: Content hash of the source document.
        vectors: List of embedding vectors for each chunk.
    """

    doc_hash: str
    vectors: list[EmbeddingVector]


@dataclass
class StoreResult:
    """Result of storing embeddings for a document.

    Attributes:
        doc_hash: Content hash of the document.
        stored: Number of vectors stored.
        skipped: Number of vectors skipped (already existed).
    """

    doc_hash: str
    stored: int
    skipped: int


# =============================================================================
# Chunker Protocol
# =============================================================================


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol for document chunking implementations.

    Implementations split document content into chunks suitable for embedding.
    Different strategies include:
    - Fixed-size chunking with overlap
    - Sentence-based chunking
    - Semantic chunking (paragraph/section aware)
    """

    def chunk(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split content into chunks.

        Args:
            content: Document content to chunk.
            metadata: Optional document metadata for context-aware chunking.

        Returns:
            List of Chunk objects.
        """
        ...
