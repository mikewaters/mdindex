"""Base protocol and types for document sources.

This module defines the core abstractions that all document sources must implement.
Uses Protocol (structural subtyping) for flexibility - sources don't need to
inherit from a base class, just implement the required methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

from pmd.core.exceptions import PMDError
from pmd.sources.metadata.types import ExtractedMetadata

if TYPE_CHECKING:
    pass


# =============================================================================
# Exceptions
# =============================================================================


class SourceError(PMDError):
    """Base exception for source operations."""

    pass


class SourceListError(SourceError):
    """Failed to list documents from source."""

    def __init__(self, source_uri: str, reason: str):
        self.source_uri = source_uri
        self.reason = reason
        super().__init__(f"Failed to list documents from {source_uri}: {reason}")


class SourceFetchError(SourceError):
    """Failed to fetch document content."""

    def __init__(self, uri: str, reason: str, retryable: bool = False):
        self.uri = uri
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Failed to fetch {uri}: {reason}")


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for a document source.

    This is a base configuration that all sources share. Source-specific
    configuration is stored in the `extra` dict.

    Attributes:
        uri: Base URI for the source (e.g., 'file:///path', 'https://docs.example.com')
        extra: Source-specific configuration options
    """

    uri: str
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from extra."""
        return self.extra.get(key, default)


@dataclass(frozen=True)
class DocumentReference:
    """Reference to a document from a source.

    This is a lightweight pointer to a document that can be used to fetch
    its content. It contains enough information to identify the document
    uniquely and to detect changes.

    Attributes:
        uri: Canonical URI for this document (e.g., 'file:///path/to/doc.md')
        path: Relative path for storage (used as the document's identity in the index)
        title: Optional title hint extracted from source metadata
        metadata: Source-specific metadata (etag, last-modified, size, etc.)
    """

    uri: str
    path: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self.metadata.get(key, default)


@dataclass
class FetchResult:
    """Result of fetching document content.

    Contains the content and metadata about the fetch operation.

    Attributes:
        content: The document content as text
        content_type: MIME type of the content (e.g., 'text/markdown', 'text/html')
        encoding: Character encoding used
        metadata: Updated metadata from the fetch (new etag, last-modified, etc.)
        extracted_metadata: Optional normalized metadata emitted by the source
    """

    content: str
    content_type: str = "text/plain"
    encoding: str = "utf-8"
    metadata: dict[str, Any] = field(default_factory=dict)
    extracted_metadata: ExtractedMetadata | None = None


@dataclass(frozen=True)
class SourceCapabilities:
    """Describes what a source implementation supports.

    Used to optimize indexing behavior based on source capabilities.

    Attributes:
        supports_incremental: Can efficiently detect changes without fetching full content
        supports_etag: Supports ETag-based change detection
        supports_last_modified: Supports Last-Modified header/attribute
        supports_streaming: Can stream large documents
        is_readonly: Source content cannot be modified (always True for now)
        provides_document_metadata: Source emits normalized metadata directly
    """

    supports_incremental: bool = False
    supports_etag: bool = False
    supports_last_modified: bool = False
    supports_streaming: bool = False
    is_readonly: bool = True
    provides_document_metadata: bool = False


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class DocumentSource(Protocol):
    """Protocol defining the interface for document sources.

    All document sources must implement this protocol. Sources are responsible
    for listing available documents and fetching their content.

    Example implementation:

        class MyCustomSource:
            def __init__(self, config: SourceConfig):
                self.config = config

            def list_documents(self) -> Iterator[DocumentReference]:
                for item in self._get_items():
                    yield DocumentReference(
                        uri=f"custom://{item.id}",
                        path=item.path,
                        title=item.name,
                    )

            async def fetch_content(self, ref: DocumentReference) -> FetchResult:
                content = await self._fetch(ref.uri)
                return FetchResult(content=content)

            def capabilities(self) -> SourceCapabilities:
                return SourceCapabilities(supports_incremental=True)
    """

    def list_documents(self) -> Iterator[DocumentReference]:
        """Enumerate all documents available from this source.

        Yields DocumentReference objects for each document that should be
        indexed. The yielded references contain enough information to
        fetch the document content later.

        Yields:
            DocumentReference for each available document.

        Raises:
            SourceListError: If enumeration fails.
        """
        ...

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        """Fetch the content of a document.

        Args:
            ref: Reference to the document to fetch.

        Returns:
            FetchResult containing the document content, transport metadata,
            and optional extracted document metadata.

        Raises:
            SourceFetchError: If fetching fails.
        """
        ...

    def capabilities(self) -> SourceCapabilities:
        """Return the capabilities of this source.

        Returns:
            SourceCapabilities describing what this source supports.
        """
        ...

    async def check_modified(
        self,
        ref: DocumentReference,
        stored_metadata: dict[str, Any],
    ) -> bool:
        """Check if a document has been modified since last fetch.

        This method allows sources to implement efficient change detection
        without fetching the full content. For example, using ETags or
        Last-Modified headers.

        The default implementation returns True (always consider modified),
        which causes the indexer to fetch and hash-compare the content.

        Args:
            ref: Reference to the document to check.
            stored_metadata: Metadata from the last successful fetch.

        Returns:
            True if the document may have been modified, False if definitely unchanged.
        """
        ...


# =============================================================================
# Default Implementation Helpers
# =============================================================================


class BaseDocumentSource:
    """Optional base class providing default implementations.

    Sources can inherit from this class to get sensible defaults, or they
    can implement the DocumentSource protocol directly.
    """

    def capabilities(self) -> SourceCapabilities:
        """Default: no special capabilities."""
        return SourceCapabilities()

    async def check_modified(
        self,
        ref: DocumentReference,
        stored_metadata: dict[str, Any],
    ) -> bool:
        """Default: always consider documents potentially modified."""
        return True
