"""Data access layer for document loading operations.

This module provides the LoadingData class which wraps all repositories
needed by LoadingService, taking only Database in its constructor.
"""

from __future__ import annotations

from pmd.core.types import DocumentResult, SourceCollection
from pmd.store.database import Database
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.source_metadata import SourceMetadata, SourceMetadataRepository


class LoadingData:
    """Data access layer for document loading operations.

    Wraps all repositories needed by LoadingService, providing a unified
    interface for collection lookup, document retrieval, and source metadata
    access required during document loading.

    Example:
        data = LoadingData(db)
        collection = data.get_collection_by_name("my-docs")
        doc = data.get_document(collection.id, "path/to/file.md")
    """

    def __init__(self, db: Database) -> None:
        """Initialize with database connection.

        Creates all required repositories internally.

        Args:
            db: Database instance to use for operations.
        """
        self._db = db
        self._collections = SourceCollectionRepository(db)
        self._documents = DocumentRepository(db)
        self._source_metadata = SourceMetadataRepository(db)

    # =========================================================================
    # Collection Operations
    # =========================================================================

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get a source collection by name.

        Args:
            name: Name of the source collection to retrieve.

        Returns:
            SourceCollection if found, None otherwise.
        """
        return self._collections.get_by_name(name)

    # =========================================================================
    # Document Operations
    # =========================================================================

    def get_document(
        self, source_collection_id: int, path: str
    ) -> DocumentResult | None:
        """Retrieve a document by collection ID and path.

        Args:
            source_collection_id: ID of the source collection.
            path: Document path relative to collection.

        Returns:
            DocumentResult if found and active, None otherwise.
        """
        return self._documents.get(source_collection_id, path)

    def get_document_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID by collection and path.

        Args:
            source_collection_id: ID of the source collection.
            path: Document path relative to collection.

        Returns:
            Document ID if found and active, None otherwise.
        """
        return self._documents.get_id(source_collection_id, path)

    # =========================================================================
    # Source Metadata Operations
    # =========================================================================

    def get_source_metadata_by_document(self, document_id: int) -> SourceMetadata | None:
        """Get source metadata for a document.

        Source metadata includes HTTP caching headers (ETag, Last-Modified),
        fetch timestamps, and other information used for change detection
        during incremental updates.

        Args:
            document_id: ID of the document to get metadata for.

        Returns:
            SourceMetadata if the document has source metadata, None otherwise.
        """
        return self._source_metadata.get_by_document(document_id)
