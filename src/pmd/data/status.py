"""Data access layer for status reporting operations.

This module provides the StatusData class that wraps all repositories needed
for read-only aggregation operations used by StatusService.
"""

from pmd.core.types import SourceCollection
from pmd.store.database import Database
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.embeddings import EmbeddingRepository
from pmd.store.repositories.fts import FTS5SearchRepository


class StatusData:
    """Data access layer for status reporting operations (mostly read-only).

    This class wraps multiple repositories needed by StatusService, creating
    them internally from a single Database instance. It provides a unified
    interface for status-related queries across collections, documents, FTS,
    and embeddings.

    Example:
        db = Database(Path("index.db"))
        db.connect()
        status_data = StatusData(db)

        # Get collection info
        collections = status_data.list_all_collections()

        # Check index health
        missing_fts = status_data.count_documents_missing_fts()
        missing_embeddings = status_data.count_documents_missing_embeddings()
    """

    def __init__(self, db: Database) -> None:
        """Initialize with database connection.

        Creates all required repository instances internally.

        Args:
            db: Database instance to use for operations.
        """
        self._db = db
        self._collections = SourceCollectionRepository(db)
        self._documents = DocumentRepository(db)
        self._fts = FTS5SearchRepository(db)
        self._embeddings = EmbeddingRepository(db)

    # -------------------------------------------------------------------------
    # Collection operations
    # -------------------------------------------------------------------------

    def list_all_collections(self) -> list[SourceCollection]:
        """Get all source collections.

        Returns:
            List of all SourceCollection objects ordered by name.
        """
        return self._collections.list_all()

    def get_collection_by_id(self, collection_id: int) -> SourceCollection | None:
        """Get source collection by ID.

        Args:
            collection_id: Source collection ID to search for.

        Returns:
            SourceCollection object if found, None otherwise.
        """
        return self._collections.get_by_id(collection_id)

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get source collection by name.

        Args:
            name: Source collection name to search for.

        Returns:
            SourceCollection object if found, None otherwise.
        """
        return self._collections.get_by_name(name)

    # -------------------------------------------------------------------------
    # Document operations
    # -------------------------------------------------------------------------

    def count_active_documents(self, source_collection_id: int | None = None) -> int:
        """Count active documents, optionally scoped to a collection.

        Args:
            source_collection_id: Optional collection ID to scope count.

        Returns:
            Number of active documents.
        """
        return self._documents.count_active(source_collection_id)

    def count_documents_with_embeddings(
        self, source_collection_id: int | None = None
    ) -> int:
        """Count documents that have embeddings.

        Args:
            source_collection_id: Optional collection ID to scope count.

        Returns:
            Number of distinct content hashes with embeddings.
        """
        return self._documents.count_with_embeddings(source_collection_id)

    # -------------------------------------------------------------------------
    # FTS operations
    # -------------------------------------------------------------------------

    def count_documents_missing_fts(
        self, source_collection_id: int | None = None
    ) -> int:
        """Count active documents missing FTS index entries.

        Args:
            source_collection_id: Optional collection ID to scope count.

        Returns:
            Number of documents without FTS entries.
        """
        return self._fts.count_documents_missing_fts(source_collection_id)

    def list_paths_missing_fts(
        self, source_collection_id: int | None = None, limit: int = 20
    ) -> list[str]:
        """List paths of documents missing FTS index entries.

        Args:
            source_collection_id: Optional collection ID to scope query.
            limit: Maximum number of paths to return.

        Returns:
            List of document paths without FTS entries.
        """
        return self._fts.list_paths_missing_fts(source_collection_id, limit)

    def count_orphaned_fts(self) -> int:
        """Count FTS entries not linked to any active document.

        Returns:
            Number of orphaned FTS entries.
        """
        return self._fts.count_orphaned()

    # -------------------------------------------------------------------------
    # Embedding operations
    # -------------------------------------------------------------------------

    def count_embeddings(self, model: str | None = None) -> int:
        """Count stored embeddings.

        Args:
            model: Optional model name filter.

        Returns:
            Number of embedding records.
        """
        return self._embeddings.count_embeddings(model)

    def count_distinct_embedding_hashes(self) -> int:
        """Count distinct content hashes with embeddings.

        Returns:
            Number of unique content hashes that have embeddings.
        """
        return self._embeddings.count_distinct_hashes()

    def count_documents_missing_embeddings(
        self, source_collection_id: int | None = None
    ) -> int:
        """Count active documents missing embeddings.

        Args:
            source_collection_id: Optional collection ID to scope count.

        Returns:
            Number of documents without embeddings.
        """
        return self._embeddings.count_documents_missing_embeddings(source_collection_id)

    def list_paths_missing_embeddings(
        self, source_collection_id: int | None = None, limit: int = 20
    ) -> list[str]:
        """List paths of documents missing embeddings.

        Args:
            source_collection_id: Optional collection ID to scope query.
            limit: Maximum number of paths to return.

        Returns:
            List of document paths without embeddings.
        """
        return self._embeddings.list_paths_missing_embeddings(
            source_collection_id, limit
        )

    def count_orphaned_embeddings(self) -> int:
        """Count embedding records not referenced by any active document.

        Returns:
            Number of distinct orphaned content hashes with embeddings.
        """
        return self._embeddings.count_orphaned()
