"""Data access layer for indexing operations.

This module provides the IndexingData class which aggregates all repository
access needed by IndexingService into a single interface.
"""

from pmd.core.types import DocumentResult, SourceCollection
from pmd.store.database import Database
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.repositories.content import ContentRepository
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.embeddings import EmbeddingRepository
from pmd.store.repositories.fts import FTS5SearchRepository


class IndexingData:
    """Data access layer for indexing operations.

    Wraps all repositories needed by IndexingService, taking only Database
    in its constructor. This provides a clean abstraction over the underlying
    repository implementations.

    Example:
        >>> db = Database(Path("index.db"))
        >>> db.connect()
        >>> data = IndexingData(db)
        >>> collection = data.get_collection_by_name("docs")
    """

    def __init__(self, db: Database) -> None:
        """Initialize with database connection.

        Args:
            db: Database instance to use for all operations.
        """
        self._db = db
        self._collections = SourceCollectionRepository(db)
        self._documents = DocumentRepository(db)
        self._fts = FTS5SearchRepository(db)
        self._content = ContentRepository(db)
        self._embeddings = EmbeddingRepository(db)

    @property
    def db(self) -> Database:
        """Direct database access for raw SQL operations.

        Use this for operations that require direct SQL execution,
        such as backfill_metadata.
        """
        return self._db

    # -------------------------------------------------------------------------
    # Repository access (for transition period - will be removed in Phase 3)
    # -------------------------------------------------------------------------

    @property
    def source_collection_repo(self) -> SourceCollectionRepository:
        """Access to source collection repository (transition period)."""
        return self._collections

    @property
    def document_repo(self) -> DocumentRepository:
        """Access to document repository (transition period)."""
        return self._documents

    @property
    def fts_repo(self) -> FTS5SearchRepository:
        """Access to FTS repository (transition period)."""
        return self._fts

    @property
    def content_repo(self) -> ContentRepository:
        """Access to content repository (transition period)."""
        return self._content

    @property
    def embedding_repo(self) -> EmbeddingRepository:
        """Access to embedding repository (transition period)."""
        return self._embeddings

    # -------------------------------------------------------------------------
    # Collection operations
    # -------------------------------------------------------------------------

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get a source collection by name.

        Args:
            name: The collection name to look up.

        Returns:
            SourceCollection if found, None otherwise.
        """
        return self._collections.get_by_name(name)

    def list_all_collections(self) -> list[SourceCollection]:
        """Get all source collections.

        Returns:
            List of all SourceCollection objects, ordered by name.
        """
        return self._collections.list_all()

    # -------------------------------------------------------------------------
    # Document operations
    # -------------------------------------------------------------------------

    def add_or_update_document(
        self,
        source_collection_id: int,
        path: str,
        title: str,
        content: str,
    ) -> tuple[DocumentResult, bool]:
        """Add or update a document in the index.

        Uses content-addressable storage: stores content in content table
        and references it from documents table via hash.

        Args:
            source_collection_id: ID of the collection.
            path: Document path relative to collection.
            title: Document title.
            content: Document content.

        Returns:
            Tuple of (DocumentResult, is_new) where is_new indicates if this
            was a new document vs an update.
        """
        return self._documents.add_or_update(source_collection_id, path, title, content)

    def get_document(
        self, source_collection_id: int, path: str
    ) -> DocumentResult | None:
        """Retrieve a document by collection and path.

        Args:
            source_collection_id: ID of the collection.
            path: Document path relative to collection.

        Returns:
            DocumentResult if found, None otherwise.
        """
        return self._documents.get(source_collection_id, path)

    def list_documents_by_collection(
        self, source_collection_id: int, active_only: bool = True
    ) -> list[DocumentResult]:
        """List all documents in a collection.

        Args:
            source_collection_id: ID of the collection.
            active_only: If True, only return active documents (default: True).

        Returns:
            List of DocumentResult objects.
        """
        return self._documents.list_by_collection(source_collection_id, active_only)

    def delete_document(self, source_collection_id: int, path: str) -> bool:
        """Soft-delete a document (mark as inactive).

        Args:
            source_collection_id: ID of the collection.
            path: Document path relative to collection.

        Returns:
            True if document was deleted, False if not found.
        """
        return self._documents.delete(source_collection_id, path)

    def get_document_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID by collection and path.

        Args:
            source_collection_id: Collection ID.
            path: Document path.

        Returns:
            Document ID if found, None otherwise.
        """
        return self._documents.get_id(source_collection_id, path)

    # -------------------------------------------------------------------------
    # FTS operations
    # -------------------------------------------------------------------------

    def index_document_for_search(self, doc_id: int, path: str, body: str) -> None:
        """Index a document in the FTS5 full-text search index.

        Adds or updates a document in the FTS5 index for full-text search.
        The document must already exist in the documents table.

        Args:
            doc_id: Document ID from documents table (used as rowid).
            path: Document path (indexed for path-based queries).
            body: Document content to index for full-text search.
        """
        self._fts.index_document(doc_id, path, body)

    def remove_from_search_index(self, doc_id: int) -> None:
        """Remove a document from the FTS5 search index.

        Args:
            doc_id: Document ID to remove from the index.
        """
        self._fts.remove_from_index(doc_id)

    # -------------------------------------------------------------------------
    # Content cleanup
    # -------------------------------------------------------------------------

    def delete_orphaned_content(self) -> int:
        """Delete content entries not referenced by any active document.

        Returns:
            Number of content entries deleted.
        """
        return self._content.delete_orphaned()

    # -------------------------------------------------------------------------
    # Embedding cleanup
    # -------------------------------------------------------------------------

    def delete_orphaned_embeddings(self) -> int:
        """Delete embedding records not referenced by any active document.

        Returns:
            Number of embedding records deleted.
        """
        return self._embeddings.delete_orphaned()
