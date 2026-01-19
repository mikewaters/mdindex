"""Repository facades for service layer.

This module provides facade classes that wrap multiple repositories into
unified interfaces for each service. Each facade takes only a Database
instance and creates repositories internally, simplifying dependency
injection in service constructors.

Classes:
    IndexFacade: Facade for indexing operations (documents, FTS, embeddings).
    LoadFacade: Facade for document loading operations.
    SearchFacade: Facade for search operations.
    StatusFacade: Facade for status reporting (read-only aggregations).

Example:
    from pmd.store import Database, IndexFacade

    db = Database("pmd.db")
    db.connect()
    facade = IndexFacade(db)
    collection = facade.get_collection_by_name("docs")
"""

from __future__ import annotations

from pmd.core.types import DocumentResult, SearchResult, SourceCollection
from pmd.store.database import Database
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.repositories.content import ContentRepository
from pmd.store.repositories.documents import DocumentRepository
from pmd.store.repositories.embeddings import EmbeddingRepository
from pmd.store.repositories.fts import FTS5SearchRepository
from pmd.store.repositories.source_metadata import SourceMetadata, SourceMetadataRepository


# =============================================================================
# IndexFacade
# =============================================================================


class IndexFacade:
    """Facade for indexing operations.

    Wraps repositories needed by IndexingService: collections, documents,
    FTS, content, and embeddings.

    Example:
        >>> db = Database(Path("index.db"))
        >>> db.connect()
        >>> facade = IndexFacade(db)
        >>> collection = facade.get_collection_by_name("docs")
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
        """Direct database access for raw SQL operations."""
        return self._db

    # -------------------------------------------------------------------------
    # Collection operations
    # -------------------------------------------------------------------------

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get a source collection by name."""
        return self._collections.get_by_name(name)

    def list_all_collections(self) -> list[SourceCollection]:
        """Get all source collections ordered by name."""
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

        Returns:
            Tuple of (DocumentResult, is_new).
        """
        return self._documents.add_or_update(source_collection_id, path, title, content)

    def get_document(
        self, source_collection_id: int, path: str
    ) -> DocumentResult | None:
        """Retrieve a document by collection and path."""
        return self._documents.get(source_collection_id, path)

    def list_documents_by_collection(
        self, source_collection_id: int, active_only: bool = True
    ) -> list[DocumentResult]:
        """List all documents in a collection."""
        return self._documents.list_by_collection(source_collection_id, active_only)

    def list_active_with_content(
        self, source_collection_id: int
    ) -> list[tuple[str, str, str]]:
        """List active documents with their content.

        Returns:
            List of (path, hash, content) tuples.
        """
        return self._documents.list_active_with_content(source_collection_id)

    def delete_document(self, source_collection_id: int, path: str) -> bool:
        """Soft-delete a document (mark as inactive)."""
        return self._documents.delete(source_collection_id, path)

    def get_document_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID by collection and path."""
        return self._documents.get_id(source_collection_id, path)

    # -------------------------------------------------------------------------
    # FTS operations
    # -------------------------------------------------------------------------

    def index_document_for_search(self, doc_id: int, path: str, body: str) -> None:
        """Index a document in the FTS5 full-text search index."""
        self._fts.index_document(doc_id, path, body)

    def remove_from_search_index(self, doc_id: int) -> None:
        """Remove a document from the FTS5 search index."""
        self._fts.remove_from_index(doc_id)

    # -------------------------------------------------------------------------
    # Content cleanup
    # -------------------------------------------------------------------------

    def delete_orphaned_content(self) -> int:
        """Delete content entries not referenced by any active document."""
        return self._content.delete_orphaned()

    # -------------------------------------------------------------------------
    # Embedding operations
    # -------------------------------------------------------------------------

    def delete_orphaned_embeddings(self) -> int:
        """Delete embedding records not referenced by any active document."""
        return self._embeddings.delete_orphaned()

    def has_embeddings(self, hash_value: str, model: str | None = None) -> bool:
        """Check if a document has embeddings stored."""
        return self._embeddings.has_embeddings(hash_value, model)


# =============================================================================
# LoadFacade
# =============================================================================


class LoadFacade:
    """Facade for document loading operations.

    Wraps repositories needed by LoadingService: collections, documents,
    and source metadata for change detection.

    Example:
        facade = LoadFacade(db)
        collection = facade.get_collection_by_name("my-docs")
        doc = facade.get_document(collection.id, "path/to/file.md")
    """

    def __init__(self, db: Database) -> None:
        """Initialize with database connection."""
        self._db = db
        self._collections = SourceCollectionRepository(db)
        self._documents = DocumentRepository(db)
        self._source_metadata = SourceMetadataRepository(db)

    # -------------------------------------------------------------------------
    # Collection operations
    # -------------------------------------------------------------------------

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get a source collection by name."""
        return self._collections.get_by_name(name)

    # -------------------------------------------------------------------------
    # Document operations
    # -------------------------------------------------------------------------

    def get_document(
        self, source_collection_id: int, path: str
    ) -> DocumentResult | None:
        """Retrieve a document by collection ID and path."""
        return self._documents.get(source_collection_id, path)

    def get_document_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID by collection and path."""
        return self._documents.get_id(source_collection_id, path)

    # -------------------------------------------------------------------------
    # Source metadata operations
    # -------------------------------------------------------------------------

    def get_source_metadata_by_document(self, document_id: int) -> SourceMetadata | None:
        """Get source metadata for a document (ETags, Last-Modified, etc.)."""
        return self._source_metadata.get_by_document(document_id)


# =============================================================================
# SearchFacade
# =============================================================================


class SearchFacade:
    """Facade for search operations.

    Wraps repositories needed by SearchService: collections and FTS.

    Example:
        facade = SearchFacade(db)
        results = facade.search("query text", limit=10)
    """

    def __init__(self, db: Database) -> None:
        """Initialize with database connection."""
        self._db = db
        self._collections = SourceCollectionRepository(db)
        self._fts = FTS5SearchRepository(db)

    @property
    def vec_available(self) -> bool:
        """Check if vector search is available."""
        return self._db.vec_available

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get source collection by name."""
        return self._collections.get_by_name(name)

    def search(
        self,
        query: str,
        limit: int = 5,
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Execute FTS5 full-text search.

        Args:
            query: Search query string (supports FTS5 syntax).
            limit: Maximum number of results.
            source_collection_id: Optional collection ID to limit scope.
            min_score: Minimum normalized score threshold (0.0-1.0).

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        return self._fts.search(
            query,
            limit=limit,
            source_collection_id=source_collection_id,
            min_score=min_score,
        )


# =============================================================================
# StatusFacade
# =============================================================================


class StatusFacade:
    """Facade for status reporting operations (mostly read-only).

    Wraps repositories needed by StatusService for aggregation queries
    across collections, documents, FTS, and embeddings.

    Example:
        facade = StatusFacade(db)
        collections = facade.list_all_collections()
        missing_fts = facade.count_documents_missing_fts()
    """

    def __init__(self, db: Database) -> None:
        """Initialize with database connection."""
        self._db = db
        self._collections = SourceCollectionRepository(db)
        self._documents = DocumentRepository(db)
        self._fts = FTS5SearchRepository(db)
        self._embeddings = EmbeddingRepository(db)

    # -------------------------------------------------------------------------
    # Collection operations
    # -------------------------------------------------------------------------

    def list_all_collections(self) -> list[SourceCollection]:
        """Get all source collections ordered by name."""
        return self._collections.list_all()

    def get_collection_by_id(self, collection_id: int) -> SourceCollection | None:
        """Get source collection by ID."""
        return self._collections.get_by_id(collection_id)

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get source collection by name."""
        return self._collections.get_by_name(name)

    # -------------------------------------------------------------------------
    # Document operations
    # -------------------------------------------------------------------------

    def count_active_documents(self, source_collection_id: int | None = None) -> int:
        """Count active documents, optionally scoped to a collection."""
        return self._documents.count_active(source_collection_id)

    def count_documents_with_embeddings(
        self, source_collection_id: int | None = None
    ) -> int:
        """Count documents that have embeddings."""
        return self._documents.count_with_embeddings(source_collection_id)

    # -------------------------------------------------------------------------
    # FTS operations
    # -------------------------------------------------------------------------

    def count_documents_missing_fts(
        self, source_collection_id: int | None = None
    ) -> int:
        """Count active documents missing FTS index entries."""
        return self._fts.count_documents_missing_fts(source_collection_id)

    def list_paths_missing_fts(
        self, source_collection_id: int | None = None, limit: int = 20
    ) -> list[str]:
        """List paths of documents missing FTS index entries."""
        return self._fts.list_paths_missing_fts(source_collection_id, limit)

    def count_orphaned_fts(self) -> int:
        """Count FTS entries not linked to any active document."""
        return self._fts.count_orphaned()

    # -------------------------------------------------------------------------
    # Embedding operations
    # -------------------------------------------------------------------------

    def count_embeddings(self, model: str | None = None) -> int:
        """Count stored embeddings."""
        return self._embeddings.count_embeddings(model)

    def count_distinct_embedding_hashes(self) -> int:
        """Count distinct content hashes with embeddings."""
        return self._embeddings.count_distinct_hashes()

    def count_documents_missing_embeddings(
        self, source_collection_id: int | None = None
    ) -> int:
        """Count active documents missing embeddings."""
        return self._embeddings.count_documents_missing_embeddings(source_collection_id)

    def list_paths_missing_embeddings(
        self, source_collection_id: int | None = None, limit: int = 20
    ) -> list[str]:
        """List paths of documents missing embeddings."""
        return self._embeddings.list_paths_missing_embeddings(
            source_collection_id, limit
        )

    def count_orphaned_embeddings(self) -> int:
        """Count embedding records not referenced by any active document."""
        return self._embeddings.count_orphaned()
