"""Status service for index health and status reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Awaitable, Protocol

from loguru import logger

from ..core.types import IndexStatus
from ..app.protocols import (
    SourceCollectionRepositoryProtocol,
)


class DocumentRepositoryProtocol(Protocol):
    """Protocol for document repository operations needed by StatusService."""

    def count_active(self, source_collection_id: int | None = None) -> int: ...
    def count_with_embeddings(self, source_collection_id: int | None = None) -> int: ...


class EmbeddingRepositoryProtocol(Protocol):
    """Protocol for embedding repository operations needed by StatusService."""

    def count_embeddings(self, model: str | None = None) -> int: ...
    def count_distinct_hashes(self) -> int: ...
    def count_documents_missing_embeddings(self, source_collection_id: int | None = None) -> int: ...
    def list_paths_missing_embeddings(self, source_collection_id: int | None = None, limit: int = 20) -> list[str]: ...
    def count_orphaned(self) -> int: ...


class FTSRepositoryProtocol(Protocol):
    """Protocol for FTS repository operations needed by StatusService."""

    def count_documents_missing_fts(self, source_collection_id: int | None = None) -> int: ...
    def list_paths_missing_fts(self, source_collection_id: int | None = None, limit: int = 20) -> list[str]: ...
    def count_orphaned(self) -> int: ...


class StatusService:
    """Service for index status and health reporting.

    This service provides information about:
    - Index statistics (document counts, sizes)
    - Collection information
    - LLM provider availability

    Example:

        status_service = StatusService(
            document_repo=document_repo,
            embedding_repo=embedding_repo,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            db_path=config.db_path,
        )
        status = status_service.get_index_status()
    """

    def __init__(
        self,
        document_repo: DocumentRepositoryProtocol,
        embedding_repo: EmbeddingRepositoryProtocol,
        fts_repo: FTSRepositoryProtocol,
        source_collection_repo: SourceCollectionRepositoryProtocol,
        db_path: Path | None = None,
        llm_provider: str = "unknown",
        llm_available_check: Callable[[], Awaitable[bool]] | None = None,
        vec_available: bool = False,
    ):
        """Initialize StatusService.

        Args:
            document_repo: Repository for document operations.
            embedding_repo: Repository for embedding operations.
            fts_repo: Repository for FTS operations.
            source_collection_repo: Repository for source collection operations.
            db_path: Path to the database file.
            llm_provider: Name of the LLM provider.
            llm_available_check: Async function to check if LLM is available.
            vec_available: Whether vector storage is available.
        """
        self._document_repo = document_repo
        self._embedding_repo = embedding_repo
        self._fts_repo = fts_repo
        self._source_collection_repo = source_collection_repo
        self._db_path = db_path
        self._llm_provider = llm_provider
        self._llm_available_check = llm_available_check
        self._vec_available = vec_available

    @property
    def vec_available(self) -> bool:
        """Check if vector storage is available."""
        return self._vec_available

    def get_index_status(self) -> IndexStatus:
        """Get current index status.

        Returns:
            IndexStatus with collection and document information.
        """
        logger.debug("Getting index status")

        source_collections = self._source_collection_repo.list_all()

        # Count total documents
        total_documents = self._document_repo.count_active()

        # Count embedded documents (documents with at least one embedding)
        embedded_documents = 0
        if self.vec_available:
            embedded_documents = self._embedding_repo.count_distinct_hashes()

        # Get database file size
        try:
            index_size_bytes = self._db_path.stat().st_size if self._db_path else 0
        except (OSError, AttributeError):
            index_size_bytes = 0

        # Count embeddings (for cache entries metric)
        cache_entries = 0
        if self.vec_available:
            cache_entries = self._embedding_repo.count_embeddings()

        logger.debug(
            f"Index status: source_collections={len(source_collections)}, "
            f"documents={total_documents}, embedded={embedded_documents}"
        )

        return IndexStatus(
            source_collections=source_collections,
            total_documents=total_documents,
            embedded_documents=embedded_documents,
            index_size_bytes=index_size_bytes,
            cache_entries=cache_entries,
            ollama_available=False,  # Updated async below if needed
            models_available={},
        )

    async def get_full_status(self) -> dict:
        """Get comprehensive status including LLM availability.

        Returns:
            Dictionary with all status information.
        """
        index_status = self.get_index_status()

        # Check LLM availability
        llm_available = False
        if self._llm_available_check:
            llm_available = await self._llm_available_check()

        return {
            "source_collections_count": len(index_status.source_collections),
            "source_collections": [
                {
                    "name": c.name,
                    "path": c.pwd,
                    "glob_pattern": c.glob_pattern,
                }
                for c in index_status.source_collections
            ],
            "total_documents": index_status.total_documents,
            "embedded_documents": index_status.embedded_documents,
            "index_size_bytes": index_status.index_size_bytes,
            "embeddings_count": index_status.cache_entries,
            "database_path": str(self._db_path) if self._db_path else "",
            "llm_provider": self._llm_provider,
            "llm_available": llm_available,
            "vec_available": self.vec_available,
        }

    def get_collection_stats(self, collection_name: str) -> dict | None:
        """Get statistics for a specific collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Dictionary with collection statistics, or None if not found.
        """
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            return None

        # Count documents in source collection
        doc_count = self._document_repo.count_active(source_collection.id)

        # Count embedded documents in source collection
        embedded_count = 0
        if self.vec_available:
            embedded_count = self._document_repo.count_with_embeddings(source_collection.id)

        return {
            "name": source_collection.name,
            "path": source_collection.pwd,
            "glob_pattern": source_collection.glob_pattern,
            "documents": doc_count,
            "embedded": embedded_count,
            "created_at": source_collection.created_at,
            "updated_at": source_collection.updated_at,
        }

    def get_index_sync_report(
        self,
        collection_name: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Report FTS and vector synchronization status.

        Args:
            collection_name: Optional collection name to scope the report.
            limit: Maximum number of sample paths to return per category.

        Returns:
            Dictionary with counts and sample paths for mismatches.
        """
        source_collection_id = None
        if collection_name:
            source_collection = self._source_collection_repo.get_by_name(collection_name)
            if not source_collection:
                return {"error": f"Source collection '{collection_name}' not found"}
            source_collection_id = source_collection.id

        # Documents missing FTS entries
        missing_fts_count = self._fts_repo.count_documents_missing_fts(source_collection_id)
        missing_fts_paths = self._fts_repo.list_paths_missing_fts(source_collection_id, limit)

        # Documents missing embeddings
        missing_vec_count = self._embedding_repo.count_documents_missing_embeddings(source_collection_id)
        missing_vec_paths = self._embedding_repo.list_paths_missing_embeddings(source_collection_id, limit)

        # Orphaned embeddings (no active documents)
        orphan_vec_count = self._embedding_repo.count_orphaned()

        # Orphaned FTS entries (no active documents)
        orphan_fts_count = self._fts_repo.count_orphaned()

        return {
            "collection": collection_name,
            "missing_fts_count": missing_fts_count,
            "missing_fts_paths": missing_fts_paths,
            "missing_vectors_count": missing_vec_count,
            "missing_vectors_paths": missing_vec_paths,
            "orphan_vectors_count": orphan_vec_count,
            "orphan_fts_count": orphan_fts_count,
        }
