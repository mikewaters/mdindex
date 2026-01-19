"""Status service for index health and status reporting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from pmd.store import StatusFacade
from ..core.types import IndexStatus

if TYPE_CHECKING:
    from ..llm.base import LLMProvider


class StatusService:
    """Service for index status and health reporting.

    This service provides information about:
    - Index statistics (document counts, sizes)
    - Collection information
    - LLM provider availability (for diagnostics)

    Example:

        status_service = StatusService(
            facade=status_facade,
            db_path=config.db_path,
        )
        status = status_service.get_index_status()
    """

    def __init__(
        self,
        facade: StatusFacade,
        db_path: Path | None = None,
        llm_provider_name: str = "unknown",
        llm_provider_instance: "LLMProvider | None" = None,
        vec_available: bool = False,
    ):
        """Initialize StatusService.

        Args:
            facade: Facade for status data operations.
            db_path: Path to the database file.
            llm_provider_name: Name of the LLM provider (for display).
            llm_provider_instance: LLM provider for availability checks.
            vec_available: Whether vector storage is available.
        """
        self._data = facade
        self._db_path = db_path
        self._llm_provider_name = llm_provider_name
        self._llm_provider_instance = llm_provider_instance
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

        source_collections = self._data.list_all_collections()

        # Count total documents
        total_documents = self._data.count_active_documents()

        # Count embedded documents (documents with at least one embedding)
        embedded_documents = 0
        if self.vec_available:
            embedded_documents = self._data.count_distinct_embedding_hashes()

        # Get database file size
        try:
            index_size_bytes = self._db_path.stat().st_size if self._db_path else 0
        except (OSError, AttributeError):
            index_size_bytes = 0

        # Count embeddings (for cache entries metric)
        cache_entries = 0
        if self.vec_available:
            cache_entries = self._data.count_embeddings()

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

    async def check_llm_available(self) -> bool:
        """Check if LLM provider is available and reachable.

        This is a diagnostic method for callers who want to verify
        LLM connectivity before operations. Note that operations should
        generally just attempt to use the LLM and handle errors naturally
        rather than pre-checking availability.

        Returns:
            True if LLM provider can be reached, False otherwise.
        """
        if not self._llm_provider_instance:
            return False
        try:
            return await self._llm_provider_instance.is_available()
        except Exception as e:
            logger.warning(f"LLM availability check failed: {e}")
            return False

    async def get_full_status(self) -> dict:
        """Get comprehensive status including LLM availability.

        Returns:
            Dictionary with all status information.
        """
        index_status = self.get_index_status()

        # Check LLM availability (for status reporting)
        llm_available = await self.check_llm_available()

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
            "llm_provider": self._llm_provider_name,
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
        source_collection = self._data.get_collection_by_name(collection_name)
        if not source_collection:
            return None

        # Count documents in source collection
        doc_count = self._data.count_active_documents(source_collection.id)

        # Count embedded documents in source collection
        embedded_count = 0
        if self.vec_available:
            embedded_count = self._data.count_documents_with_embeddings(source_collection.id)

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
            source_collection = self._data.get_collection_by_name(collection_name)
            if not source_collection:
                return {"error": f"Source collection '{collection_name}' not found"}
            source_collection_id = source_collection.id

        # Documents missing FTS entries
        missing_fts_count = self._data.count_documents_missing_fts(source_collection_id)
        missing_fts_paths = self._data.list_paths_missing_fts(source_collection_id, limit)

        # Documents missing embeddings
        missing_vec_count = self._data.count_documents_missing_embeddings(source_collection_id)
        missing_vec_paths = self._data.list_paths_missing_embeddings(source_collection_id, limit)

        # Orphaned embeddings (no active documents)
        orphan_vec_count = self._data.count_orphaned_embeddings()

        # Orphaned FTS entries (no active documents)
        orphan_fts_count = self._data.count_orphaned_fts()

        return {
            "collection": collection_name,
            "missing_fts_count": missing_fts_count,
            "missing_fts_paths": missing_fts_paths,
            "missing_vectors_count": missing_vec_count,
            "missing_vectors_paths": missing_vec_paths,
            "orphan_vectors_count": orphan_vec_count,
            "orphan_fts_count": orphan_fts_count,
        }
