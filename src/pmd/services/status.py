"""Status service for index health and status reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..core.types import IndexStatus

if TYPE_CHECKING:
    from .container import ServiceContainer


class StatusService:
    """Service for index status and health reporting.

    This service provides information about:
    - Index statistics (document counts, sizes)
    - Collection information
    - LLM provider availability

    Example:

        async with ServiceContainer(config) as services:
            status = services.status.get_index_status()
            print(f"Documents: {status.total_documents}")
            print(f"Collections: {len(status.collections)}")
    """

    def __init__(self, container: "ServiceContainer"):
        """Initialize StatusService.

        Args:
            container: Service container with shared resources.
        """
        self._container = container

    def get_index_status(self) -> IndexStatus:
        """Get current index status.

        Returns:
            IndexStatus with collection and document information.
        """
        logger.debug("Getting index status")

        collections = self._container.collection_repo.list_all()

        # Count total documents
        cursor = self._container.db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE active = 1"
        )
        total_documents = cursor.fetchone()["count"]

        # Count embedded documents (documents with at least one embedding)
        embedded_documents = 0
        if self._container.vec_available:
            cursor = self._container.db.execute(
                """
                SELECT COUNT(DISTINCT hash) as count FROM content_vectors
                """
            )
            embedded_documents = cursor.fetchone()["count"]

        # Get database file size
        try:
            index_size_bytes = self._container.config.db_path.stat().st_size
        except (OSError, AttributeError):
            index_size_bytes = 0

        # Count embeddings (for cache entries metric)
        cache_entries = 0
        if self._container.vec_available:
            cursor = self._container.db.execute(
                "SELECT COUNT(*) as count FROM content_vectors"
            )
            cache_entries = cursor.fetchone()["count"]

        logger.debug(
            f"Index status: collections={len(collections)}, "
            f"documents={total_documents}, embedded={embedded_documents}"
        )

        return IndexStatus(
            collections=collections,
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
        llm_available = await self._container.is_llm_available()

        return {
            "collections_count": len(index_status.collections),
            "collections": [
                {
                    "name": c.name,
                    "path": c.pwd,
                    "glob_pattern": c.glob_pattern,
                }
                for c in index_status.collections
            ],
            "total_documents": index_status.total_documents,
            "embedded_documents": index_status.embedded_documents,
            "index_size_bytes": index_status.index_size_bytes,
            "embeddings_count": index_status.cache_entries,
            "database_path": str(self._container.config.db_path),
            "llm_provider": self._container.config.llm_provider,
            "llm_available": llm_available,
            "vec_available": self._container.vec_available,
        }

    def get_collection_stats(self, collection_name: str) -> dict | None:
        """Get statistics for a specific collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Dictionary with collection statistics, or None if not found.
        """
        collection = self._container.collection_repo.get_by_name(collection_name)
        if not collection:
            return None

        # Count documents in collection
        cursor = self._container.db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE collection_id = ? AND active = 1",
            (collection.id,),
        )
        doc_count = cursor.fetchone()["count"]

        # Count embedded documents in collection
        embedded_count = 0
        if self._container.vec_available:
            cursor = self._container.db.execute(
                """
                SELECT COUNT(DISTINCT d.hash) as count
                FROM documents d
                JOIN content_vectors cv ON d.hash = cv.hash
                WHERE d.collection_id = ? AND d.active = 1
                """,
                (collection.id,),
            )
            embedded_count = cursor.fetchone()["count"]

        return {
            "name": collection.name,
            "path": collection.pwd,
            "glob_pattern": collection.glob_pattern,
            "documents": doc_count,
            "embedded": embedded_count,
            "created_at": collection.created_at,
            "updated_at": collection.updated_at,
        }
