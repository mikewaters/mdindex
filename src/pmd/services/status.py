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
        collection_id = None
        if collection_name:
            collection = self._container.collection_repo.get_by_name(collection_name)
            if not collection:
                return {"error": f"Collection '{collection_name}' not found"}
            collection_id = collection.id

        params: tuple = ()
        collection_filter = ""
        if collection_id is not None:
            collection_filter = "AND d.collection_id = ?"
            params = (collection_id,)

        # Documents missing FTS entries
        missing_fts_count = self._container.db.execute(
            f"""
            SELECT COUNT(*) as count
            FROM documents d
            LEFT JOIN documents_fts fts ON fts.rowid = d.id
            WHERE d.active = 1 {collection_filter}
            AND fts.rowid IS NULL
            """,
            params,
        ).fetchone()["count"]

        missing_fts_paths = self._container.db.execute(
            f"""
            SELECT d.path
            FROM documents d
            LEFT JOIN documents_fts fts ON fts.rowid = d.id
            WHERE d.active = 1 {collection_filter}
            AND fts.rowid IS NULL
            ORDER BY d.path
            LIMIT ?
            """,
            params + (limit,),
        ).fetchall()

        # Documents missing embeddings
        missing_vec_count = self._container.db.execute(
            f"""
            SELECT COUNT(DISTINCT d.id) as count
            FROM documents d
            LEFT JOIN content_vectors cv ON cv.hash = d.hash
            WHERE d.active = 1 {collection_filter}
            AND cv.hash IS NULL
            """,
            params,
        ).fetchone()["count"]

        missing_vec_paths = self._container.db.execute(
            f"""
            SELECT d.path
            FROM documents d
            LEFT JOIN content_vectors cv ON cv.hash = d.hash
            WHERE d.active = 1 {collection_filter}
            AND cv.hash IS NULL
            ORDER BY d.path
            LIMIT ?
            """,
            params + (limit,),
        ).fetchall()

        # Orphaned embeddings (no active documents)
        orphan_vec_count = self._container.db.execute(
            """
            SELECT COUNT(DISTINCT cv.hash) as count
            FROM content_vectors cv
            LEFT JOIN documents d ON d.hash = cv.hash AND d.active = 1
            WHERE d.hash IS NULL
            """
        ).fetchone()["count"]

        # Orphaned FTS entries (no active documents)
        orphan_fts_count = self._container.db.execute(
            """
            SELECT COUNT(*) as count
            FROM documents_fts fts
            LEFT JOIN documents d ON d.id = fts.rowid AND d.active = 1
            WHERE d.id IS NULL
            """
        ).fetchone()["count"]

        return {
            "collection": collection_name,
            "missing_fts_count": missing_fts_count,
            "missing_fts_paths": [row["path"] for row in missing_fts_paths],
            "missing_vectors_count": missing_vec_count,
            "missing_vectors_paths": [row["path"] for row in missing_vec_paths],
            "orphan_vectors_count": orphan_vec_count,
            "orphan_fts_count": orphan_fts_count,
        }
