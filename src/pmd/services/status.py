"""Status service for index health and status reporting."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable

from loguru import logger

from ..core.types import IndexStatus
from ..app.types import (
    CollectionRepositoryProtocol,
    DatabaseProtocol,
)

if TYPE_CHECKING:
    from .container import ServiceContainer


class StatusService:
    """Service for index status and health reporting.

    This service provides information about:
    - Index statistics (document counts, sizes)
    - Collection information
    - LLM provider availability

    Example with explicit dependencies (recommended):

        status_service = StatusService(
            db=db,
            collection_repo=collection_repo,
            db_path=config.db_path,
        )
        status = status_service.get_index_status()

    Example with ServiceContainer (deprecated):

        async with ServiceContainer(config) as services:
            status = services.status.get_index_status()
    """

    def __init__(
        self,
        # Explicit dependencies (new API)
        db: DatabaseProtocol | None = None,
        collection_repo: CollectionRepositoryProtocol | None = None,
        db_path: Path | None = None,
        llm_provider: str = "unknown",
        llm_available_check: Callable[[], Awaitable[bool]] | None = None,
        # Deprecated: ServiceContainer
        container: "ServiceContainer | None" = None,
    ):
        """Initialize StatusService.

        Args:
            db: Database for direct SQL operations.
            collection_repo: Repository for collection operations.
            db_path: Path to the database file.
            llm_provider: Name of the LLM provider.
            llm_available_check: Async function to check if LLM is available.
            container: DEPRECATED. Use explicit dependencies instead.
        """
        # Support backward compatibility with container
        if container is not None:
            warnings.warn(
                "Passing 'container' to StatusService is deprecated. "
                "Use explicit dependencies instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._container = container
            self._db = container.db
            self._collection_repo = container.collection_repo
            self._db_path = container.config.db_path
            self._llm_provider = container.config.llm_provider
            self._llm_available_check = container.is_llm_available
        else:
            self._container = None
            if db is None or collection_repo is None:
                raise ValueError(
                    "StatusService requires db and collection_repo"
                )
            self._db = db
            self._collection_repo = collection_repo
            self._db_path = db_path
            self._llm_provider = llm_provider
            self._llm_available_check = llm_available_check

    @classmethod
    def from_container(cls, container: "ServiceContainer") -> "StatusService":
        """Create StatusService from a ServiceContainer.

        This is a convenience method for backward compatibility.
        Prefer using explicit dependencies in new code.

        Args:
            container: Service container with shared resources.

        Returns:
            StatusService instance.
        """
        return cls(
            db=container.db,
            collection_repo=container.collection_repo,
            db_path=container.config.db_path,
            llm_provider=container.config.llm_provider,
            llm_available_check=container.is_llm_available,
        )

    @property
    def vec_available(self) -> bool:
        """Check if vector storage is available."""
        return self._db.vec_available

    def get_index_status(self) -> IndexStatus:
        """Get current index status.

        Returns:
            IndexStatus with collection and document information.
        """
        logger.debug("Getting index status")

        collections = self._collection_repo.list_all()

        # Count total documents
        cursor = self._db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE active = 1"
        )
        total_documents = cursor.fetchone()["count"]

        # Count embedded documents (documents with at least one embedding)
        embedded_documents = 0
        if self.vec_available:
            cursor = self._db.execute(
                """
                SELECT COUNT(DISTINCT hash) as count FROM content_vectors
                """
            )
            embedded_documents = cursor.fetchone()["count"]

        # Get database file size
        try:
            index_size_bytes = self._db_path.stat().st_size if self._db_path else 0
        except (OSError, AttributeError):
            index_size_bytes = 0

        # Count embeddings (for cache entries metric)
        cache_entries = 0
        if self.vec_available:
            cursor = self._db.execute(
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
        llm_available = False
        if self._llm_available_check:
            llm_available = await self._llm_available_check()

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
        collection = self._collection_repo.get_by_name(collection_name)
        if not collection:
            return None

        # Count documents in collection
        cursor = self._db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE collection_id = ? AND active = 1",
            (collection.id,),
        )
        doc_count = cursor.fetchone()["count"]

        # Count embedded documents in collection
        embedded_count = 0
        if self.vec_available:
            cursor = self._db.execute(
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
            collection = self._collection_repo.get_by_name(collection_name)
            if not collection:
                return {"error": f"Collection '{collection_name}' not found"}
            collection_id = collection.id

        params: tuple = ()
        collection_filter = ""
        if collection_id is not None:
            collection_filter = "AND d.collection_id = ?"
            params = (collection_id,)

        # Documents missing FTS entries
        missing_fts_count = self._db.execute(
            f"""
            SELECT COUNT(*) as count
            FROM documents d
            LEFT JOIN documents_fts fts ON fts.rowid = d.id
            WHERE d.active = 1 {collection_filter}
            AND fts.rowid IS NULL
            """,
            params,
        ).fetchone()["count"]

        missing_fts_paths = self._db.execute(
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
        missing_vec_count = self._db.execute(
            f"""
            SELECT COUNT(DISTINCT d.id) as count
            FROM documents d
            LEFT JOIN content_vectors cv ON cv.hash = d.hash
            WHERE d.active = 1 {collection_filter}
            AND cv.hash IS NULL
            """,
            params,
        ).fetchone()["count"]

        missing_vec_paths = self._db.execute(
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
        orphan_vec_count = self._db.execute(
            """
            SELECT COUNT(DISTINCT cv.hash) as count
            FROM content_vectors cv
            LEFT JOIN documents d ON d.hash = cv.hash AND d.active = 1
            WHERE d.hash IS NULL
            """
        ).fetchone()["count"]

        # Orphaned FTS entries (no active documents)
        orphan_fts_count = self._db.execute(
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
