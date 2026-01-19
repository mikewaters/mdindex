"""Resource CRUD operations for PMD.

This module provides the ResourceRepository class for managing resources
in the resources table. Resources track fetch/index lifecycle state
separately from the documents table.
"""

from datetime import datetime
from typing import Any

from loguru import logger

from ..database import Database
from ..models import IndexState, LoadStatus, ResourceModel


class ResourceRepository:
    """Repository for resource operations.

    Resources represent URIs that can be loaded and indexed independently.
    This separates fetch/index state tracking from document storage.
    """

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def upsert(
        self,
        collection_id: int,
        uri: str,
        *,
        resource_type: str | None = None,
        hash: str | None = None,
        content_ref: str | None = None,
        source_created_at: str | None = None,
        source_modified_at: str | None = None,
        loaded_at: str | None = None,
        load_method: str | None = None,
        load_status: LoadStatus | str | None = None,
        load_error: str | None = None,
        indexed_at: str | None = None,
        index_state: IndexState | str | None = None,
        index_method: str | None = None,
        index_error: str | None = None,
        resource_metadata: dict[str, Any] | str | None = None,
    ) -> ResourceModel:
        """Insert or update a resource by (collection_id, uri).

        If a resource with the given collection_id and uri exists, it will be
        updated with the provided attributes. Otherwise, a new resource is created.

        Args:
            collection_id: ID of the source collection.
            uri: Unique resource URI within the collection.
            resource_type: Optional type hint (e.g., 'markdown', 'pdf').
            hash: Content hash after loading.
            content_ref: Reference to stored content.
            source_created_at: When resource was created at source.
            source_modified_at: When resource was last modified at source.
            loaded_at: When content was successfully loaded.
            load_method: Method used for loading (e.g., 'http', 'filesystem').
            load_status: Current load status (pending/loaded/error/skipped).
            load_error: Error message if load_status is 'error'.
            indexed_at: When resource was successfully indexed.
            index_state: Current index state (pending/indexed/error/skipped).
            index_method: Method used for indexing.
            index_error: Error message if index_state is 'error'.
            resource_metadata: JSON-serializable metadata dict or JSON string.

        Returns:
            The created or updated ResourceModel.
        """
        import json

        now = datetime.utcnow().isoformat()

        # Normalize enum values to strings
        if isinstance(load_status, LoadStatus):
            load_status = load_status.value
        if isinstance(index_state, IndexState):
            index_state = index_state.value

        # Normalize metadata to JSON string
        if isinstance(resource_metadata, dict):
            resource_metadata = json.dumps(resource_metadata)

        logger.debug(
            f"Upserting resource: collection_id={collection_id}, uri={uri!r}"
        )

        with self.db.transaction() as cursor:
            # Check if resource exists
            cursor.execute(
                """
                SELECT id FROM resources
                WHERE source_collection_id = ? AND uri = ?
                """,
                (collection_id, uri),
            )
            existing = cursor.fetchone()

            if existing:
                # Build dynamic UPDATE statement for provided fields
                resource_id = existing["id"]
                updates = ["updated_at = ?"]
                params: list[Any] = [now]

                if resource_type is not None:
                    updates.append("resource_type = ?")
                    params.append(resource_type)
                if hash is not None:
                    updates.append("hash = ?")
                    params.append(hash)
                if content_ref is not None:
                    updates.append("content_ref = ?")
                    params.append(content_ref)
                if source_created_at is not None:
                    updates.append("source_created_at = ?")
                    params.append(source_created_at)
                if source_modified_at is not None:
                    updates.append("source_modified_at = ?")
                    params.append(source_modified_at)
                if loaded_at is not None:
                    updates.append("loaded_at = ?")
                    params.append(loaded_at)
                if load_method is not None:
                    updates.append("load_method = ?")
                    params.append(load_method)
                if load_status is not None:
                    updates.append("load_status = ?")
                    params.append(load_status)
                if load_error is not None:
                    updates.append("load_error = ?")
                    params.append(load_error)
                if indexed_at is not None:
                    updates.append("indexed_at = ?")
                    params.append(indexed_at)
                if index_state is not None:
                    updates.append("index_state = ?")
                    params.append(index_state)
                if index_method is not None:
                    updates.append("index_method = ?")
                    params.append(index_method)
                if index_error is not None:
                    updates.append("index_error = ?")
                    params.append(index_error)
                if resource_metadata is not None:
                    updates.append("metadata = ?")
                    params.append(resource_metadata)

                params.append(resource_id)
                cursor.execute(
                    f"UPDATE resources SET {', '.join(updates)} WHERE id = ?",
                    tuple(params),
                )
                logger.debug(f"Resource updated: id={resource_id}, uri={uri!r}")
            else:
                # Insert new resource with defaults
                cursor.execute(
                    """
                    INSERT INTO resources (
                        source_collection_id, uri, resource_type, hash, content_ref,
                        source_created_at, source_modified_at,
                        loaded_at, load_method, load_status, load_error,
                        indexed_at, index_state, index_method, index_error,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection_id,
                        uri,
                        resource_type,
                        hash,
                        content_ref,
                        source_created_at,
                        source_modified_at,
                        loaded_at,
                        load_method,
                        load_status or LoadStatus.PENDING.value,
                        load_error,
                        indexed_at,
                        index_state or IndexState.PENDING.value,
                        index_method,
                        index_error,
                        resource_metadata,
                        now,
                        now,
                    ),
                )
                resource_id = cursor.lastrowid
                logger.debug(f"Resource created: id={resource_id}, uri={uri!r}")

        # Fetch and return the complete resource
        return self.get_by_id(resource_id)  # type: ignore

    def get_by_uri(self, collection_id: int, uri: str) -> ResourceModel | None:
        """Find a resource by collection_id and uri.

        Args:
            collection_id: ID of the source collection.
            uri: Resource URI within the collection.

        Returns:
            ResourceModel if found, None otherwise.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM resources
            WHERE source_collection_id = ? AND uri = ?
            """,
            (collection_id, uri),
        )
        row = cursor.fetchone()
        return self._row_to_resource(row) if row else None

    def get_by_id(self, resource_id: int) -> ResourceModel | None:
        """Find a resource by id.

        Args:
            resource_id: Unique resource identifier.

        Returns:
            ResourceModel if found, None otherwise.
        """
        cursor = self.db.execute(
            "SELECT * FROM resources WHERE id = ?",
            (resource_id,),
        )
        row = cursor.fetchone()
        return self._row_to_resource(row) if row else None

    def list_by_collection(
        self,
        collection_id: int,
        *,
        status: LoadStatus | str | None = None,
        state: IndexState | str | None = None,
    ) -> list[ResourceModel]:
        """List resources for a collection with optional filters.

        Args:
            collection_id: ID of the source collection.
            status: Optional load_status filter.
            state: Optional index_state filter.

        Returns:
            List of ResourceModel objects matching the criteria.
        """
        # Normalize enum values
        if isinstance(status, LoadStatus):
            status = status.value
        if isinstance(state, IndexState):
            state = state.value

        # Build query with optional filters
        conditions = ["source_collection_id = ?"]
        params: list[Any] = [collection_id]

        if status is not None:
            conditions.append("load_status = ?")
            params.append(status)
        if state is not None:
            conditions.append("index_state = ?")
            params.append(state)

        where_clause = " AND ".join(conditions)
        cursor = self.db.execute(
            f"SELECT * FROM resources WHERE {where_clause} ORDER BY uri",
            tuple(params),
        )
        return [self._row_to_resource(row) for row in cursor.fetchall()]

    def delete(self, resource_id: int) -> bool:
        """Delete a resource by ID.

        Args:
            resource_id: Unique resource identifier.

        Returns:
            True if resource was deleted, False if not found.
        """
        cursor = self.db.execute(
            "SELECT id FROM resources WHERE id = ?",
            (resource_id,),
        )
        if not cursor.fetchone():
            logger.debug(f"Resource not found for deletion: id={resource_id}")
            return False

        logger.debug(f"Deleting resource: id={resource_id}")
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM resources WHERE id = ?", (resource_id,))

        logger.debug(f"Resource deleted: id={resource_id}")
        return True

    def count_by_collection(
        self,
        collection_id: int,
        *,
        status: LoadStatus | str | None = None,
        state: IndexState | str | None = None,
    ) -> int:
        """Count resources in a collection with optional filters.

        Args:
            collection_id: ID of the source collection.
            status: Optional load_status filter.
            state: Optional index_state filter.

        Returns:
            Number of resources matching the criteria.
        """
        # Normalize enum values
        if isinstance(status, LoadStatus):
            status = status.value
        if isinstance(state, IndexState):
            state = state.value

        conditions = ["source_collection_id = ?"]
        params: list[Any] = [collection_id]

        if status is not None:
            conditions.append("load_status = ?")
            params.append(status)
        if state is not None:
            conditions.append("index_state = ?")
            params.append(state)

        where_clause = " AND ".join(conditions)
        cursor = self.db.execute(
            f"SELECT COUNT(*) as count FROM resources WHERE {where_clause}",
            tuple(params),
        )
        return cursor.fetchone()["count"]

    # -------------------------------------------------------------------------
    # State Transition Methods
    # -------------------------------------------------------------------------

    def mark_loading(self, resource_id: int) -> None:
        """Mark a resource as currently loading.

        Sets load_status='loading' and clears any previous load error.
        Also updates the updated_at timestamp.

        Args:
            resource_id: Resource ID to update.
        """
        now = datetime.utcnow().isoformat()
        logger.debug(f"Marking resource as loading: id={resource_id}")

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE resources SET
                    load_status = 'loading',
                    load_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, resource_id),
            )

    def mark_loaded(
        self,
        resource_id: int,
        hash: str,
        content_ref: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a resource as successfully loaded.

        Sets load_status='loaded', updates hash and content_ref,
        records the loaded_at timestamp, and updates updated_at.

        Args:
            resource_id: Resource ID to update.
            hash: Content hash of the loaded content.
            content_ref: Reference to stored content (e.g., cache path).
            metadata: Optional metadata to merge with existing metadata.
        """
        import json

        now = datetime.utcnow().isoformat()
        logger.debug(f"Marking resource as loaded: id={resource_id}, hash={hash[:12]}...")

        with self.db.transaction() as cursor:
            if metadata:
                # Fetch existing metadata to merge
                cursor.execute(
                    "SELECT metadata FROM resources WHERE id = ?",
                    (resource_id,),
                )
                row = cursor.fetchone()
                existing_metadata: dict[str, Any] = {}
                if row and row["metadata"]:
                    existing_metadata = json.loads(row["metadata"])
                existing_metadata.update(metadata)
                metadata_json = json.dumps(existing_metadata)

                cursor.execute(
                    """
                    UPDATE resources SET
                        load_status = 'loaded',
                        hash = ?,
                        content_ref = ?,
                        loaded_at = ?,
                        load_error = NULL,
                        metadata = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (hash, content_ref, now, metadata_json, now, resource_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE resources SET
                        load_status = 'loaded',
                        hash = ?,
                        content_ref = ?,
                        loaded_at = ?,
                        load_error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (hash, content_ref, now, now, resource_id),
                )

    def mark_load_failed(self, resource_id: int, error: str) -> None:
        """Mark a resource as failed to load.

        Sets load_status='error', records the error message,
        and updates updated_at timestamp.

        Args:
            resource_id: Resource ID to update.
            error: Error message describing the failure.
        """
        now = datetime.utcnow().isoformat()
        logger.debug(f"Marking resource as load failed: id={resource_id}, error={error!r}")

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE resources SET
                    load_status = 'error',
                    load_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error, now, resource_id),
            )

    def mark_indexing(self, resource_id: int) -> None:
        """Mark a resource as currently being indexed.

        Sets index_state='indexing', clears any previous index error,
        and updates updated_at timestamp.

        Args:
            resource_id: Resource ID to update.
        """
        now = datetime.utcnow().isoformat()
        logger.debug(f"Marking resource as indexing: id={resource_id}")

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE resources SET
                    index_state = 'indexing',
                    index_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, resource_id),
            )

    def mark_indexed(
        self,
        resource_id: int,
        method: str | None = None,
    ) -> None:
        """Mark a resource as successfully indexed.

        Sets index_state='indexed', optionally records the method,
        records the indexed_at timestamp, and updates updated_at.

        Args:
            resource_id: Resource ID to update.
            method: Optional indexing method used (e.g., 'fts', 'embedding').
        """
        now = datetime.utcnow().isoformat()
        logger.debug(f"Marking resource as indexed: id={resource_id}, method={method!r}")

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE resources SET
                    index_state = 'indexed',
                    index_method = COALESCE(?, index_method),
                    indexed_at = ?,
                    index_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (method, now, now, resource_id),
            )

    def mark_index_failed(self, resource_id: int, error: str) -> None:
        """Mark a resource as failed to index.

        Sets index_state='error', records the error message,
        and updates updated_at timestamp.

        Args:
            resource_id: Resource ID to update.
            error: Error message describing the failure.
        """
        now = datetime.utcnow().isoformat()
        logger.debug(f"Marking resource as index failed: id={resource_id}, error={error!r}")

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE resources SET
                    index_state = 'error',
                    index_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error, now, resource_id),
            )

    def mark_stale(
        self,
        resource_id: int,
        reason: str,  # Literal['load', 'index']
    ) -> None:
        """Mark a resource as stale, requiring re-processing.

        Sets load_status='stale' or index_state='stale' based on reason,
        and updates updated_at timestamp.

        Args:
            resource_id: Resource ID to update.
            reason: Which aspect is stale - 'load' for content needs
                re-fetching, 'index' for content needs re-indexing.

        Raises:
            ValueError: If reason is not 'load' or 'index'.
        """
        if reason not in ("load", "index"):
            raise ValueError(f"reason must be 'load' or 'index', got {reason!r}")

        now = datetime.utcnow().isoformat()
        logger.debug(f"Marking resource as stale: id={resource_id}, reason={reason!r}")

        with self.db.transaction() as cursor:
            if reason == "load":
                cursor.execute(
                    """
                    UPDATE resources SET
                        load_status = 'stale',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, resource_id),
                )
            else:  # reason == "index"
                cursor.execute(
                    """
                    UPDATE resources SET
                        index_state = 'stale',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, resource_id),
                )

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def list_needing_index(self, collection_id: int) -> list[ResourceModel]:
        """List resources that need indexing.

        Returns resources where load_status='loaded' and
        index_state in ('pending', 'stale').

        Args:
            collection_id: Source collection ID.

        Returns:
            List of ResourceModel objects needing indexing.
        """
        cursor = self.db.execute(
            """
            SELECT * FROM resources
            WHERE source_collection_id = ?
              AND load_status = 'loaded'
              AND index_state IN ('pending', 'stale')
            ORDER BY uri
            """,
            (collection_id,),
        )
        return [self._row_to_resource(row) for row in cursor.fetchall()]

    def delete_orphaned(self, collection_id: int, valid_uris: set[str]) -> int:
        """Delete resources not in the valid URIs set.

        Used during sync to clean up resources that no longer exist
        at the source.

        Args:
            collection_id: Source collection ID.
            valid_uris: Set of URIs that are currently valid.

        Returns:
            Number of resources deleted.
        """
        if not valid_uris:
            # Delete all resources for this collection
            with self.db.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM resources WHERE source_collection_id = ?",
                    (collection_id,),
                )
                deleted = cursor.rowcount
            logger.debug(f"Deleted all {deleted} orphaned resources for collection {collection_id}")
            return deleted

        # Build placeholders for the IN clause
        placeholders = ", ".join("?" for _ in valid_uris)

        with self.db.transaction() as cursor:
            cursor.execute(
                f"""
                DELETE FROM resources
                WHERE source_collection_id = ?
                  AND uri NOT IN ({placeholders})
                """,
                (collection_id, *valid_uris),
            )
            deleted = cursor.rowcount

        logger.debug(f"Deleted {deleted} orphaned resources for collection {collection_id}")
        return deleted

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_to_resource(row) -> ResourceModel:
        """Convert database row to ResourceModel object.

        Args:
            row: Database row from sqlite3.Row or LegacyRow.

        Returns:
            ResourceModel object.
        """
        return ResourceModel(
            id=row["id"],
            source_collection_id=row["source_collection_id"],
            uri=row["uri"],
            resource_type=row["resource_type"],
            hash=row["hash"],
            content_ref=row["content_ref"],
            source_created_at=row["source_created_at"],
            source_modified_at=row["source_modified_at"],
            loaded_at=row["loaded_at"],
            load_method=row["load_method"],
            load_status=row["load_status"],
            load_error=row["load_error"],
            indexed_at=row["indexed_at"],
            index_state=row["index_state"],
            index_method=row["index_method"],
            index_error=row["index_error"],
            resource_metadata=row["metadata"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
