"""Collection CRUD operations for PMD."""

from datetime import datetime
from pathlib import Path

from ..core.exceptions import CollectionExistsError, CollectionNotFoundError
from ..core.types import Collection
from .database import Database


class CollectionRepository:
    """Repository for collection operations."""

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def list_all(self) -> list[Collection]:
        """Get all collections.

        Returns:
            List of all Collection objects.
        """
        cursor = self.db.execute("SELECT * FROM collections ORDER BY name")
        return [self._row_to_collection(row) for row in cursor.fetchall()]

    def get_by_name(self, name: str) -> Collection | None:
        """Get collection by name.

        Args:
            name: Collection name to search for.

        Returns:
            Collection object if found, None otherwise.
        """
        cursor = self.db.execute("SELECT * FROM collections WHERE name = ?", (name,))
        row = cursor.fetchone()
        return self._row_to_collection(row) if row else None

    def get_by_id(self, collection_id: int) -> Collection | None:
        """Get collection by ID.

        Args:
            collection_id: Collection ID to search for.

        Returns:
            Collection object if found, None otherwise.
        """
        cursor = self.db.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
        row = cursor.fetchone()
        return self._row_to_collection(row) if row else None

    def create(
        self, name: str, pwd: str, glob_pattern: str = "**/*.md"
    ) -> Collection:
        """Create a new collection.

        Args:
            name: Unique name for the collection.
            pwd: Base directory path to index.
            glob_pattern: File pattern to match (default: **/*.md).

        Returns:
            Created Collection object.

        Raises:
            CollectionExistsError: If collection with this name already exists.
        """
        if self.get_by_name(name):
            raise CollectionExistsError(f"Collection '{name}' already exists")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, pwd, glob_pattern, now, now),
            )
            collection_id = cursor.lastrowid

        return Collection(
            id=collection_id,
            name=name,
            pwd=pwd,
            glob_pattern=glob_pattern,
            created_at=now,
            updated_at=now,
        )

    def remove(self, collection_id: int) -> tuple[int, int]:
        """Remove a collection and clean up associated data.

        Args:
            collection_id: ID of collection to remove.

        Returns:
            Tuple of (documents_deleted, orphaned_hashes_cleaned).

        Raises:
            CollectionNotFoundError: If collection does not exist.
        """
        collection = self.get_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")

        with self.db.transaction() as cursor:
            # Get count of documents to delete
            cursor.execute("SELECT COUNT(*) FROM documents WHERE collection_id = ?", (collection_id,))
            docs_count = cursor.fetchone()[0]

            # Delete documents
            cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))

            # Delete path contexts
            cursor.execute("DELETE FROM path_contexts WHERE collection_id = ?", (collection_id,))

            # Find and delete orphaned content hashes
            cursor.execute(
                """
                SELECT DISTINCT hash FROM content
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents)
                """
            )
            orphaned_hashes = [row[0] for row in cursor.fetchall()]

            for hash_val in orphaned_hashes:
                cursor.execute("DELETE FROM content WHERE hash = ?", (hash_val,))
                cursor.execute("DELETE FROM content_vectors WHERE hash = ?", (hash_val,))

            # Delete the collection
            cursor.execute("DELETE FROM collections WHERE id = ?", (collection_id,))

        return (docs_count, len(orphaned_hashes))

    def rename(self, collection_id: int, new_name: str) -> None:
        """Rename a collection.

        Args:
            collection_id: ID of collection to rename.
            new_name: New name for the collection.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            CollectionExistsError: If new name already exists.
        """
        collection = self.get_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")

        if new_name != collection.name and self.get_by_name(new_name):
            raise CollectionExistsError(f"Collection '{new_name}' already exists")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            cursor.execute(
                "UPDATE collections SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, now, collection_id),
            )

    def update_collection_path(
        self, collection_id: int, pwd: str, glob_pattern: str | None = None
    ) -> None:
        """Update a collection's path and/or glob pattern.

        Args:
            collection_id: ID of collection to update.
            pwd: New base directory path.
            glob_pattern: New glob pattern (optional).

        Raises:
            CollectionNotFoundError: If collection does not exist.
        """
        collection = self.get_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            if glob_pattern:
                cursor.execute(
                    "UPDATE collections SET pwd = ?, glob_pattern = ?, updated_at = ? WHERE id = ?",
                    (pwd, glob_pattern, now, collection_id),
                )
            else:
                cursor.execute(
                    "UPDATE collections SET pwd = ?, updated_at = ? WHERE id = ?",
                    (pwd, now, collection_id),
                )

    @staticmethod
    def _row_to_collection(row: tuple) -> Collection:
        """Convert database row to Collection object.

        Args:
            row: Database row from sqlite3.Row.

        Returns:
            Collection object.
        """
        return Collection(
            id=row["id"],
            name=row["name"],
            pwd=row["pwd"],
            glob_pattern=row["glob_pattern"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
