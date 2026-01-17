"""Source collection CRUD operations for PMD."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import SourceCollectionExistsError, SourceCollectionNotFoundError
from ...core.types import SourceCollection
from ..database import Database


class SourceCollectionRepository:
    """Repository for source collection operations."""

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def list_all(self) -> list[SourceCollection]:
        """Get all source collections.

        Returns:
            List of all SourceCollection objects.
        """
        cursor = self.db.execute("SELECT * FROM source_collections ORDER BY name")
        return [self._row_to_source_collection(row) for row in cursor.fetchall()]

    def get_by_name(self, name: str) -> SourceCollection | None:
        """Get source collection by name.

        Args:
            name: Source collection name to search for.

        Returns:
            SourceCollection object if found, None otherwise.
        """
        cursor = self.db.execute("SELECT * FROM source_collections WHERE name = ?", (name,))
        row = cursor.fetchone()
        return self._row_to_source_collection(row) if row else None

    def get_by_id(self, source_collection_id: int) -> SourceCollection | None:
        """Get source collection by ID.

        Args:
            source_collection_id: Source collection ID to search for.

        Returns:
            SourceCollection object if found, None otherwise.
        """
        cursor = self.db.execute("SELECT * FROM source_collections WHERE id = ?", (source_collection_id,))
        row = cursor.fetchone()
        return self._row_to_source_collection(row) if row else None

    def create(
        self,
        name: str,
        pwd: str,
        glob_patterns: list[str] | str = "**/*.md",
        source_type: str = "filesystem",
        source_config: dict[str, Any] | None = None,
    ) -> SourceCollection:
        """Create a new source collection.

        Args:
            name: Unique name for the source collection.
            pwd: Base directory path (filesystem) or base URI (other sources).
            glob_patterns: File pattern(s) to match. Can be a single pattern string
                or a list of patterns. Use ! prefix for exclusion patterns.
            source_type: Type of source ('filesystem', 'http', 'entity').
            source_config: Source-specific configuration as JSON-serializable dict.

        Returns:
            Created SourceCollection object.

        Raises:
            SourceCollectionExistsError: If source collection with this name already exists.
        """
        # Normalize to list
        if isinstance(glob_patterns, str):
            patterns_list = [glob_patterns]
        else:
            patterns_list = list(glob_patterns)

        # Use first pattern for legacy glob_pattern column (for display)
        primary_pattern = patterns_list[0] if patterns_list else "**/*.md"

        logger.debug(
            f"Creating source collection: name={name!r}, pwd={pwd!r}, "
            f"patterns={patterns_list!r}, source_type={source_type!r}"
        )

        if self.get_by_name(name):
            raise SourceCollectionExistsError(f"Source collection '{name}' already exists")

        now = datetime.utcnow().isoformat()

        # Merge glob_patterns into source_config
        merged_config = dict(source_config) if source_config else {}
        merged_config["glob_patterns"] = patterns_list
        source_config_json = json.dumps(merged_config)

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO source_collections
                (name, pwd, glob_pattern, source_type, source_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, pwd, primary_pattern, source_type, source_config_json, now, now),
            )
            source_collection_id = cursor.lastrowid

        logger.info(f"Source collection created: id={source_collection_id}, name={name!r}, type={source_type!r}")

        return SourceCollection(
            id=source_collection_id,  # type: ignore
            name=name,
            pwd=pwd,
            glob_pattern=primary_pattern,
            source_type=source_type,
            source_config=merged_config,
            created_at=now,
            updated_at=now,
        )

    def remove(self, source_collection_id: int) -> tuple[int, int]:
        """Remove a source collection and clean up associated data.

        Args:
            source_collection_id: ID of source collection to remove.

        Returns:
            Tuple of (documents_deleted, orphaned_hashes_cleaned).

        Raises:
            SourceCollectionNotFoundError: If source collection does not exist.
        """
        source_collection = self.get_by_id(source_collection_id)
        if not source_collection:
            raise SourceCollectionNotFoundError(f"Source collection {source_collection_id} not found")

        logger.debug(f"Removing source collection: id={source_collection_id}, name={source_collection.name!r}")

        with self.db.transaction() as cursor:
            # Get count of documents to delete
            cursor.execute("SELECT COUNT(*) FROM documents WHERE source_collection_id = ?", (source_collection_id,))
            docs_count = cursor.fetchone()[0]

            # Delete documents
            cursor.execute("DELETE FROM documents WHERE source_collection_id = ?", (source_collection_id,))

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

            # Delete the source collection
            cursor.execute("DELETE FROM source_collections WHERE id = ?", (source_collection_id,))

        logger.info(f"Source collection removed: name={source_collection.name!r}, docs={docs_count}, orphans={len(orphaned_hashes)}")

        return (docs_count, len(orphaned_hashes))

    def rename(self, source_collection_id: int, new_name: str) -> None:
        """Rename a source collection.

        Args:
            source_collection_id: ID of source collection to rename.
            new_name: New name for the source collection.

        Raises:
            SourceCollectionNotFoundError: If source collection does not exist.
            SourceCollectionExistsError: If new name already exists.
        """
        source_collection = self.get_by_id(source_collection_id)
        if not source_collection:
            raise SourceCollectionNotFoundError(f"Source collection {source_collection_id} not found")

        if new_name != source_collection.name and self.get_by_name(new_name):
            raise SourceCollectionExistsError(f"Source collection '{new_name}' already exists")

        logger.debug(f"Renaming source collection: {source_collection.name!r} -> {new_name!r}")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            cursor.execute(
                "UPDATE source_collections SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, now, source_collection_id),
            )

        logger.info(f"Source collection renamed: {source_collection.name!r} -> {new_name!r}")

    def update_path(
        self, source_collection_id: int, pwd: str, glob_pattern: str | None = None
    ) -> None:
        """Update a source collection's path and/or glob pattern.

        Args:
            source_collection_id: ID of source collection to update.
            pwd: New base directory path.
            glob_pattern: New glob pattern (optional).

        Raises:
            SourceCollectionNotFoundError: If source collection does not exist.
        """
        source_collection = self.get_by_id(source_collection_id)
        if not source_collection:
            raise SourceCollectionNotFoundError(f"Source collection {source_collection_id} not found")

        logger.debug(f"Updating source collection path: name={source_collection.name!r}, pwd={pwd!r}, pattern={glob_pattern!r}")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            if glob_pattern:
                cursor.execute(
                    "UPDATE source_collections SET pwd = ?, glob_pattern = ?, updated_at = ? WHERE id = ?",
                    (pwd, glob_pattern, now, source_collection_id),
                )
            else:
                cursor.execute(
                    "UPDATE source_collections SET pwd = ?, updated_at = ? WHERE id = ?",
                    (pwd, now, source_collection_id),
                )

        logger.info(f"Source collection path updated: name={source_collection.name!r}")

    @staticmethod
    def _row_to_source_collection(row) -> SourceCollection:
        """Convert database row to SourceCollection object.

        Args:
            row: Database row from sqlite3.Row.

        Returns:
            SourceCollection object.
        """
        # Handle source_config JSON parsing
        source_config_json = row["source_config"] if "source_config" in row.keys() else None
        source_config = json.loads(source_config_json) if source_config_json else None

        # Handle source_type with default for backward compatibility
        source_type = row["source_type"] if "source_type" in row.keys() else "filesystem"

        return SourceCollection(
            id=row["id"],
            name=row["name"],
            pwd=row["pwd"],
            glob_pattern=row["glob_pattern"],
            source_type=source_type or "filesystem",
            source_config=source_config,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Backwards compatibility alias (deprecated)
CollectionRepository = SourceCollectionRepository
