"""Content storage repository for PMD.

This module provides repository access to the content-addressable storage table.

Classes:
    ContentRepository: Repository for content table operations including orphan cleanup.
"""

from loguru import logger

from .database import Database


class ContentRepository:
    """Repository for content-addressable storage operations.

    The content table stores document content by hash, allowing deduplication
    across multiple documents. This repository provides methods for:
    - Counting and cleaning up orphaned content (not referenced by active documents)
    - Content retrieval by hash

    Example:
        >>> repo = ContentRepository(db)
        >>> orphan_count = repo.count_orphaned()
        >>> if orphan_count > 0:
        ...     deleted = repo.delete_orphaned()
        ...     print(f"Cleaned up {deleted} orphaned content entries")
    """

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def count_orphaned(self) -> int:
        """Count content entries not referenced by any active document.

        Returns:
            Number of orphaned content entries.
        """
        cursor = self.db.execute(
            """
            SELECT COUNT(*) as count FROM content
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            """
        )
        return cursor.fetchone()["count"]

    def delete_orphaned(self) -> int:
        """Delete content entries not referenced by any active document.

        Returns:
            Number of content entries deleted.
        """
        count = self.count_orphaned()
        if count > 0:
            logger.debug(f"Deleting {count} orphaned content entries")
            self.db.execute(
                """
                DELETE FROM content
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
                """
            )
            logger.info(f"Deleted {count} orphaned content entries")
        return count

    def get_by_hash(self, hash_value: str) -> str | None:
        """Retrieve content by its hash.

        Args:
            hash_value: SHA256 hash of content.

        Returns:
            Content string if found, None otherwise.
        """
        cursor = self.db.execute(
            "SELECT doc FROM content WHERE hash = ?",
            (hash_value,),
        )
        row = cursor.fetchone()
        return row["doc"] if row else None

    def exists(self, hash_value: str) -> bool:
        """Check if content exists by hash.

        Args:
            hash_value: SHA256 hash to check.

        Returns:
            True if content exists, False otherwise.
        """
        cursor = self.db.execute(
            "SELECT 1 FROM content WHERE hash = ? LIMIT 1",
            (hash_value,),
        )
        return cursor.fetchone() is not None

    def count(self) -> int:
        """Count total content entries.

        Returns:
            Total number of content entries.
        """
        cursor = self.db.execute("SELECT COUNT(*) as count FROM content")
        return cursor.fetchone()["count"]
