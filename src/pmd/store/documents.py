"""Document storage and retrieval for PMD."""

from datetime import datetime

from ..core.exceptions import DocumentNotFoundError
from ..core.types import DocumentResult
from ..utils.hashing import sha256_hash
from .database import Database


class DocumentRepository:
    """Repository for document operations with content-addressable storage."""

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def add_or_update(
        self, collection_id: int, path: str, title: str, content: str
    ) -> tuple[DocumentResult, bool]:
        """Add or update a document.

        Uses content-addressable storage: stores content in content table
        and references it from documents table via hash.

        Args:
            collection_id: ID of the collection.
            path: Document path relative to collection.
            title: Document title.
            content: Document content.

        Returns:
            Tuple of (DocumentResult, is_new) where is_new indicates if this
            was a new document vs an update.
        """
        hash_value = sha256_hash(content)
        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            # Ensure content exists in content-addressable storage
            cursor.execute(
                """
                INSERT OR IGNORE INTO content (hash, doc, created_at)
                VALUES (?, ?, ?)
                """,
                (hash_value, content, now),
            )

            # Check if document already exists
            cursor.execute(
                """
                SELECT id FROM documents
                WHERE collection_id = ? AND path = ?
                """,
                (collection_id, path),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing document
                cursor.execute(
                    """
                    UPDATE documents
                    SET title = ?, hash = ?, modified_at = ?, active = 1
                    WHERE collection_id = ? AND path = ?
                    """,
                    (title, hash_value, now, collection_id, path),
                )
                is_new = False
            else:
                # Insert new document
                cursor.execute(
                    """
                    INSERT INTO documents
                    (collection_id, path, title, hash, modified_at, active)
                    VALUES (?, ?, ?, ?, ?, 1)
                    """,
                    (collection_id, path, title, hash_value, now),
                )
                is_new = True

        return (
            DocumentResult(
                filepath=path,
                display_path=path,
                title=title,
                context=None,
                hash=hash_value,
                collection_id=collection_id,
                modified_at=now,
                body_length=len(content),
                body=content,
            ),
            is_new,
        )

    def get(self, collection_id: int, path: str) -> DocumentResult | None:
        """Retrieve a document by path.

        Args:
            collection_id: ID of the collection.
            path: Document path relative to collection.

        Returns:
            DocumentResult if found, None otherwise.
        """
        cursor = self.db.execute(
            """
            SELECT d.*, c.doc FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.collection_id = ? AND d.path = ? AND d.active = 1
            """,
            (collection_id, path),
        )
        row = cursor.fetchone()
        return self._row_to_document(row) if row else None

    def get_by_hash(self, hash_value: str) -> str | None:
        """Retrieve content by content hash.

        Args:
            hash_value: SHA256 hash of content.

        Returns:
            Content string if found, None otherwise.
        """
        cursor = self.db.execute("SELECT doc FROM content WHERE hash = ?", (hash_value,))
        row = cursor.fetchone()
        return row["doc"] if row else None

    def list_by_collection(self, collection_id: int, active_only: bool = True) -> list[DocumentResult]:
        """List all documents in a collection.

        Args:
            collection_id: ID of the collection.
            active_only: If True, only return active documents (default: True).

        Returns:
            List of DocumentResult objects.
        """
        active_filter = "AND d.active = 1" if active_only else ""
        cursor = self.db.execute(
            f"""
            SELECT d.*, c.doc FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.collection_id = ? {active_filter}
            ORDER BY d.path
            """,
            (collection_id,),
        )
        return [self._row_to_document(row) for row in cursor.fetchall()]

    def delete(self, collection_id: int, path: str) -> bool:
        """Soft-delete a document (mark as inactive).

        Args:
            collection_id: ID of the collection.
            path: Document path relative to collection.

        Returns:
            True if document was deleted, False if not found.
        """
        cursor = self.db.execute(
            """
            SELECT id FROM documents
            WHERE collection_id = ? AND path = ? AND active = 1
            """,
            (collection_id, path),
        )

        if not cursor.fetchone():
            return False

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE documents SET active = 0
                WHERE collection_id = ? AND path = ?
                """,
                (collection_id, path),
            )

        return True

    def check_if_modified(
        self, collection_id: int, path: str, new_hash: str
    ) -> bool:
        """Check if a document has been modified.

        Args:
            collection_id: ID of the collection.
            path: Document path relative to collection.
            new_hash: SHA256 hash of the new content.

        Returns:
            True if document has been modified (hash differs), False otherwise.
        """
        cursor = self.db.execute(
            """
            SELECT hash FROM documents
            WHERE collection_id = ? AND path = ? AND active = 1
            """,
            (collection_id, path),
        )
        row = cursor.fetchone()

        if not row:
            # Document doesn't exist, so it's "modified" (new)
            return True

        return row["hash"] != new_hash

    def get_content_length(self, collection_id: int, path: str) -> int | None:
        """Get the length of document content.

        Args:
            collection_id: ID of the collection.
            path: Document path relative to collection.

        Returns:
            Length of content in bytes, or None if not found.
        """
        cursor = self.db.execute(
            """
            SELECT LENGTH(c.doc) as len FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.collection_id = ? AND d.path = ? AND d.active = 1
            """,
            (collection_id, path),
        )
        row = cursor.fetchone()
        return row["len"] if row else None

    def count_by_collection(self, collection_id: int, active_only: bool = True) -> int:
        """Count documents in a collection.

        Args:
            collection_id: ID of the collection.
            active_only: If True, only count active documents (default: True).

        Returns:
            Number of documents.
        """
        active_filter = "AND active = 1" if active_only else ""
        cursor = self.db.execute(
            f"""
            SELECT COUNT(*) as count FROM documents
            WHERE collection_id = ? {active_filter}
            """,
            (collection_id,),
        )
        return cursor.fetchone()["count"]

    def get_by_path_prefix(self, collection_id: int, prefix: str) -> list[DocumentResult]:
        """Get all documents matching a path prefix.

        Args:
            collection_id: ID of the collection.
            prefix: Path prefix to match.

        Returns:
            List of DocumentResult objects.
        """
        cursor = self.db.execute(
            """
            SELECT d.*, c.doc FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.collection_id = ? AND d.path LIKE ? AND d.active = 1
            ORDER BY d.path
            """,
            (collection_id, f"{prefix}%"),
        )
        return [self._row_to_document(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_document(row) -> DocumentResult:  # type: ignore
        """Convert database row to DocumentResult object.

        Args:
            row: Database row from sqlite3.Row.

        Returns:
            DocumentResult object.
        """
        content = row.get("doc") if isinstance(row, dict) else getattr(row, "doc", None)
        return DocumentResult(
            filepath=row["path"],
            display_path=row["path"],
            title=row["title"],
            context=None,
            hash=row["hash"],
            collection_id=row["collection_id"],
            modified_at=row["modified_at"],
            body_length=len(content) if content else 0,
            body=content,
        )
