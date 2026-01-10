"""Document metadata persistence for PMD.

This module provides storage for extracted document metadata including
tags, attributes, and the profile used for extraction.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pmd.metadata.model import StoredDocumentMetadata

if TYPE_CHECKING:
    from pmd.store.database import Database


class DocumentMetadataRepository:
    """Repository for document metadata operations."""

    def __init__(self, db: "Database"):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure the document_metadata and document_tags tables exist."""
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id),
                profile_name TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                source_tags_json TEXT NOT NULL,
                attributes_json TEXT,
                extracted_at TEXT NOT NULL
            )
            """
        )
        # Junction table for fast tag lookups
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_tags (
                document_id INTEGER NOT NULL REFERENCES documents(id),
                tag TEXT NOT NULL,
                PRIMARY KEY (document_id, tag)
            )
            """
        )
        # Index for fast lookups by tag
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag)"
        )

    def upsert(self, metadata: StoredDocumentMetadata) -> None:
        """Insert or update document metadata.

        Also maintains the document_tags junction table for fast lookups.

        Args:
            metadata: StoredDocumentMetadata to store.
        """
        tags_json = json.dumps(sorted(metadata.tags))
        source_tags_json = json.dumps(metadata.source_tags)
        attributes_json = json.dumps(metadata.attributes) if metadata.attributes else None

        with self.db.transaction() as cursor:
            # Upsert metadata
            cursor.execute(
                """
                INSERT INTO document_metadata (
                    document_id, profile_name, tags_json, source_tags_json,
                    attributes_json, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    profile_name = excluded.profile_name,
                    tags_json = excluded.tags_json,
                    source_tags_json = excluded.source_tags_json,
                    attributes_json = excluded.attributes_json,
                    extracted_at = excluded.extracted_at
                """,
                (
                    metadata.document_id,
                    metadata.profile_name,
                    tags_json,
                    source_tags_json,
                    attributes_json,
                    metadata.extracted_at,
                ),
            )

            # Update junction table: delete old, insert new
            cursor.execute(
                "DELETE FROM document_tags WHERE document_id = ?",
                (metadata.document_id,),
            )
            if metadata.tags:
                cursor.executemany(
                    "INSERT INTO document_tags (document_id, tag) VALUES (?, ?)",
                    [(metadata.document_id, tag) for tag in metadata.tags],
                )

    def get_by_document(self, document_id: int) -> StoredDocumentMetadata | None:
        """Get metadata for a document.

        Args:
            document_id: Document ID to look up.

        Returns:
            StoredDocumentMetadata if found, None otherwise.
        """
        cursor = self.db.execute(
            "SELECT * FROM document_metadata WHERE document_id = ?",
            (document_id,),
        )
        row = cursor.fetchone()
        return self._row_to_metadata(row) if row else None

    def get_tags(self, document_id: int) -> set[str]:
        """Get tags for a document.

        Args:
            document_id: Document ID to look up.

        Returns:
            Set of normalized tags, empty set if not found.
        """
        cursor = self.db.execute(
            "SELECT tags_json FROM document_metadata WHERE document_id = ?",
            (document_id,),
        )
        row = cursor.fetchone()
        if row and row["tags_json"]:
            return set(json.loads(row["tags_json"]))
        return set()

    def find_documents_with_tag(self, tag: str) -> list[int]:
        """Find all documents with a specific tag.

        Uses the document_tags junction table for fast indexed lookups.

        Args:
            tag: Normalized tag to search for.

        Returns:
            List of document IDs with this tag.
        """
        cursor = self.db.execute(
            "SELECT document_id FROM document_tags WHERE tag = ?",
            (tag,),
        )
        return [row["document_id"] for row in cursor.fetchall()]

    def find_documents_with_any_tag(self, tags: set[str]) -> list[int]:
        """Find documents with any of the specified tags.

        Uses the document_tags junction table for fast indexed lookups.

        Args:
            tags: Set of tags to search for.

        Returns:
            List of document IDs with at least one matching tag.
        """
        if not tags:
            return []

        placeholders = ", ".join("?" for _ in tags)
        cursor = self.db.execute(
            f"SELECT DISTINCT document_id FROM document_tags WHERE tag IN ({placeholders})",
            tuple(tags),
        )
        return [row["document_id"] for row in cursor.fetchall()]

    def find_documents_with_all_tags(self, tags: set[str]) -> list[int]:
        """Find documents that have ALL of the specified tags.

        Uses the document_tags junction table for fast indexed lookups.

        Args:
            tags: Set of tags that must all be present.

        Returns:
            List of document IDs with all matching tags.
        """
        if not tags:
            return []

        placeholders = ", ".join("?" for _ in tags)
        cursor = self.db.execute(
            f"""
            SELECT document_id FROM document_tags
            WHERE tag IN ({placeholders})
            GROUP BY document_id
            HAVING COUNT(DISTINCT tag) = ?
            """,
            (*tags, len(tags)),
        )
        return [row["document_id"] for row in cursor.fetchall()]

    def delete_by_document(self, document_id: int) -> bool:
        """Delete metadata for a document.

        Also removes entries from the document_tags junction table.

        Args:
            document_id: Document ID to delete metadata for.

        Returns:
            True if metadata was deleted, False if not found.
        """
        cursor = self.db.execute(
            "SELECT id FROM document_metadata WHERE document_id = ?",
            (document_id,),
        )
        if not cursor.fetchone():
            return False

        with self.db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM document_tags WHERE document_id = ?",
                (document_id,),
            )
            cursor.execute(
                "DELETE FROM document_metadata WHERE document_id = ?",
                (document_id,),
            )
        return True

    def cleanup_orphans(self) -> int:
        """Remove metadata for deleted documents.

        Also cleans up the document_tags junction table.

        Returns:
            Number of orphaned records removed.
        """
        cursor = self.db.execute(
            """
            SELECT COUNT(*) as count FROM document_metadata
            WHERE document_id NOT IN (SELECT id FROM documents WHERE active = 1)
            """
        )
        count = cursor.fetchone()["count"]

        if count > 0:
            with self.db.transaction() as cursor:
                # Clean up junction table first
                cursor.execute(
                    """
                    DELETE FROM document_tags
                    WHERE document_id NOT IN (SELECT id FROM documents WHERE active = 1)
                    """
                )
                cursor.execute(
                    """
                    DELETE FROM document_metadata
                    WHERE document_id NOT IN (SELECT id FROM documents WHERE active = 1)
                    """
                )

        return count

    def get_all_tags(self) -> set[str]:
        """Get all unique tags across all documents.

        Uses the document_tags junction table for fast lookups.

        Returns:
            Set of all normalized tags in the database.
        """
        cursor = self.db.execute("SELECT DISTINCT tag FROM document_tags")
        return {row["tag"] for row in cursor.fetchall()}

    def count_documents_by_tag(self) -> dict[str, int]:
        """Get tag frequency counts.

        Uses the document_tags junction table for fast lookups.

        Returns:
            Dictionary mapping tag to document count.
        """
        cursor = self.db.execute(
            """
            SELECT tag, COUNT(*) as count
            FROM document_tags
            GROUP BY tag
            ORDER BY count DESC
            """
        )
        return {row["tag"]: row["count"] for row in cursor.fetchall()}

    @staticmethod
    def _row_to_metadata(row) -> StoredDocumentMetadata:
        """Convert database row to StoredDocumentMetadata.

        Args:
            row: Database row from sqlite3.Row.

        Returns:
            StoredDocumentMetadata object.
        """
        tags = set(json.loads(row["tags_json"])) if row["tags_json"] else set()
        source_tags = json.loads(row["source_tags_json"]) if row["source_tags_json"] else []
        attributes = json.loads(row["attributes_json"]) if row["attributes_json"] else {}

        return StoredDocumentMetadata(
            document_id=row["document_id"],
            profile_name=row["profile_name"],
            tags=tags,
            source_tags=source_tags,
            attributes=attributes,
            extracted_at=row["extracted_at"],
        )
