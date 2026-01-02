"""Document metadata storage for PMD.

This module provides storage for extracted document metadata including
tags, attributes, and the profile used for extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .database import Database


@dataclass
class StoredDocumentMetadata:
    """Stored metadata for a document.

    Attributes:
        document_id: ID of the document in the documents table.
        profile_name: Name of the profile used for extraction.
        tags: Set of normalized tags.
        source_tags: Original tags as found in the document.
        attributes: Key-value metadata attributes.
        extracted_at: When the metadata was extracted.
    """

    document_id: int
    profile_name: str
    tags: set[str]
    source_tags: list[str]
    extracted_at: str
    attributes: dict[str, Any] = field(default_factory=dict)


class DocumentMetadataRepository:
    """Repository for document metadata operations."""

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure the document_metadata table exists."""
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

    def upsert(self, metadata: StoredDocumentMetadata) -> None:
        """Insert or update document metadata.

        Args:
            metadata: StoredDocumentMetadata to store.
        """
        tags_json = json.dumps(sorted(metadata.tags))
        source_tags_json = json.dumps(metadata.source_tags)
        attributes_json = json.dumps(metadata.attributes) if metadata.attributes else None

        with self.db.transaction() as cursor:
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

        Args:
            tag: Normalized tag to search for.

        Returns:
            List of document IDs with this tag.
        """
        # Search in JSON array (SQLite JSON functions)
        cursor = self.db.execute(
            """
            SELECT document_id FROM document_metadata
            WHERE EXISTS (
                SELECT 1 FROM json_each(tags_json)
                WHERE value = ?
            )
            """,
            (tag,),
        )
        return [row["document_id"] for row in cursor.fetchall()]

    def find_documents_with_any_tag(self, tags: set[str]) -> list[int]:
        """Find documents with any of the specified tags.

        Args:
            tags: Set of tags to search for.

        Returns:
            List of document IDs with at least one matching tag.
        """
        if not tags:
            return []

        # Build query for any matching tag
        placeholders = ", ".join("?" for _ in tags)
        cursor = self.db.execute(
            f"""
            SELECT DISTINCT document_id FROM document_metadata
            WHERE EXISTS (
                SELECT 1 FROM json_each(tags_json)
                WHERE value IN ({placeholders})
            )
            """,
            tuple(tags),
        )
        return [row["document_id"] for row in cursor.fetchall()]

    def delete_by_document(self, document_id: int) -> bool:
        """Delete metadata for a document.

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
                "DELETE FROM document_metadata WHERE document_id = ?",
                (document_id,),
            )
        return True

    def cleanup_orphans(self) -> int:
        """Remove metadata for deleted documents.

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
                cursor.execute(
                    """
                    DELETE FROM document_metadata
                    WHERE document_id NOT IN (SELECT id FROM documents WHERE active = 1)
                    """
                )

        return count

    def get_all_tags(self) -> set[str]:
        """Get all unique tags across all documents.

        Returns:
            Set of all normalized tags in the database.
        """
        cursor = self.db.execute(
            """
            SELECT DISTINCT value as tag FROM document_metadata, json_each(tags_json)
            """
        )
        return {row["tag"] for row in cursor.fetchall()}

    def count_documents_by_tag(self) -> dict[str, int]:
        """Get tag frequency counts.

        Returns:
            Dictionary mapping tag to document count.
        """
        cursor = self.db.execute(
            """
            SELECT value as tag, COUNT(DISTINCT document_id) as count
            FROM document_metadata, json_each(tags_json)
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
