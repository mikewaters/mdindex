"""Source metadata storage for PMD.

This module provides storage for source-specific metadata like ETags,
Last-Modified timestamps, and other information needed for efficient
incremental updates of remote documents.

Note: Source metadata columns have been merged into the documents table.
This repository acts as a compatibility wrapper that reads/writes the
merged columns on documents directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..database import Database


@dataclass
class SourceMetadata:
    """Metadata about a document's source.

    Attributes:
        document_id: ID of the document in the documents table.
        source_uri: Original URI the document was fetched from.
        etag: HTTP ETag header value (for change detection).
        last_modified: HTTP Last-Modified header value.
        last_fetched_at: When the document was last fetched.
        fetch_duration_ms: How long the fetch took in milliseconds.
        http_status: HTTP status code from the last fetch.
        content_type: MIME type of the content.
        extra: Additional source-specific metadata.
    """

    document_id: int
    source_uri: str
    last_fetched_at: str
    etag: str | None = None
    last_modified: str | None = None
    fetch_duration_ms: int | None = None
    http_status: int | None = None
    content_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class SourceMetadataRepository:
    """Repository for source metadata operations.

    Source metadata columns are stored directly on the documents table.
    This repository provides a compatibility API that reads/writes
    those merged columns.
    """

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def upsert(self, metadata: SourceMetadata) -> None:
        """Insert or update source metadata.

        Updates the source metadata columns on the documents row.

        Args:
            metadata: SourceMetadata to store.
        """
        extra_json = json.dumps(metadata.extra) if metadata.extra else None

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE documents SET
                    source_uri = ?,
                    etag = ?,
                    last_modified = ?,
                    last_fetched_at = ?,
                    fetch_duration_ms = ?,
                    http_status = ?,
                    content_type = ?,
                    extra_metadata = ?
                WHERE id = ?
                """,
                (
                    metadata.source_uri,
                    metadata.etag,
                    metadata.last_modified,
                    metadata.last_fetched_at,
                    metadata.fetch_duration_ms,
                    metadata.http_status,
                    metadata.content_type,
                    extra_json,
                    metadata.document_id,
                ),
            )

    def get_by_document(self, document_id: int) -> SourceMetadata | None:
        """Get source metadata for a document.

        Args:
            document_id: Document ID to look up.

        Returns:
            SourceMetadata if the document has source metadata, None otherwise.
        """
        cursor = self.db.execute(
            """
            SELECT id as document_id, source_uri, etag, last_modified,
                   last_fetched_at, fetch_duration_ms, http_status,
                   content_type, extra_metadata
            FROM documents WHERE id = ? AND source_uri IS NOT NULL
            """,
            (document_id,),
        )
        row = cursor.fetchone()
        return self._row_to_metadata(row) if row else None

    def get_by_uri(self, source_uri: str) -> list[SourceMetadata]:
        """Get all metadata for a source URI.

        Args:
            source_uri: Source URI to look up.

        Returns:
            List of SourceMetadata objects.
        """
        cursor = self.db.execute(
            """
            SELECT id as document_id, source_uri, etag, last_modified,
                   last_fetched_at, fetch_duration_ms, http_status,
                   content_type, extra_metadata
            FROM documents WHERE source_uri = ? AND active = 1
            """,
            (source_uri,),
        )
        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def delete_by_document(self, document_id: int) -> bool:
        """Delete source metadata for a document.

        Clears the source metadata columns on the document row.

        Args:
            document_id: Document ID to delete metadata for.

        Returns:
            True if metadata was cleared, False if document not found or had no metadata.
        """
        # Check if document exists and has source metadata
        cursor = self.db.execute(
            "SELECT source_uri FROM documents WHERE id = ?",
            (document_id,),
        )
        row = cursor.fetchone()
        if not row or row["source_uri"] is None:
            return False

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE documents SET
                    source_uri = NULL,
                    etag = NULL,
                    last_modified = NULL,
                    last_fetched_at = NULL,
                    fetch_duration_ms = NULL,
                    http_status = NULL,
                    content_type = NULL,
                    extra_metadata = NULL
                WHERE id = ?
                """,
                (document_id,),
            )
        return True

    def needs_refresh(
        self,
        document_id: int,
        max_age_seconds: int = 3600,
    ) -> bool:
        """Check if a document needs to be refreshed.

        A document needs refresh if:
        - No source metadata exists (never fetched / no last_fetched_at)
        - Last fetch was more than max_age_seconds ago

        Args:
            document_id: Document ID to check.
            max_age_seconds: Maximum age in seconds before refresh needed.

        Returns:
            True if the document should be re-fetched.
        """
        metadata = self.get_by_document(document_id)
        if metadata is None:
            return True

        try:
            last_fetched = datetime.fromisoformat(metadata.last_fetched_at)
            age = datetime.utcnow() - last_fetched
            return age > timedelta(seconds=max_age_seconds)
        except (ValueError, TypeError):
            return True

    def get_stale_documents(
        self,
        source_collection_id: int,
        max_age_seconds: int = 3600,
    ) -> list[int]:
        """Get document IDs that need refresh in a collection.

        Args:
            source_collection_id: Collection to check.
            max_age_seconds: Maximum age in seconds.

        Returns:
            List of document IDs needing refresh.
        """
        cutoff = (
            datetime.utcnow() - timedelta(seconds=max_age_seconds)
        ).isoformat()

        cursor = self.db.execute(
            """
            SELECT id FROM documents
            WHERE source_collection_id = ? AND active = 1
            AND (last_fetched_at IS NULL OR last_fetched_at < ?)
            """,
            (source_collection_id, cutoff),
        )
        return [row["id"] for row in cursor.fetchall()]

    def cleanup_orphans(self) -> int:
        """Remove metadata for deleted documents.

        Since source metadata is now stored in the documents table,
        orphan cleanup is handled by document deletion. This method
        is kept for API compatibility but always returns 0.

        Returns:
            Always 0 (no separate table to clean).
        """
        return 0

    @staticmethod
    def _row_to_metadata(row) -> SourceMetadata:
        """Convert database row to SourceMetadata.

        Args:
            row: Database row from documents table query.

        Returns:
            SourceMetadata object.
        """
        extra_json = row["extra_metadata"]
        extra = json.loads(extra_json) if extra_json else {}

        return SourceMetadata(
            document_id=row["document_id"],
            source_uri=row["source_uri"],
            etag=row["etag"],
            last_modified=row["last_modified"],
            last_fetched_at=row["last_fetched_at"],
            fetch_duration_ms=row["fetch_duration_ms"],
            http_status=row["http_status"],
            content_type=row["content_type"],
            extra=extra,
        )
