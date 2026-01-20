"""idx.store.fts - FTS5 full-text search support.

Provides FTS5 virtual table management and search operations for documents.
The FTS5 table uses porter stemming and unicode61 tokenizer for robust search.
Supports both explicit session injection and ambient session via contextvars.

Example usage:
    from idx.store.fts import FTSManager

    # With ambient session
    with use_session(session):
        manager = FTSManager()
        manager.upsert(doc_id=1, path="notes/test.md", body="Hello world")
        results = manager.search("hello")

    # Or with explicit session
    manager = FTSManager(session)
    manager.upsert(doc_id=1, path="notes/test.md", body="Hello world")
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.store.session_context import current_session

__all__ = [
    "FTSManager",
    "FTSResult",
    "create_fts_table",
    "drop_fts_table",
]

logger = get_logger(__name__)


@dataclass
class FTSResult:
    """Result from an FTS search."""

    doc_id: int
    path: str
    rank: float
    snippet: str | None = None


def create_fts_table(engine: Engine) -> None:
    """Create the FTS5 virtual table if it doesn't exist.

    Creates documents_fts with porter stemming and unicode61 tokenizer.
    Uses content='' for external content (we manage content ourselves).

    Args:
        engine: SQLAlchemy engine.
    """
    with engine.connect() as conn:
        conn.execute(
            text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    path,
                    body,
                    content='',
                    contentless_delete=1,
                    tokenize='porter unicode61'
                )
            """)
        )
        conn.commit()
    logger.debug("FTS5 table created or already exists")


def drop_fts_table(engine: Engine) -> None:
    """Drop the FTS5 virtual table.

    Args:
        engine: SQLAlchemy engine.
    """
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS documents_fts"))
        conn.commit()
    logger.debug("FTS5 table dropped")


class FTSManager:
    """Manages FTS5 full-text search operations.

    Handles indexing, updating, deleting, and searching documents
    using SQLite's FTS5 virtual table.

    The FTS5 table uses the document's primary key (id) as rowid
    for efficient update and delete operations.

    Can be initialized with an explicit session or use the ambient session
    from the current context (set via `use_session()`).
    """

    def __init__(self, session: Session | None = None) -> None:
        """Initialize the FTS manager.

        Args:
            session: SQLAlchemy session for database operations. If None,
                uses the ambient session from current_session().
        """
        self._explicit_session = session

    @property
    def _session(self) -> Session:
        """Get the session to use for database operations."""
        if self._explicit_session is not None:
            return self._explicit_session
        return current_session()

    def ensure_table_exists(self) -> None:
        """Ensure the FTS5 table exists.

        Creates the table if it doesn't exist.
        """
        engine = self._session.get_bind()
        if engine is not None:
            create_fts_table(engine)  # type: ignore

    def upsert(self, doc_id: int, path: str, body: str) -> None:
        """Insert or update a document in the FTS index.

        Uses INSERT OR REPLACE semantics with explicit rowid.

        Args:
            doc_id: Document ID (used as rowid).
            path: Document path for searching.
            body: Document body text for searching.
        """
        # Delete existing entry if any
        self._session.execute(
            text("DELETE FROM documents_fts WHERE rowid = :rowid"),
            {"rowid": doc_id},
        )
        # Insert new entry
        self._session.execute(
            text("""
                INSERT INTO documents_fts(rowid, path, body)
                VALUES (:rowid, :path, :body)
            """),
            {"rowid": doc_id, "path": path, "body": body},
        )
        logger.debug(f"FTS indexed document {doc_id}: {path}")

    def delete(self, doc_id: int) -> None:
        """Delete a document from the FTS index.

        Args:
            doc_id: Document ID (rowid) to delete.
        """
        self._session.execute(
            text("DELETE FROM documents_fts WHERE rowid = :rowid"),
            {"rowid": doc_id},
        )
        logger.debug(f"FTS deleted document {doc_id}")

    def delete_many(self, doc_ids: list[int]) -> int:
        """Delete multiple documents from the FTS index.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        if not doc_ids:
            return 0

        # SQLite has limits on IN clause size, batch if needed
        deleted = 0
        batch_size = 500
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i : i + batch_size]
            placeholders = ", ".join(f":id{j}" for j in range(len(batch)))
            params = {f"id{j}": id_ for j, id_ in enumerate(batch)}
            result = self._session.execute(
                text(f"DELETE FROM documents_fts WHERE rowid IN ({placeholders})"),
                params,
            )
            deleted += result.rowcount
        return deleted

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        dataset_filter: int | None = None,
    ) -> list[FTSResult]:
        """Search documents using FTS5.

        Args:
            query: FTS5 search query (supports FTS5 syntax).
            limit: Maximum number of results.
            dataset_filter: Optional dataset ID to filter by (requires join).

        Returns:
            List of FTSResult objects sorted by relevance.
        """
        if dataset_filter is not None:
            # Join with documents table to filter by dataset
            results = self._session.execute(
                text("""
                    SELECT
                        d.id as doc_id,
                        d.path,
                        bm25(documents_fts) as rank,
                        snippet(documents_fts, 1, '<b>', '</b>', '...', 32) as snippet
                    FROM documents_fts f
                    JOIN documents d ON d.id = f.rowid
                    WHERE documents_fts MATCH :query
                    AND d.dataset_id = :dataset_id
                    AND d.active = 1
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": query, "dataset_id": dataset_filter, "limit": limit},
            )
        else:
            # Search without dataset filter
            results = self._session.execute(
                text("""
                    SELECT
                        d.id as doc_id,
                        d.path,
                        bm25(documents_fts) as rank,
                        snippet(documents_fts, 1, '<b>', '</b>', '...', 32) as snippet
                    FROM documents_fts f
                    JOIN documents d ON d.id = f.rowid
                    WHERE documents_fts MATCH :query
                    AND d.active = 1
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": query, "limit": limit},
            )

        return [
            FTSResult(
                doc_id=row.doc_id,
                path=row.path,
                rank=row.rank,
                snippet=row.snippet,
            )
            for row in results
        ]

    def search_with_scores(
        self,
        query: str,
        *,
        limit: int = 100,
        dataset_filter: int | None = None,
    ) -> list[tuple[int, str, float]]:
        """Search and return (doc_id, path, normalized_score) tuples.

        Normalizes BM25 scores to 0-1 range using max normalization.

        Args:
            query: FTS5 search query.
            limit: Maximum number of results.
            dataset_filter: Optional dataset ID filter.

        Returns:
            List of (doc_id, path, score) tuples where score is 0-1.
        """
        results = self.search(query, limit=limit, dataset_filter=dataset_filter)
        if not results:
            return []

        # BM25 returns negative values (lower is better)
        # Convert to positive and normalize
        raw_scores = [-r.rank for r in results]
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            max_score = 1.0

        return [
            (r.doc_id, r.path, (-r.rank) / max_score)
            for r in results
        ]

    def count(self) -> int:
        """Count total documents in the FTS index.

        Returns:
            Number of indexed documents.
        """
        result = self._session.execute(
            text("SELECT COUNT(*) FROM documents_fts")
        )
        count = result.scalar()
        return count or 0

    def rebuild(self) -> None:
        """Rebuild the FTS index.

        This can help optimize the index after many updates.
        """
        self._session.execute(
            text("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        )
        logger.info("FTS index rebuilt")
