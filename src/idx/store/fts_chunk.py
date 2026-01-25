"""idx.store.fts_chunk - FTS5 full-text search support for chunks.

Provides FTS5 virtual table management and search operations for document chunks.
The FTS5 table uses porter stemming and unicode61 tokenizer for robust search.
Supports both explicit session injection and ambient session via contextvars.

Unlike the document-level FTS table, this uses string node_ids as identifiers
(format: `{hash}:{seq}`) rather than integer rowids.

Example usage:
    from idx.store.fts_chunk import FTSChunkManager

    # With ambient session
    with use_session(session):
        manager = FTSChunkManager()
        manager.upsert(
            node_id="abc123:0",
            text="Hello world",
            source_doc_id="obsidian:notes/test.md"
        )
        results = manager.search_with_scores("hello")

    # Or with explicit session
    manager = FTSChunkManager(session)
    manager.upsert(...)
"""

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.store.session_context import current_session

__all__ = [
    "FTSChunkManager",
    "FTSChunkResult",
    "create_chunks_fts_table",
    "drop_chunks_fts_table",
]

logger = get_logger(__name__)


@dataclass
class FTSChunkResult:
    """Result from an FTS chunk search.

    Attributes:
        node_id: Canonical chunk ID in format `{hash}:{seq}`.
        text: The chunk text content.
        source_doc_id: Composite document key `{dataset_name}:{path}`.
        score: Normalized relevance score (0-1, higher is better).
    """

    node_id: str
    text: str
    source_doc_id: str
    score: float


def create_chunks_fts_table(engine: Engine) -> None:
    """Create the chunks FTS5 virtual table if it doesn't exist.

    Creates chunks_fts with porter stemming and unicode61 tokenizer.
    Unlike document-level FTS which uses contentless mode with external joins,
    chunk FTS stores content directly to enable text retrieval without joins.

    Args:
        engine: SQLAlchemy engine.
    """
    with engine.connect() as conn:
        conn.execute(
            sql_text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    node_id,
                    text,
                    source_doc_id,
                    tokenize='porter unicode61'
                )
            """)
        )
        conn.commit()
    logger.debug("chunks_fts FTS5 table created or already exists")


def drop_chunks_fts_table(engine: Engine) -> None:
    """Drop the chunks FTS5 virtual table.

    Args:
        engine: SQLAlchemy engine.
    """
    with engine.connect() as conn:
        conn.execute(sql_text("DROP TABLE IF EXISTS chunks_fts"))
        conn.commit()
    logger.debug("chunks_fts FTS5 table dropped")


class FTSChunkManager:
    """Manages FTS5 full-text search operations for document chunks.

    Handles indexing, updating, deleting, and searching chunks
    using SQLite's FTS5 virtual table.

    Unlike document-level FTS which uses integer rowids, this uses
    string node_ids as identifiers (format: `{hash}:{seq}`).

    Can be initialized with an explicit session or use the ambient session
    from the current context (set via `use_session()`).
    """

    def __init__(self, session: Session | None = None) -> None:
        """Initialize the FTS chunk manager.

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
        """Ensure the chunks FTS5 table exists.

        Creates the table if it doesn't exist.
        """
        engine = self._session.get_bind()
        if engine is not None:
            create_chunks_fts_table(engine)  # type: ignore

    def upsert(self, node_id: str, text: str, source_doc_id: str) -> None:
        """Insert or update a chunk in the FTS index.

        Uses DELETE then INSERT semantics since FTS5 with contentless tables
        doesn't support UPDATE.

        Args:
            node_id: Canonical chunk ID (format: `{hash}:{seq}`).
            text: Chunk text content for searching.
            source_doc_id: Composite document key `{dataset_name}:{path}`.
        """
        # Delete existing entry if any
        self._session.execute(
            sql_text("DELETE FROM chunks_fts WHERE node_id = :node_id"),
            {"node_id": node_id},
        )
        # Insert new entry
        self._session.execute(
            sql_text("""
                INSERT INTO chunks_fts(node_id, text, source_doc_id)
                VALUES (:node_id, :text, :source_doc_id)
            """),
            {"node_id": node_id, "text": text, "source_doc_id": source_doc_id},
        )
        logger.debug(f"FTS indexed chunk {node_id} from {source_doc_id}")

    def delete(self, node_id: str) -> None:
        """Delete a chunk from the FTS index.

        Args:
            node_id: Chunk ID to delete.
        """
        self._session.execute(
            sql_text("DELETE FROM chunks_fts WHERE node_id = :node_id"),
            {"node_id": node_id},
        )
        logger.debug(f"FTS deleted chunk {node_id}")

    def delete_by_source_doc_id(self, source_doc_id: str) -> int:
        """Delete all chunks for a document from the FTS index.

        Args:
            source_doc_id: Composite document key `{dataset_name}:{path}`.

        Returns:
            Number of chunks deleted.
        """
        result = self._session.execute(
            sql_text("DELETE FROM chunks_fts WHERE source_doc_id = :source_doc_id"),
            {"source_doc_id": source_doc_id},
        )
        deleted = result.rowcount
        logger.debug(f"FTS deleted {deleted} chunks for {source_doc_id}")
        return deleted

    def delete_many(self, node_ids: list[str]) -> int:
        """Delete multiple chunks from the FTS index.

        Args:
            node_ids: List of chunk IDs to delete.

        Returns:
            Number of chunks deleted.
        """
        if not node_ids:
            return 0

        # SQLite has limits on IN clause size, batch if needed
        deleted = 0
        batch_size = 500
        for i in range(0, len(node_ids), batch_size):
            batch = node_ids[i : i + batch_size]
            placeholders = ", ".join(f":id{j}" for j in range(len(batch)))
            params = {f"id{j}": id_ for j, id_ in enumerate(batch)}
            result = self._session.execute(
                sql_text(f"DELETE FROM chunks_fts WHERE node_id IN ({placeholders})"),
                params,
            )
            deleted += result.rowcount
        return deleted

    def search_with_scores(
        self,
        query: str,
        limit: int = 100,
        source_doc_id_prefix: str | None = None,
    ) -> list[FTSChunkResult]:
        """Search chunks and return results with normalized scores.

        Normalizes BM25 scores to 0-1 range using max normalization.

        Args:
            query: FTS5 search query (supports FTS5 syntax like AND, OR, NEAR).
            limit: Maximum number of results.
            source_doc_id_prefix: Optional prefix filter for source_doc_id
                (e.g., "obsidian:" to search only obsidian documents).

        Returns:
            List of FTSChunkResult objects sorted by relevance (highest score first).
        """
        if source_doc_id_prefix is not None:
            # Filter by source_doc_id prefix
            results = self._session.execute(
                sql_text("""
                    SELECT
                        node_id,
                        text,
                        source_doc_id,
                        bm25(chunks_fts) as rank
                    FROM chunks_fts
                    WHERE chunks_fts MATCH :query
                    AND source_doc_id LIKE :prefix
                    ORDER BY rank
                    LIMIT :limit
                """),
                {
                    "query": query,
                    "prefix": f"{source_doc_id_prefix}%",
                    "limit": limit,
                },
            )
        else:
            # Search without filter
            results = self._session.execute(
                sql_text("""
                    SELECT
                        node_id,
                        text,
                        source_doc_id,
                        bm25(chunks_fts) as rank
                    FROM chunks_fts
                    WHERE chunks_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": query, "limit": limit},
            )

        rows = list(results)
        if not rows:
            return []

        # BM25 returns negative values (lower/more negative is better)
        # Convert to positive and normalize to 0-1
        raw_scores = [-row.rank for row in rows]
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            max_score = 1.0

        return [
            FTSChunkResult(
                node_id=row.node_id,
                text=row.text,
                source_doc_id=row.source_doc_id,
                score=(-row.rank) / max_score,
            )
            for row in rows
        ]

    def count(self) -> int:
        """Count total chunks in the FTS index.

        Returns:
            Number of indexed chunks.
        """
        result = self._session.execute(sql_text("SELECT COUNT(*) FROM chunks_fts"))
        count = result.scalar()
        return count or 0

    def rebuild(self) -> None:
        """Rebuild the FTS index.

        This can help optimize the index after many updates.
        """
        self._session.execute(
            sql_text("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        )
        logger.info("chunks_fts FTS index rebuilt")
