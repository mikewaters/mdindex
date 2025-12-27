"""Embedding storage and vector search for PMD."""

from datetime import datetime

from ..core.types import SearchResult, SearchSource
from .database import Database


class EmbeddingRepository:
    """Repository for embedding storage and vector similarity search."""

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def store_embedding(
        self,
        hash_value: str,
        seq: int,
        pos: int,
        embedding: list[float],
        model: str,
    ) -> None:
        """Store embedding vector for a content chunk.

        Args:
            hash_value: SHA256 hash of content.
            seq: Sequence number for chunks (0, 1, 2, ...).
            pos: Character position in original document.
            embedding: Vector embedding (list of floats).
            model: Model name used to generate embedding.
        """
        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            # Store metadata
            cursor.execute(
                """
                INSERT OR REPLACE INTO content_vectors
                (hash, seq, pos, model, embedded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (hash_value, seq, pos, model, now),
            )

            # Note: sqlite-vec storage is abstracted here for Phase 2
            # Phase 3 will integrate actual vector storage

    def get_embeddings_for_content(self, hash_value: str) -> list[tuple[int, int, str]]:
        """Get all embeddings for a content hash.

        Args:
            hash_value: SHA256 hash of content.

        Returns:
            List of (seq, pos, model) tuples.
        """
        cursor = self.db.execute(
            """
            SELECT seq, pos, model FROM content_vectors
            WHERE hash = ?
            ORDER BY seq
            """,
            (hash_value,),
        )
        return [(row["seq"], row["pos"], row["model"]) for row in cursor.fetchall()]

    def has_embeddings(self, hash_value: str, model: str | None = None) -> bool:
        """Check if content has embeddings.

        Args:
            hash_value: SHA256 hash of content.
            model: Optional model name filter.

        Returns:
            True if embeddings exist, False otherwise.
        """
        if model:
            cursor = self.db.execute(
                """
                SELECT 1 FROM content_vectors
                WHERE hash = ? AND model = ?
                LIMIT 1
                """,
                (hash_value, model),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT 1 FROM content_vectors
                WHERE hash = ?
                LIMIT 1
                """,
                (hash_value,),
            )

        return cursor.fetchone() is not None

    def delete_embeddings(self, hash_value: str) -> int:
        """Delete all embeddings for content.

        Args:
            hash_value: SHA256 hash of content.

        Returns:
            Number of embedding records deleted.
        """
        with self.db.transaction() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM content_vectors WHERE hash = ?",
                (hash_value,),
            )
            count = cursor.fetchone()["count"]

            cursor.execute("DELETE FROM content_vectors WHERE hash = ?", (hash_value,))

        return count

    def clear_embeddings_by_model(self, model: str) -> int:
        """Delete all embeddings for a specific model.

        Args:
            model: Model name.

        Returns:
            Number of embedding records deleted.
        """
        with self.db.transaction() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM content_vectors WHERE model = ?",
                (model,),
            )
            count = cursor.fetchone()["count"]

            cursor.execute("DELETE FROM content_vectors WHERE model = ?", (model,))

        return count

    def count_embeddings(self, model: str | None = None) -> int:
        """Count stored embeddings.

        Args:
            model: Optional model name filter.

        Returns:
            Number of embedding records.
        """
        if model:
            cursor = self.db.execute(
                "SELECT COUNT(*) as count FROM content_vectors WHERE model = ?",
                (model,),
            )
        else:
            cursor = self.db.execute("SELECT COUNT(*) as count FROM content_vectors")

        return cursor.fetchone()["count"]

    def search_vectors(
        self,
        query_embedding: list[float],
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for documents by vector similarity.

        Note: This is a placeholder for Phase 2. Phase 3 will implement
        actual cosine similarity search via sqlite-vec.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results to return.
            collection_id: Optional collection ID to limit search scope.
            min_score: Minimum similarity score (0.0-1.0).

        Returns:
            List of SearchResult objects.
        """
        # Phase 2: Return empty list (no vector search yet)
        # Phase 3: Implement actual cosine similarity search via sqlite-vec
        return []
