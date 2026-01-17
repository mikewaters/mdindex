"""Embedding storage and vector search for PMD."""

import struct
import time
from datetime import datetime

from loguru import logger

from ...core.types import SearchResult, SearchSource
from ..database import Database


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize embedding to binary format for sqlite-vec.

    Args:
        embedding: List of floats.

    Returns:
        Binary representation of the embedding.
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


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
        logger.debug(f"Storing embedding: hash={hash_value[:12]}..., seq={seq}, dim={len(embedding)}")
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

            # Store actual vector in sqlite-vec if available
            if self.db.vec_available:
                # Create composite key for the vector
                vec_key = f"{hash_value}:{seq}"
                vec_bytes = _serialize_embedding(embedding)

                # Delete existing entry if present
                cursor.execute(
                    "DELETE FROM content_vectors_vec WHERE hash = ?",
                    (vec_key,),
                )

                # Insert the embedding vector
                cursor.execute(
                    """
                    INSERT INTO content_vectors_vec (hash, seq, embedding)
                    VALUES (?, ?, ?)
                    """,
                    (vec_key, seq, vec_bytes),
                )

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

            # Delete from metadata table
            cursor.execute("DELETE FROM content_vectors WHERE hash = ?", (hash_value,))

            # Delete from vector table if available
            if self.db.vec_available:
                cursor.execute(
                    "DELETE FROM content_vectors_vec WHERE hash LIKE ?",
                    (f"{hash_value}:%",),
                )

        if count > 0:
            logger.debug(f"Deleted {count} embeddings for hash={hash_value[:12]}...")

        return count

    def clear_embeddings_by_model(self, model: str) -> int:
        """Delete all embeddings for a specific model.

        Args:
            model: Model name.

        Returns:
            Number of embedding records deleted.
        """
        logger.debug(f"Clearing embeddings for model: {model!r}")

        with self.db.transaction() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM content_vectors WHERE model = ?",
                (model,),
            )
            count = cursor.fetchone()["count"]

            cursor.execute("DELETE FROM content_vectors WHERE model = ?", (model,))

        if count > 0:
            logger.info(f"Cleared {count} embeddings for model: {model!r}")

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
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for documents by vector similarity using sqlite-vec.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results to return.
            source_collection_id: Optional collection ID to limit search scope.
            min_score: Minimum similarity score (0.0-1.0).

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        logger.debug(f"Vector search: limit={limit}, source_collection_id={source_collection_id}, min_score={min_score}")
        start_time = time.perf_counter()

        if not self.db.vec_available or not query_embedding:
            logger.debug("Vector search skipped: vec not available or no query embedding")
            return []

        # Serialize query embedding
        query_bytes = _serialize_embedding(query_embedding)

        # Build query - sqlite-vec uses L2 distance by default
        # We join with content_vectors to get the actual hash, then join with documents
        if source_collection_id is not None:
            sql = """
                SELECT
                    d.id,
                    d.source_collection_id,
                    d.path,
                    d.path as display_path,
                    d.title,
                    d.hash,
                    d.modified_at,
                    c.doc as body,
                    cv.seq,
                    cv.pos,
                    v.distance
                FROM content_vectors_vec v
                JOIN content_vectors cv ON (
                    cv.hash || ':' || cv.seq = v.hash
                )
                JOIN documents d ON d.hash = cv.hash AND d.active = 1
                JOIN content c ON d.hash = c.hash
                WHERE v.embedding MATCH ? AND k = ?
                AND d.source_collection_id = ?
                ORDER BY v.distance ASC
            """
            cursor = self.db.execute(sql, (query_bytes, limit * 3, source_collection_id))
        else:
            sql = """
                SELECT
                    d.id,
                    d.source_collection_id,
                    d.path,
                    d.path as display_path,
                    d.title,
                    d.hash,
                    d.modified_at,
                    c.doc as body,
                    cv.seq,
                    cv.pos,
                    v.distance
                FROM content_vectors_vec v
                JOIN content_vectors cv ON (
                    cv.hash || ':' || cv.seq = v.hash
                )
                JOIN documents d ON d.hash = cv.hash AND d.active = 1
                JOIN content c ON d.hash = c.hash
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance ASC
            """
            
            cursor = self.db.execute(sql, (query_bytes, limit * 3))

        results = []
        seen_docs: set[str] = set()

        for row in cursor.fetchall():
            
            # Skip duplicate documents (we want best chunk per doc)
            if row["path"] in seen_docs:
                continue

            # Convert L2 distance to similarity score (0-1 range)
            # Using: score = 1 / (1 + distance)
            distance = row["distance"]
            score = 1.0 / (1.0 + distance)

            if score >= min_score:
                seen_docs.add(row["path"])
                results.append(
                    SearchResult(
                        filepath=row["path"],
                        display_path=row["display_path"],
                        title=row["title"],
                        context=None,
                        hash=row["hash"],
                        source_collection_id=row["source_collection_id"],
                        modified_at=row["modified_at"],
                        body_length=len(row["body"]) if row["body"] else 0,
                        body=row["body"],  # Include body for reranking
                        score=score,
                        source=SearchSource.VECTOR,
                        chunk_pos=row["pos"],
                    )
                )

                if len(results) >= limit:
                    break

        elapsed = (time.perf_counter() - start_time) * 1000
        if results:
            top_score = results[0].score if results else 0
            logger.debug(f"Vector search complete: {len(results)} results, top_score={top_score:.3f}, {elapsed:.1f}ms")
        else:
            logger.debug(f"Vector search complete: no results, {elapsed:.1f}ms")

        return results

    def count_distinct_hashes(self) -> int:
        """Count distinct content hashes with embeddings.

        Returns:
            Number of unique content hashes that have embeddings.
        """
        cursor = self.db.execute(
            "SELECT COUNT(DISTINCT hash) as count FROM content_vectors"
        )
        return cursor.fetchone()["count"]

    def count_documents_missing_embeddings(
        self, source_collection_id: int | None = None
    ) -> int:
        """Count active documents missing embeddings.

        Args:
            source_collection_id: Optional collection ID to scope count.

        Returns:
            Number of documents without embeddings.
        """
        if source_collection_id is not None:
            cursor = self.db.execute(
                """
                SELECT COUNT(DISTINCT d.id) as count
                FROM documents d
                LEFT JOIN content_vectors cv ON cv.hash = d.hash
                WHERE d.active = 1 AND d.source_collection_id = ?
                AND cv.hash IS NULL
                """,
                (source_collection_id,),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT COUNT(DISTINCT d.id) as count
                FROM documents d
                LEFT JOIN content_vectors cv ON cv.hash = d.hash
                WHERE d.active = 1 AND cv.hash IS NULL
                """
            )
        return cursor.fetchone()["count"]

    def list_paths_missing_embeddings(
        self, source_collection_id: int | None = None, limit: int = 20
    ) -> list[str]:
        """List paths of documents missing embeddings.

        Args:
            source_collection_id: Optional collection ID to scope query.
            limit: Maximum number of paths to return.

        Returns:
            List of document paths without embeddings.
        """
        if source_collection_id is not None:
            cursor = self.db.execute(
                """
                SELECT d.path
                FROM documents d
                LEFT JOIN content_vectors cv ON cv.hash = d.hash
                WHERE d.active = 1 AND d.source_collection_id = ?
                AND cv.hash IS NULL
                ORDER BY d.path
                LIMIT ?
                """,
                (source_collection_id, limit),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT d.path
                FROM documents d
                LEFT JOIN content_vectors cv ON cv.hash = d.hash
                WHERE d.active = 1 AND cv.hash IS NULL
                ORDER BY d.path
                LIMIT ?
                """,
                (limit,),
            )
        return [row["path"] for row in cursor.fetchall()]

    def count_orphaned(self) -> int:
        """Count embedding records not referenced by any active document.

        Returns:
            Number of distinct orphaned content hashes with embeddings.
        """
        cursor = self.db.execute(
            """
            SELECT COUNT(DISTINCT cv.hash) as count
            FROM content_vectors cv
            LEFT JOIN documents d ON d.hash = cv.hash AND d.active = 1
            WHERE d.hash IS NULL
            """
        )
        return cursor.fetchone()["count"]

    def delete_orphaned(self) -> int:
        """Delete embedding records not referenced by any active document.

        Returns:
            Number of embedding records deleted.
        """
        # Count first for logging
        cursor = self.db.execute(
            """
            SELECT COUNT(*) as count FROM content_vectors
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            """
        )
        count = cursor.fetchone()["count"]

        if count > 0:
            logger.debug(f"Deleting {count} orphaned embedding records")
            self.db.execute(
                """
                DELETE FROM content_vectors
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
                """
            )
            logger.info(f"Deleted {count} orphaned embedding records")

        return count
