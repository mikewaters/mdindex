"""Indexing service for document indexing and embedding operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ..core.exceptions import CollectionNotFoundError
from ..search.text import is_indexable

if TYPE_CHECKING:
    from .container import ServiceContainer


@dataclass
class IndexResult:
    """Result of an indexing operation."""

    indexed: int
    """Number of documents indexed."""

    skipped: int
    """Number of documents skipped (unchanged)."""

    errors: list[tuple[str, str]]
    """List of (path, error_message) for failed documents."""


@dataclass
class EmbedResult:
    """Result of an embedding operation."""

    embedded: int
    """Number of documents embedded."""

    skipped: int
    """Number of documents skipped (already embedded)."""

    chunks_total: int
    """Total number of chunks created."""


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""

    orphaned_content: int
    """Number of orphaned content hashes removed."""

    orphaned_embeddings: int
    """Number of orphaned embeddings removed."""


class IndexingService:
    """Service for document indexing and embedding operations.

    This service handles:
    - Filesystem scanning and document indexing (FTS)
    - Embedding generation for vector search
    - Cleanup of orphaned data

    Example:

        async with ServiceContainer(config) as services:
            # Index a collection
            result = await services.indexing.index_collection("my-docs")
            print(f"Indexed {result.indexed} documents")

            # Generate embeddings
            embed_result = await services.indexing.embed_collection("my-docs")
            print(f"Embedded {embed_result.embedded} documents")
    """

    def __init__(self, container: "ServiceContainer"):
        """Initialize IndexingService.

        Args:
            container: Service container with shared resources.
        """
        self._container = container

    async def index_collection(
        self,
        collection_name: str,
        force: bool = False,
    ) -> IndexResult:
        """Index all documents in a collection from the filesystem.

        Scans the collection's directory using its glob pattern, reads matching
        files, and stores them in the database with FTS5 indexing.

        Args:
            collection_name: Name of the collection to index.
            force: If True, reindex all documents even if unchanged.

        Returns:
            IndexResult with counts of indexed, skipped, and errored files.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            ValueError: If collection path does not exist on filesystem.
        """
        collection = self._container.collection_repo.get_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        collection_path = Path(collection.pwd)
        if not collection_path.exists():
            raise ValueError(f"Collection path does not exist: {collection_path}")

        logger.info(
            f"Indexing collection: name={collection.name!r}, "
            f"path={collection_path}, force={force}"
        )
        start_time = time.perf_counter()

        indexed_count = 0
        skipped_count = 0
        errors: list[tuple[str, str]] = []

        glob_pattern = collection.glob_pattern or "**/*.md"

        for file_path in collection_path.glob(glob_pattern):
            if not file_path.is_file():
                continue

            relative_path = str(file_path.relative_to(collection_path))

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, IOError) as e:
                errors.append((relative_path, str(e)))
                logger.warning(f"Failed to read file: {relative_path}: {e}")
                continue

            # Extract title from first markdown heading or use filename
            title = self._extract_title(content, file_path.stem)

            # Check if document has been modified (skip if unchanged and not forcing)
            if not force:
                from ..utils.hashing import sha256_hash

                content_hash = sha256_hash(content)
                if not self._container.document_repo.check_if_modified(
                    collection.id, relative_path, content_hash
                ):
                    skipped_count += 1
                    continue

            # Store document in database
            self._container.document_repo.add_or_update(
                collection.id,
                relative_path,
                title,
                content,
            )

            # Index in FTS5 only if document has sufficient quality
            doc_id = self._get_document_id(collection.id, relative_path)
            if doc_id is not None and is_indexable(content):
                self._container.fts_repo.index_document(doc_id, relative_path, content)

            indexed_count += 1
            logger.debug(f"Indexed: {relative_path} ({len(content)} chars)")

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Indexing complete: name={collection.name!r}, indexed={indexed_count}, "
            f"skipped={skipped_count}, errors={len(errors)}, {elapsed:.1f}ms"
        )

        return IndexResult(
            indexed=indexed_count,
            skipped=skipped_count,
            errors=errors,
        )

    async def embed_collection(
        self,
        collection_name: str,
        force: bool = False,
    ) -> EmbedResult:
        """Generate embeddings for all documents in a collection.

        Args:
            collection_name: Name of the collection to embed.
            force: If True, regenerate embeddings even if they exist.

        Returns:
            EmbedResult with embedding counts.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            RuntimeError: If vector storage is not available.
        """
        if not self._container.vec_available:
            raise RuntimeError(
                "Vector storage not available (sqlite-vec extension not loaded)"
            )

        collection = self._container.collection_repo.get_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        logger.info(f"Embedding collection: name={collection.name!r}, force={force}")
        start_time = time.perf_counter()

        # Get embedding generator
        embedding_generator = await self._container.get_embedding_generator()

        # Get all documents in collection
        cursor = self._container.db.execute(
            """
            SELECT d.path, d.hash, c.doc
            FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.collection_id = ? AND d.active = 1
            """,
            (collection.id,),
        )
        documents = cursor.fetchall()

        embedded_count = 0
        skipped_count = 0
        chunks_total = 0

        for doc in documents:
            # Check if already embedded (unless force)
            if not force and self._container.embedding_repo.has_embeddings(doc["hash"]):
                skipped_count += 1
                continue

            # Generate and store embeddings
            chunks_embedded = await embedding_generator.embed_document(
                doc["hash"],
                doc["doc"],
            )

            if chunks_embedded > 0:
                embedded_count += 1
                chunks_total += chunks_embedded
                logger.debug(f"Embedded: {doc['path']} ({chunks_embedded} chunks)")

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Embedding complete: name={collection.name!r}, embedded={embedded_count}, "
            f"skipped={skipped_count}, chunks={chunks_total}, {elapsed:.1f}ms"
        )

        return EmbedResult(
            embedded=embedded_count,
            skipped=skipped_count,
            chunks_total=chunks_total,
        )

    async def update_all_collections(self) -> dict[str, IndexResult]:
        """Update all collections by reindexing modified documents.

        Returns:
            Dictionary mapping collection name to IndexResult.
        """
        logger.info("Updating all collections")
        start_time = time.perf_counter()

        collections = self._container.collection_repo.list_all()
        results: dict[str, IndexResult] = {}

        for collection in collections:
            try:
                result = await self.index_collection(collection.name, force=False)
                results[collection.name] = result
            except Exception as e:
                logger.error(f"Failed to update collection {collection.name}: {e}")
                results[collection.name] = IndexResult(indexed=0, skipped=0, errors=[(collection.name, str(e))])

        elapsed = (time.perf_counter() - start_time) * 1000
        total_indexed = sum(r.indexed for r in results.values())
        logger.info(
            f"Update complete: {len(collections)} collections, "
            f"{total_indexed} documents indexed, {elapsed:.1f}ms"
        )

        return results

    async def cleanup_orphans(self) -> CleanupResult:
        """Clean up orphaned content and embeddings.

        Removes content and embeddings that are no longer referenced
        by any active document.

        Returns:
            CleanupResult with cleanup counts.
        """
        logger.info("Cleaning up orphaned data")
        start_time = time.perf_counter()

        # Find and remove orphaned content
        cursor = self._container.db.execute(
            """
            SELECT COUNT(*) as count FROM content
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            """
        )
        orphaned_content = cursor.fetchone()["count"]

        if orphaned_content > 0:
            self._container.db.execute(
                """
                DELETE FROM content
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
                """
            )

        # Find and remove orphaned embeddings
        cursor = self._container.db.execute(
            """
            SELECT COUNT(*) as count FROM content_vectors
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            """
        )
        orphaned_embeddings = cursor.fetchone()["count"]

        if orphaned_embeddings > 0:
            self._container.db.execute(
                """
                DELETE FROM content_vectors
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
                """
            )

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Cleanup complete: content={orphaned_content}, "
            f"embeddings={orphaned_embeddings}, {elapsed:.1f}ms"
        )

        return CleanupResult(
            orphaned_content=orphaned_content,
            orphaned_embeddings=orphaned_embeddings,
        )

    def _get_document_id(self, collection_id: int, path: str) -> int | None:
        """Get document ID for FTS indexing.

        Args:
            collection_id: Collection ID.
            path: Document path.

        Returns:
            Document ID or None if not found.
        """
        cursor = self._container.db.execute(
            "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
            (collection_id, path),
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        """Extract title from markdown content.

        Looks for the first line starting with '# ' and uses that as the title.
        Falls back to the provided fallback (typically the filename stem).

        Args:
            content: Markdown content to extract title from.
            fallback: Fallback title if no heading found.

        Returns:
            Extracted or fallback title.
        """
        for line in content.split("\n"):
            if line.startswith("# "):
                return line[2:].strip()
        return fallback
