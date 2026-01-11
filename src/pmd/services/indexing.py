"""Indexing service for document indexing and embedding operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Awaitable

from loguru import logger

from pmd.core.exceptions import SourceCollectionNotFoundError
from pmd.core.types import SourceCollection
from pmd.sources import (
    DocumentSource,
    SourceRegistry,
    get_default_registry,
)
from pmd.metadata import (
    ExtractedMetadata, get_default_profile_registry, DocumentMetadataRepository
)
from pmd.app.types import (
    SourceCollectionRepositoryProtocol,
    DatabaseProtocol,
    DocumentRepositoryProtocol,
    EmbeddingGeneratorProtocol,
    EmbeddingRepositoryProtocol,
    FTSRepositoryProtocol,
    LoadingServiceProtocol,
)



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

        indexing = IndexingService(
            db=db,
            source_collection_repo=source_collection_repo,
            document_repo=document_repo,
            fts_repo=fts_repo,
            embedding_repo=embedding_repo,
            loader=loading_service,
        )
        result = await indexing.index_collection("my-docs", source=source)
    """

    def __init__(
        self,
        db: DatabaseProtocol,
        source_collection_repo: SourceCollectionRepositoryProtocol,
        document_repo: DocumentRepositoryProtocol,
        fts_repo: FTSRepositoryProtocol,
        loader: LoadingServiceProtocol,
        embedding_repo: EmbeddingRepositoryProtocol | None = None,
        embedding_generator_factory: Callable[[], Awaitable[EmbeddingGeneratorProtocol]] | None = None,
        llm_available_check: Callable[[], Awaitable[bool]] | None = None,
        source_registry: SourceRegistry | None = None,
    ):
        """Initialize IndexingService.

        Args:
            db: Database for direct SQL operations.
            source_collection_repo: Repository for collection operations.
            document_repo: Repository for document operations.
            fts_repo: Repository for FTS indexing.
            loader: Loading service for document retrieval.
            embedding_repo: Repository for embedding storage.
            embedding_generator_factory: Async factory for embedding generator.
            llm_available_check: Async function to check if LLM is available.
            source_registry: Optional source registry for creating sources.
        """
        self._db = db
        self._source_collection_repo = source_collection_repo
        self._document_repo = document_repo
        self._fts_repo = fts_repo
        self._loader = loader
        self._embedding_repo = embedding_repo
        self._embedding_generator_factory = embedding_generator_factory
        self._llm_available_check = llm_available_check
        self._source_registry = source_registry or get_default_registry()

    @property
    def vec_available(self) -> bool:
        """Check if vector storage is available."""
        return self._db.vec_available

    async def index_collection(
        self,
        collection_name: str,
        source: DocumentSource | None = None,
        force: bool = False,
        embed: bool = False,
    ) -> IndexResult:
        """Index all documents in a collection from its configured source.

        Enumerates documents from the collection's source (filesystem, HTTP, etc.),
        fetches their content, and stores them in the database with FTS5 indexing.

        Args:
            collection_name: Name of the collection to index.
            source: Optional document source; resolved from collection if None.
            force: If True, reindex all documents even if unchanged.
            embed: If True, trigger embedding generation after indexing.

        Returns:
            IndexResult with counts of indexed, skipped, and errored files.

        Raises:
            SourceCollectionNotFoundError: If collection does not exist.
            RuntimeError: If no loader is configured.
        """
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(f"Collection '{collection_name}' not found")

        logger.info(
            f"Indexing collection: name={source_collection.name!r}, "
            f"source_type={source_collection.source_type!r}, force={force}"
        )
        start_time = time.perf_counter()

        result = await self._index_via_loader(
            collection_name=collection_name,
            source=source,
            force=force,
        )

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Indexing complete: name={source_collection.name!r}, indexed={result.indexed}, "
            f"skipped={result.skipped}, errors={len(result.errors)}, {elapsed:.1f}ms"
        )

        if embed:
            await self.embed_collection(source_collection.name, force=force)

        return result

    async def _index_via_loader(
        self,
        collection_name: str,
        source: DocumentSource | None,
        force: bool,
    ) -> IndexResult:
        """Index collection using the IngestionPipeline workflow.

        This method delegates to IngestionPipeline for document loading,
        persistence, and cleanup. The pipeline uses LoadingService internally.

        Args:
            collection_name: Name of the collection to index.
            source: Optional source override (currently not passed to pipeline).
            force: If True, reload all documents.

        Returns:
            IndexResult with counts.
        """
        from pmd.workflows import IngestionPipeline, IngestionRequest

        # Create pipeline with required dependencies
        pipeline = IngestionPipeline(
            source_collection_repo=self._source_collection_repo,
            document_repo=self._document_repo,
            fts_repo=self._fts_repo,
            loader=self._loader,  # type: ignore
            db=self._db,
        )

        # Execute the ingestion workflow
        request = IngestionRequest(
            collection_name=collection_name,
            force=force,
        )

        return await pipeline.execute(request)

    async def _cleanup_stale_documents(
        self,
        collection_name: str,
        seen_paths: set[str],
    ) -> int:
        """Mark documents not in seen_paths as inactive.

        Args:
            collection_name: Collection being indexed.
            seen_paths: Paths that were enumerated by the loader.

        Returns:
            Number of documents marked inactive.
        """
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            return 0

        all_docs = self._document_repo.list_by_collection(source_collection.id, active_only=True)

        stale_count = 0
        for doc in all_docs:
            if doc.filepath not in seen_paths:
                # Mark document as inactive (soft delete)
                self._document_repo.delete(source_collection.id, doc.filepath)
                # Remove from FTS
                doc_id = self._get_document_id(source_collection.id, doc.filepath)
                if doc_id is not None:
                    self._fts_repo.remove_from_index(doc_id)
                stale_count += 1

        return stale_count

    async def embed_collection(
        self,
        collection_name: str,
        force: bool = False,
    ) -> EmbedResult:
        """Generate embeddings for all documents in a collection.

        This method delegates to EmbeddingPipeline for the actual embedding work.

        Args:
            collection_name: Name of the collection to embed.
            force: If True, regenerate embeddings even if they exist.

        Returns:
            EmbedResult with embedding counts.

        Raises:
            SourceCollectionNotFoundError: If collection does not exist.
            RuntimeError: If vector storage or LLM provider is not available.
        """
        from pmd.workflows import EmbeddingPipeline, EmbedRequest

        # Validate prerequisites before creating pipeline
        if not self._embedding_generator_factory:
            raise RuntimeError("Embedding generator not configured")

        # Create pipeline with required dependencies
        pipeline = EmbeddingPipeline(
            source_collection_repo=self._source_collection_repo,
            embedding_generator_factory=self._embedding_generator_factory,
            embedding_repo=self._embedding_repo,
            db=self._db,
            llm_available_check=self._llm_available_check,
        )

        # Execute the embedding workflow
        request = EmbedRequest(
            collection_name=collection_name,
            force=force,
        )

        return await pipeline.execute(request)

    async def update_all_collections(self, embed: bool = False) -> dict[str, IndexResult]:
        """Update all collections by reindexing modified documents.

        Args:
            embed: If True, trigger embedding generation after indexing.

        Returns:
            Dictionary mapping collection name to IndexResult.
        """
        logger.info("Updating all collections")
        start_time = time.perf_counter()

        source_collections = self._source_collection_repo.list_all()
        results: dict[str, IndexResult] = {}

        for source_collection in source_collections:
            try:
                source = self._source_registry.create_source(source_collection)
                result = await self.index_collection(
                    source_collection.name,
                    force=False,
                    embed=embed,
                    source=source,
                )
                results[source_collection.name] = result
            except Exception as e:
                logger.error(f"Failed to update collection {source_collection.name}: {e}")
                results[source_collection.name] = IndexResult(indexed=0, skipped=0, errors=[(source_collection.name, str(e))])

        elapsed = (time.perf_counter() - start_time) * 1000
        total_indexed = sum(r.indexed for r in results.values())
        logger.info(
            f"Update complete: {len(source_collections)} collections, "
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
        cursor = self._db.execute(
            """
            SELECT COUNT(*) as count FROM content
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            """
        )
        orphaned_content = cursor.fetchone()["count"]

        if orphaned_content > 0:
            self._db.execute(
                """
                DELETE FROM content
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
                """
            )

        # Find and remove orphaned embeddings
        cursor = self._db.execute(
            """
            SELECT COUNT(*) as count FROM content_vectors
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            """
        )
        orphaned_embeddings = cursor.fetchone()["count"]

        if orphaned_embeddings > 0:
            self._db.execute(
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

    def _get_document_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID for FTS indexing.

        Args:
            source_collection_id: Source collection ID.
            path: Document path.

        Returns:
            Document ID or None if not found.
        """
        cursor = self._db.execute(
            "SELECT id FROM documents WHERE source_collection_id = ? AND path = ?",
            (source_collection_id, path),
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    def _extract_metadata_via_profiles(
        self,
        content: str,
        path: str,
        source_collection: SourceCollection,
    ) -> ExtractedMetadata | None:
        """Extract document metadata using profile auto-detection."""
        try:
            registry = get_default_profile_registry()

            profile_name = None
            if source_collection.source_config:
                profile_name = source_collection.source_config.get("metadata_profile")

            if profile_name:
                profile = registry.get(profile_name)
                if not profile:
                    logger.warning(
                        f"Configured profile '{profile_name}' not found, using auto-detection"
                    )
                    profile = registry.detect_or_default(content, path)
            else:
                profile = registry.detect_or_default(content, path)

            extracted = profile.extract_metadata(content, path)
            if not extracted.extraction_source:
                extracted.extraction_source = profile.name

            if extracted.tags:
                logger.debug(
                    f"Extracted metadata: path={path!r}, profile={profile.name}, "
                    f"tags={len(extracted.tags)}"
                )

            return extracted

        except Exception as exc:
            # Don't fail indexing if metadata extraction fails
            logger.warning(f"Failed to extract metadata for {path}: {exc}")
            return None

    def _persist_document_metadata(
        self,
        doc_id: int,
        metadata: ExtractedMetadata,
        metadata_repo: DocumentMetadataRepository,
    ) -> None:
        """Persist extracted metadata to the repository."""
        from datetime import datetime
        from pmd.metadata import StoredDocumentMetadata

        stored = StoredDocumentMetadata(
            document_id=doc_id,
            profile_name=metadata.extraction_source or "unknown",
            tags=metadata.tags,
            source_tags=metadata.source_tags,
            attributes=metadata.attributes,
            extracted_at=datetime.utcnow().isoformat(),
        )
        metadata_repo.upsert(stored)

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

    def backfill_metadata(
        self,
        collection_name: str | None = None,
        force: bool = False,
    ) -> dict:
        """Backfill document metadata for existing documents.

        This migration function extracts and stores metadata for documents
        that were indexed before the metadata tables existed, or for
        documents that need metadata re-extraction.

        Args:
            collection_name: Optional collection to limit backfill to.
                           If None, backfills all collections.
            force: If True, re-extract metadata even if already present.

        Returns:
            Dict with backfill statistics:
            - processed: Number of documents processed
            - updated: Number of documents with new/updated metadata
            - skipped: Number of documents skipped (already have metadata)
            - errors: List of (path, error) for failed documents
        """

        logger.info(f"Starting metadata backfill (collection={collection_name}, force={force})")
        start_time = time.time()

        stats = {
            "processed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": [],
        }

        # Ensure document_metadata table exists
        metadata_repo = DocumentMetadataRepository(self._db) # type: ignore

        # Build query for documents needing metadata extraction
        # Note: Content is stored separately in the content table, joined via hash
        if force:
            # Re-extract all
            if collection_name:
                cursor = self._db.execute(
                    """
                    SELECT d.id, d.path, ct.doc as body, c.name as collection_name, c.source_config
                    FROM documents d
                    JOIN source_collections c ON d.source_collection_id = c.id
                    JOIN content ct ON d.hash = ct.hash
                    WHERE d.active = 1 AND c.name = ?
                    """,
                    (collection_name,),
                )
            else:
                cursor = self._db.execute(
                    """
                    SELECT d.id, d.path, ct.doc as body, c.name as collection_name, c.source_config
                    FROM documents d
                    JOIN source_collections c ON d.source_collection_id = c.id
                    JOIN content ct ON d.hash = ct.hash
                    WHERE d.active = 1
                    """
                )
        else:
            # Only documents without metadata
            if collection_name:
                cursor = self._db.execute(
                    """
                    SELECT d.id, d.path, ct.doc as body, c.name as collection_name, c.source_config
                    FROM documents d
                    JOIN source_collections c ON d.source_collection_id = c.id
                    JOIN content ct ON d.hash = ct.hash
                    LEFT JOIN document_metadata dm ON d.id = dm.document_id
                    WHERE d.active = 1 AND c.name = ? AND dm.id IS NULL
                    """,
                    (collection_name,),
                )
            else:
                cursor = self._db.execute(
                    """
                    SELECT d.id, d.path, ct.doc as body, c.name as collection_name, c.source_config
                    FROM documents d
                    JOIN source_collections c ON d.source_collection_id = c.id
                    JOIN content ct ON d.hash = ct.hash
                    LEFT JOIN document_metadata dm ON d.id = dm.document_id
                    WHERE d.active = 1 AND dm.id IS NULL
                    """
                )

        rows = cursor.fetchall()
        total = len(rows)
        logger.info(f"Found {total} documents to process")

        for row in rows:
            doc_id = row["id"]
            path = row["path"]
            body = row["body"]
            source_config = row["source_config"]

            stats["processed"] += 1

            if not body:
                stats["skipped"] += 1
                continue

            try:
                # Create a minimal collection object for the extraction
                import json
                parsed_config = json.loads(source_config) if source_config else {}
                source_collection = SourceCollection(
                    id=0,  # Not needed for extraction
                    name=row["collection_name"],
                    pwd="",  # Not needed for extraction
                    glob_pattern="",  # Not needed for extraction
                    created_at="",  # Not needed for extraction
                    updated_at="",  # Not needed for extraction
                    source_config=parsed_config,
                )

                metadata = self._extract_metadata_via_profiles(body, path, source_collection)
                if metadata:
                    self._persist_document_metadata(doc_id, metadata, metadata_repo)
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1

                if stats["processed"] % 100 == 0:
                    logger.info(f"Backfill progress: {stats['processed']}/{total}")

            except Exception as e:
                stats["errors"].append((path, str(e)))
                logger.warning(f"Failed to extract metadata for {path}: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"Metadata backfill complete: processed={stats['processed']}, "
            f"updated={stats['updated']}, skipped={stats['skipped']}, "
            f"errors={len(stats['errors'])}, time={elapsed:.1f}s"
        )

        return stats
