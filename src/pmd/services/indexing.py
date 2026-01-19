"""Indexing service for document indexing and embedding operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Awaitable, TYPE_CHECKING

from loguru import logger

from pmd.core.exceptions import SourceCollectionNotFoundError
from pmd.core.types import SourceCollection
from pmd.sources import (
    DocumentSource,
    get_default_registry,
)
from pmd.extraction.types import ExtractedMetadata, StoredDocumentMetadata
from pmd.extraction.registry import get_default_profile_registry
from pmd.store.repositories.metadata import DocumentMetadataRepository
from pmd.app.protocols import (
    EmbeddingGeneratorProtocol,
    LoadingServiceProtocol,
)
from pmd.store import IndexFacade
from pmd.search.text import is_indexable

if TYPE_CHECKING:
    from pmd.services.loading import LoadedDocument
    from pmd.store.caching import ResourceCacher
    from pmd.store.repositories.source_metadata import SourceMetadataRepository


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
            facade=IndexFacade(db),
            loader=loading_service,
        )
        result = await indexing.index_collection("my-docs", source=source)
    """

    def __init__(
        self,
        facade: IndexFacade,
        loader: LoadingServiceProtocol,
        embedding_generator_factory: Callable[[], Awaitable[EmbeddingGeneratorProtocol]] | None = None,
        cacher: "ResourceCacher | None" = None,
    ):
        """Initialize IndexingService.

        Args:
            facade: Facade for indexing data operations.
            loader: Loading service for document retrieval.
            embedding_generator_factory: Async factory for embedding generator.
            cacher: Optional document cacher for local file caching.
        """
        self._data = facade
        self._loader = loader
        self._embedding_generator_factory = embedding_generator_factory
        self._cacher = cacher

    @property
    def vec_available(self) -> bool:
        """Check if vector storage is available."""
        return self._data.db.vec_available

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
        source_collection = self._data.get_collection_by_name(collection_name)
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
        """Index collection by loading and persisting documents.

        This method orchestrates the full ingestion flow:
        - Stream documents from LoadingService
        - Persist each document (content + FTS + metadata)
        - Cleanup stale documents not present in source

        Args:
            collection_name: Name of the collection to index.
            source: Optional source override (currently not used).
            force: If True, reload all documents.

        Returns:
            IndexResult with counts.
        """
        from pmd.store.repositories.source_metadata import SourceMetadataRepository

        # Verify collection exists
        source_collection = self._data.get_collection_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(
                f"Collection '{collection_name}' not found"
            )

        # Create metadata repositories
        source_metadata_repo = SourceMetadataRepository(self._data.db)  # type: ignore
        document_metadata_repo = DocumentMetadataRepository(self._data.db)  # type: ignore

        # Stream documents from loader
        load_result = await self._loader.load_collection_stream(
            collection_name,
            source=None,  # Let loader resolve source
            force=force,
        )

        # Track progress
        indexed_count = 0
        skipped_count = 0
        persist_errors: list[tuple[str, str]] = []
        total_enumerated = len(load_result.enumerated_paths)

        # Process each loaded document
        processed = 0
        async for doc in load_result.documents:
            processed += 1
            try:
                # Cache the document content if cacher is enabled
                if self._cacher and self._cacher.enabled:
                    doc = self._cache_document(collection_name, doc)

                result = await self._persist_document(
                    doc,
                    source_metadata_repo,
                    document_metadata_repo,
                )
                if result == "indexed":
                    indexed_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                persist_errors.append((doc.path, str(e)))
                logger.warning(f"Failed to persist document: {doc.path}: {e}")

        # Cleanup stale documents
        stale_count = await self._cleanup_stale_documents(
            collection_name,
            load_result.enumerated_paths,
        )
        if stale_count > 0:
            logger.info(f"Marked {stale_count} stale documents as inactive")

        # Combine errors from loader and persistence
        all_errors = load_result.errors + persist_errors

        # Documents that were enumerated but not loaded (unchanged) are skipped
        # Plus documents that were loaded but not persisted (content hash unchanged)
        loader_skipped = total_enumerated - processed - len(load_result.errors)
        total_skipped = skipped_count + loader_skipped

        return IndexResult(
            indexed=indexed_count,
            skipped=total_skipped,
            errors=all_errors,
        )

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
        source_collection = self._data.get_collection_by_name(collection_name)
        if not source_collection:
            return 0

        all_docs = self._data.list_documents_by_collection(source_collection.id, active_only=True)

        stale_count = 0
        for doc in all_docs:
            if doc.filepath not in seen_paths:
                # Mark document as inactive (soft delete)
                self._data.delete_document(source_collection.id, doc.filepath)
                # Remove from FTS
                doc_id = self._data.get_document_id(source_collection.id, doc.filepath)
                if doc_id is not None:
                    self._data.remove_from_search_index(doc_id)
                # Remove from cache
                if self._cacher and self._cacher.enabled:
                    self._cacher.remove_resource(collection_name, doc.filepath)
                stale_count += 1

        return stale_count

    async def _persist_document(
        self,
        doc: "LoadedDocument",
        source_metadata_repo: "SourceMetadataRepository",
        document_metadata_repo: DocumentMetadataRepository,
    ) -> str:
        """Persist a loaded document to storage.

        This handles:
        - Content storage (document table + content table)
        - FTS5 indexing (if content is indexable)
        - Source metadata (fetch info, etags, etc.)
        - Document metadata (tags, attributes)

        Args:
            doc: Document that has been loaded and prepared.
            source_metadata_repo: Repository for source metadata.
            document_metadata_repo: Repository for document metadata.

        Returns:
            "indexed" if persisted, "skipped" if content unchanged.
        """
        from pmd.store.repositories.source_metadata import SourceMetadata

        # Store document content
        doc_result, is_new = self._data.add_or_update_document(
            doc.source_collection_id,
            doc.path,
            doc.title,
            doc.content,
        )

        # Get document ID for FTS and metadata
        doc_id = self._data.get_document_id(doc.source_collection_id, doc.path)

        if doc_id is not None:
            # FTS5 indexing (only if content is indexable)
            if is_indexable(doc.content):
                self._data.index_document_for_search(doc_id, doc.path, doc.content)
            else:
                self._data.remove_from_search_index(doc_id)

            # Store source metadata
            metadata = SourceMetadata(
                document_id=doc_id,
                source_uri=doc.ref.uri,
                last_fetched_at=datetime.utcnow().isoformat(),
                etag=doc.fetch_result.metadata.get("etag"),
                last_modified=doc.fetch_result.metadata.get("last_modified"),
                fetch_duration_ms=doc.fetch_duration_ms,
                http_status=doc.fetch_result.metadata.get("http_status"),
                content_type=doc.content_type,
                extra=doc.fetch_result.metadata,
            )
            source_metadata_repo.upsert(metadata)

            # Store document metadata (tags, attributes)
            if doc.extracted_metadata:
                stored = StoredDocumentMetadata(
                    document_id=doc_id,
                    profile_name=doc.extracted_metadata.extraction_source or "unknown",
                    tags=doc.extracted_metadata.tags,
                    source_tags=doc.extracted_metadata.source_tags,
                    attributes=doc.extracted_metadata.attributes,
                    extracted_at=datetime.utcnow().isoformat(),
                )
                document_metadata_repo.upsert(stored)

        logger.debug(f"Persisted: {doc.path} ({len(doc.content)} chars)")
        return "indexed"

    def _cache_document(
        self,
        collection_name: str,
        doc: "LoadedDocument",
    ) -> "LoadedDocument":
        """Cache document content and return updated document with cached URI.

        Args:
            collection_name: Name of the collection.
            doc: Document to cache.

        Returns:
            LoadedDocument with updated ref.uri pointing to cached file.
        """
        from dataclasses import replace

        from pmd.sources.content.base import DocumentReference

        if not self._cacher:
            return doc

        # Cache the content and get the new URI
        cached_uri = self._cacher.cache_resource(
            collection_name,
            doc.path,
            doc.content,
        )

        # Create new DocumentReference with cached URI
        new_ref = replace(doc.ref, uri=cached_uri)

        # Return new LoadedDocument with updated reference
        return replace(doc, ref=new_ref)

    async def embed_collection(
        self,
        collection_name: str,
        force: bool = False,
    ) -> EmbedResult:
        """Generate embeddings for all documents in a collection.

        This method orchestrates the embedding flow:
        - Verify prerequisites (vector storage, LLM availability)
        - Query documents needing embeddings
        - For each document: chunk, embed, store

        Args:
            collection_name: Name of the collection to embed.
            force: If True, regenerate embeddings even if they exist.

        Returns:
            EmbedResult with embedding counts.

        Raises:
            SourceCollectionNotFoundError: If collection does not exist.
            RuntimeError: If vector storage is not available.
        """
        # Validate prerequisites
        if not self._data.db.vec_available:
            raise RuntimeError(
                "Vector storage not available (sqlite-vec extension not loaded)"
            )

        # Verify collection exists
        source_collection = self._data.get_collection_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(
                f"Collection '{collection_name}' not found"
            )

        logger.info(
            f"Embedding collection: name={collection_name!r}, force={force}"
        )
        start_time = time.perf_counter()

        # Get embedding generator
        embedding_generator = await self._embedding_generator_factory()

        # Query documents needing embeddings
        embed_targets = self._list_embed_targets(source_collection.id, force)
        total_docs = len(embed_targets)

        logger.debug(f"Found {total_docs} documents to process for embedding")

        # Process each document
        embedded_count = 0
        skipped_count = 0
        chunks_total = 0

        for idx, (path, doc_hash, content) in enumerate(embed_targets):
            # Check if already embedded (unless force)
            if not force and self._data.has_embeddings(doc_hash):
                skipped_count += 1
                continue

            try:
                # Generate and store embeddings
                # The generator handles chunking, embedding, and storage internally
                chunks_embedded = await embedding_generator.embed_document(
                    doc_hash,
                    content,
                    force=force,
                )

                if chunks_embedded > 0:
                    embedded_count += 1
                    chunks_total += chunks_embedded
                    logger.debug(f"Embedded: {path} ({chunks_embedded} chunks)")

            except Exception as e:
                # Log error but continue with other documents
                logger.warning(f"Failed to embed document: {path}: {e}")

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Embedding complete: name={collection_name!r}, "
            f"embedded={embedded_count}, skipped={skipped_count}, "
            f"chunks={chunks_total}, {elapsed:.1f}ms"
        )

        return EmbedResult(
            embedded=embedded_count,
            skipped=skipped_count,
            chunks_total=chunks_total,
        )

    def _list_embed_targets(
        self,
        source_collection_id: int,
        force: bool,
    ) -> list[tuple[str, str, str]]:
        """Query documents needing embeddings.

        Args:
            source_collection_id: Collection to query.
            force: If True, include all documents (not just those missing embeddings).

        Returns:
            List of (path, hash, content) tuples.
        """
        # Query all active documents in collection with their content
        rows = self._data.list_active_with_content(source_collection_id)
        return list(rows)

    async def update_all_collections(self, embed: bool = False) -> dict[str, IndexResult]:
        """Update all collections by reindexing modified documents.

        Args:
            embed: If True, trigger embedding generation after indexing.

        Returns:
            Dictionary mapping collection name to IndexResult.
        """
        logger.info("Updating all collections")
        start_time = time.perf_counter()

        source_collections = self._data.list_all_collections()
        results: dict[str, IndexResult] = {}

        for source_collection in source_collections:
            try:
                source = get_default_registry().create_source(source_collection)
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
        orphaned_content = self._data.delete_orphaned_content()

        # Find and remove orphaned embeddings
        orphaned_embeddings = self._data.delete_orphaned_embeddings()

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
        return self._data.get_document_id(source_collection_id, path)

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
        metadata_repo = DocumentMetadataRepository(self._data.db) # type: ignore

        # Build query for documents needing metadata extraction
        # Note: Content is stored separately in the content table, joined via hash
        if force:
            # Re-extract all
            if collection_name:
                cursor = self._data.db.execute(
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
                cursor = self._data.db.execute(
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
                cursor = self._data.db.execute(
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
                cursor = self._data.db.execute(
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
