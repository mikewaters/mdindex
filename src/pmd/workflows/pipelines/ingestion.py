"""Ingestion pipeline for document loading and persistence.

This module implements the IngestionPipeline that orchestrates:
1. Load documents - Fetch and prepare documents (via LoadingService)
2. Persist documents - Store content, FTS index, and metadata
3. Cleanup stale - Remove documents no longer in source

The pipeline uses LoadingService for document retrieval and preparation,
keeping the workflow focused on orchestration and progress tracking.

Design notes:
- Nodes are kept coarse (single persist node) to maintain transactional semantics
- LoadingService handles change detection, fetching, and metadata extraction
- The pipeline handles persistence and cleanup
- Progress callbacks are invoked during iteration
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from pmd.core.exceptions import SourceCollectionNotFoundError
from pmd.search.text import is_indexable

if TYPE_CHECKING:
    from pmd.app.types import (
        SourceCollectionRepositoryProtocol,
        DocumentRepositoryProtocol,
        FTSRepositoryProtocol,
        LoadingServiceProtocol,
        DatabaseProtocol,
    )
    from pmd.services.caching import DocumentCacher
    from pmd.workflows.contracts import IngestionRequest
    from pmd.services.loading import LoadedDocument
    from pmd.store.source_metadata import SourceMetadataRepository
    from pmd.metadata import DocumentMetadataRepository


class IngestionPipeline:
    """Pipeline for ingesting documents from a source collection.

    This pipeline orchestrates the full ingestion flow:
    - Load documents via LoadingService (change detection, fetch, metadata extraction)
    - Persist to storage (content, FTS, metadata)
    - Cleanup stale documents not present in source

    The pipeline uses streaming to process documents one at a time, reducing
    memory usage for large collections.

    Example:
        pipeline = IngestionPipeline(
            source_collection_repo=repo,
            document_repo=doc_repo,
            fts_repo=fts_repo,
            loader=loading_service,
            db=db,
        )

        request = IngestionRequest(collection_name="my-docs", force=False)
        result = await pipeline.execute(request)
        print(f"Indexed: {result.indexed}, Skipped: {result.skipped}")
    """

    def __init__(
        self,
        source_collection_repo: "SourceCollectionRepositoryProtocol",
        document_repo: "DocumentRepositoryProtocol",
        fts_repo: "FTSRepositoryProtocol",
        loader: "LoadingServiceProtocol",
        db: "DatabaseProtocol",
        cacher: "DocumentCacher | None" = None,
    ):
        """Initialize IngestionPipeline.

        Args:
            source_collection_repo: Repository for collection lookup.
            document_repo: Repository for document persistence.
            fts_repo: Repository for FTS indexing.
            loader: LoadingService for document retrieval.
            db: Database for direct SQL operations.
            cacher: Optional cacher for document content.
        """
        self._source_collection_repo = source_collection_repo
        self._document_repo = document_repo
        self._fts_repo = fts_repo
        self._loader = loader
        self._db = db
        self._cacher = cacher

    async def execute(self, request: "IngestionRequest") -> "IndexResult":
        """Execute the ingestion pipeline.

        Pipeline flow:
        1. Stream documents from LoadingService
        2. Persist each document (content + FTS + metadata)
        3. Cleanup stale documents not in source

        Args:
            request: Ingestion request with collection name and options.

        Returns:
            IndexResult with counts of indexed, skipped, and errored documents.

        Raises:
            SourceCollectionNotFoundError: If collection does not exist.
        """
        from pmd.services.indexing import IndexResult
        from pmd.store.source_metadata import SourceMetadataRepository
        from pmd.metadata import DocumentMetadataRepository

        collection_name = request.collection_name
        force = request.force
        context = request.context
        progress_callback = request.progress_callback

        # Log with trace context if available
        trace_info = f", trace_id={context.trace_id[:8]}" if context else ""
        logger.info(
            f"IngestionPipeline.execute: collection={collection_name!r}, "
            f"force={force}{trace_info}"
        )
        start_time = time.perf_counter()

        # Verify collection exists
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(
                f"Collection '{collection_name}' not found"
            )

        # Create metadata repositories
        source_metadata_repo = SourceMetadataRepository(self._db)  # type: ignore
        document_metadata_repo = DocumentMetadataRepository(self._db)  # type: ignore

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

                # Report progress
                if progress_callback:
                    progress_callback(processed, total_enumerated, doc.path)

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

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"IngestionPipeline complete: collection={collection_name!r}, "
            f"indexed={indexed_count}, skipped={total_skipped}, "
            f"errors={len(all_errors)}, stale={stale_count}, {elapsed:.1f}ms"
        )

        return IndexResult(
            indexed=indexed_count,
            skipped=total_skipped,
            errors=all_errors,
        )

    async def _persist_document(
        self,
        doc: "LoadedDocument",
        source_metadata_repo: "SourceMetadataRepository",
        document_metadata_repo: "DocumentMetadataRepository",
    ) -> str:
        """Persist a loaded document to storage.

        This node handles:
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
        from pmd.store.source_metadata import SourceMetadata
        from pmd.metadata import StoredDocumentMetadata

        # Store document content
        doc_result, is_new = self._document_repo.add_or_update(
            doc.source_collection_id,
            doc.path,
            doc.title,
            doc.content,
        )

        # Get document ID for FTS and metadata
        doc_id = self._get_document_id(doc.source_collection_id, doc.path)

        if doc_id is not None:
            # FTS5 indexing (only if content is indexable)
            if is_indexable(doc.content):
                self._fts_repo.index_document(doc_id, doc.path, doc.content)
            else:
                self._fts_repo.remove_from_index(doc_id)

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

        all_docs = self._document_repo.list_by_collection(
            source_collection.id, active_only=True
        )

        stale_count = 0
        for doc in all_docs:
            if doc.filepath not in seen_paths:
                # Mark document as inactive (soft delete)
                self._document_repo.delete(source_collection.id, doc.filepath)
                # Remove from FTS
                doc_id = self._get_document_id(source_collection.id, doc.filepath)
                if doc_id is not None:
                    self._fts_repo.remove_from_index(doc_id)
                # Remove from cache
                if self._cacher and self._cacher.enabled:
                    self._cacher.remove_document(collection_name, doc.filepath)
                stale_count += 1

        return stale_count

    def _get_document_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID for a path.

        Args:
            source_collection_id: Source collection ID.
            path: Document path.

        Returns:
            Document ID or None if not found.
        """
        return self._document_repo.get_id(source_collection_id, path)

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
        cached_uri = self._cacher.cache_document(
            collection_name,
            doc.path,
            doc.content,
        )

        # Create new DocumentReference with cached URI
        new_ref = replace(doc.ref, uri=cached_uri)

        # Return new LoadedDocument with updated reference
        return replace(doc, ref=new_ref)
