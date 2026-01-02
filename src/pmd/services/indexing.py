"""Indexing service for document indexing and embedding operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ..core.exceptions import CollectionNotFoundError
from ..core.types import Collection
from ..search.text import is_indexable
from ..sources import (
    DocumentReference,
    DocumentSource,
    SourceConfig,
    SourceFetchError,
    SourceListError,
    get_default_registry, 
    FetchResult
)
from ..store.source_metadata import SourceMetadata, SourceMetadataRepository

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
        embed: bool = False,
    ) -> IndexResult:
        """Index all documents in a collection from its configured source.

        Enumerates documents from the collection's source (filesystem, HTTP, etc.),
        fetches their content, and stores them in the database with FTS5 indexing.

        Args:
            collection_name: Name of the collection to index.
            force: If True, reindex all documents even if unchanged.
            embed: If True, trigger embedding generation after indexing.

        Returns:
            IndexResult with counts of indexed, skipped, and errored files.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            SourceListError: If the source cannot enumerate documents.
        """
        collection = self._container.collection_repo.get_by_name(collection_name)
        if not collection:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        # Create document source from collection configuration
        source = self._create_source(collection)

        logger.info(
            f"Indexing collection: name={collection.name!r}, "
            f"source_type={collection.source_type!r}, force={force}"
        )
        start_time = time.perf_counter()

        indexed_count = 0
        skipped_count = 0
        errors: list[tuple[str, str]] = []

        # Get source metadata repository for tracking fetch info
        from ..store.source_metadata import SourceMetadataRepository
        metadata_repo = SourceMetadataRepository(self._container.db)

        try:
            for ref in source.list_documents():
                try:
                    result = await self._index_document(
                        collection=collection,
                        source=source,
                        ref=ref,
                        metadata_repo=metadata_repo,
                        force=force,
                    )
                    if result == "indexed":
                        indexed_count += 1
                    elif result == "skipped":
                        skipped_count += 1
                except SourceFetchError as e:
                    errors.append((ref.path, str(e)))
                    logger.warning(f"Failed to fetch document: {ref.path}: {e}")
                except Exception as e:
                    errors.append((ref.path, str(e)))
                    logger.warning(f"Failed to index document: {ref.path}: {e}")

        except SourceListError as e:
            logger.error(f"Failed to list documents from source: {e}")
            raise

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Indexing complete: name={collection.name!r}, indexed={indexed_count}, "
            f"skipped={skipped_count}, errors={len(errors)}, {elapsed:.1f}ms"
        )

        if embed:
            await self.embed_collection(collection.name, force=force)

        return IndexResult(
            indexed=indexed_count,
            skipped=skipped_count,
            errors=errors,
        )

    async def _index_document(
        self,
        collection: Collection,
        source: DocumentSource,
        ref: DocumentReference,
        metadata_repo: "SourceMetadataRepository",
        force: bool,
    ) -> str:
        """Index a single document from a source.

        Args:
            collection: The collection being indexed.
            source: Document source to fetch from.
            ref: Reference to the document.
            metadata_repo: Repository for source metadata.
            force: If True, reindex even if unchanged.

        Returns:
            "indexed" if document was indexed, "skipped" if unchanged.
        """
        from ..store.source_metadata import SourceMetadataRepository
        from ..utils.hashing import sha256_hash

        # Get existing document and metadata for change detection
        existing_doc = self._container.document_repo.get(collection.id, ref.path)
        doc_id = self._get_document_id(collection.id, ref.path) if existing_doc else None
        stored_metadata = {}

        if doc_id and not force:
            meta = metadata_repo.get_by_document(doc_id)
            if meta:
                stored_metadata = meta.extra.copy()
                stored_metadata["etag"] = meta.etag
                stored_metadata["last_modified"] = meta.last_modified

            # Check if source says document is modified
            if not await source.check_modified(ref, stored_metadata):
                return "skipped"

        # Fetch content
        fetch_start = time.perf_counter()
        fetch_result = await source.fetch_content(ref)
        fetch_duration_ms = int((time.perf_counter() - fetch_start) * 1000)

        content = fetch_result.content

        # Check content hash if we have existing document
        if existing_doc and not force:
            content_hash = sha256_hash(content)
            if existing_doc.hash == content_hash:
                # Content unchanged, but update metadata
                if doc_id:
                    self._update_source_metadata(
                        metadata_repo, doc_id, ref, fetch_result, fetch_duration_ms
                    )
                return "skipped"

        # Extract title
        title = ref.title or self._extract_title(content, Path(ref.path).stem)

        # Store document
        doc_result, is_new = self._container.document_repo.add_or_update(
            collection.id,
            ref.path,
            title,
            content,
        )

        # Get document ID for FTS indexing
        doc_id = self._get_document_id(collection.id, ref.path)

        # Index in FTS5 if document has sufficient quality
        if doc_id is not None:
            if is_indexable(content):
                self._container.fts_repo.index_document(doc_id, ref.path, content)
            else:
                self._container.fts_repo.remove_from_index(doc_id)

            # Store source metadata for remote sources
            self._update_source_metadata(
                metadata_repo, doc_id, ref, fetch_result, fetch_duration_ms
            )

            # Extract and store document metadata (tags, attributes)
            self._extract_and_store_metadata(doc_id, content, ref.path, collection)

        logger.debug(f"Indexed: {ref.path} ({len(content)} chars)")
        return "indexed"

    def _create_source(self, collection: Collection) -> DocumentSource:
        """Create a document source from collection configuration.

        Args:
            collection: Collection to create source for.

        Returns:
            DocumentSource instance.
        """
        registry = get_default_registry()

        # Build source config from collection
        source_uri = collection.get_source_uri()
        extra = collection.get_source_config_dict()

        config = SourceConfig(uri=source_uri, extra=extra)

        return registry.resolve(source_uri, config)

    def _update_source_metadata(
        self,
        metadata_repo: "SourceMetadataRepository",
        doc_id: int,
        ref: DocumentReference,
        fetch_result: "FetchResult",
        fetch_duration_ms: int,
    ) -> None:
        """Update source metadata after fetching a document.

        Args:
            metadata_repo: Repository for source metadata.
            doc_id: Document ID.
            ref: Document reference.
            fetch_result: Result from fetching content.
            fetch_duration_ms: How long the fetch took.
        """
        from ..sources import FetchResult

        metadata = SourceMetadata(
            document_id=doc_id,
            source_uri=ref.uri,
            last_fetched_at=datetime.utcnow().isoformat(),
            etag=fetch_result.metadata.get("etag"),
            last_modified=fetch_result.metadata.get("last_modified"),
            fetch_duration_ms=fetch_duration_ms,
            http_status=fetch_result.metadata.get("http_status"),
            content_type=fetch_result.content_type,
            extra=fetch_result.metadata,
        )
        metadata_repo.upsert(metadata)

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
            RuntimeError: If vector storage or LLM provider is not available.
        """
        if not self._container.vec_available:
            raise RuntimeError(
                "Vector storage not available (sqlite-vec extension not loaded)"
            )

        if not await self._container.is_llm_available():
            raise RuntimeError("LLM provider not available (is it running?)")

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
                force=force,
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

    async def update_all_collections(self, embed: bool = False) -> dict[str, IndexResult]:
        """Update all collections by reindexing modified documents.

        Args:
            embed: If True, trigger embedding generation after indexing.

        Returns:
            Dictionary mapping collection name to IndexResult.
        """
        logger.info("Updating all collections")
        start_time = time.perf_counter()

        collections = self._container.collection_repo.list_all()
        results: dict[str, IndexResult] = {}

        for collection in collections:
            try:
                result = await self.index_collection(
                    collection.name,
                    force=False,
                    embed=embed,
                )
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

    def _extract_and_store_metadata(
        self,
        doc_id: int,
        content: str,
        path: str,
        collection: Collection,
    ) -> None:
        """Extract and store document metadata.

        Uses the metadata profile registry to detect the appropriate
        profile, extract metadata, and store it.

        Args:
            doc_id: Document ID.
            content: Document content.
            path: Document path.
            collection: Collection the document belongs to.
        """
        from datetime import datetime

        from ..search.metadata import get_default_profile_registry
        from ..store.document_metadata import DocumentMetadataRepository, StoredDocumentMetadata

        try:
            # Get profile registry and detect appropriate profile
            registry = get_default_profile_registry()

            # Check if collection has a configured profile
            profile_name = None
            if collection.source_config:
                profile_name = collection.source_config.get("metadata_profile")

            if profile_name:
                profile = registry.get(profile_name)
                if not profile:
                    logger.warning(
                        f"Configured profile '{profile_name}' not found, using auto-detection"
                    )
                    profile = registry.detect_or_default(content, path)
            else:
                profile = registry.detect_or_default(content, path)

            # Extract metadata
            extracted = profile.extract_metadata(content, path)

            # Store metadata
            metadata_repo = DocumentMetadataRepository(self._container.db)
            stored = StoredDocumentMetadata(
                document_id=doc_id,
                profile_name=profile.name,
                tags=extracted.tags,
                source_tags=extracted.source_tags,
                attributes=extracted.attributes,
                extracted_at=datetime.utcnow().isoformat(),
            )
            metadata_repo.upsert(stored)

            if extracted.tags:
                logger.debug(
                    f"Extracted metadata: path={path!r}, profile={profile.name}, "
                    f"tags={len(extracted.tags)}"
                )

        except Exception as e:
            # Don't fail indexing if metadata extraction fails
            logger.warning(f"Failed to extract metadata for {path}: {e}")

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
