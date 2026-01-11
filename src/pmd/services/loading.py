"""Loading service for document retrieval and preparation.

This service abstracts retrieval and preparation of source data for persistence,
keeping IndexingService focused on persistence and indexing responsibilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from loguru import logger

from pmd.core.exceptions import SourceCollectionNotFoundError
from pmd.metadata import ExtractedMetadata, get_default_profile_registry
from pmd.sources.content.base import (
    DocumentReference,
    DocumentSource,
    FetchResult,
    SourceFetchError,
)

if TYPE_CHECKING:
    from pmd.app.types import (
        SourceCollectionRepositoryProtocol,
        DatabaseProtocol,
        DocumentRepositoryProtocol,
    )
    from pmd.core.types import SourceCollection
    from pmd.sources import SourceRegistry
    from pmd.store.source_metadata import SourceMetadataRepository
    from .loading_llamaindex import LlamaIndexLoaderAdapter


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class LoadedDocument:
    """A document that has been fetched and prepared for indexing.

    Attributes:
        ref: Original document reference from the source.
        fetch_result: Result from fetching content (content, metadata, etc.).
        title: Extracted or inferred title.
        fetch_duration_ms: Time taken to fetch the document.
        extracted_metadata: Metadata extracted via profiles (may override fetch_result.extracted_metadata).
        source_collection_id: ID of the source collection this document belongs to.
    """

    ref: DocumentReference
    fetch_result: FetchResult
    title: str
    fetch_duration_ms: int
    source_collection_id: int
    extracted_metadata: ExtractedMetadata | None = None

    # Convenience accessors
    @property
    def content(self) -> str:
        """Document content."""
        return self.fetch_result.content

    @property
    def content_type(self) -> str:
        """Content MIME type."""
        return self.fetch_result.content_type

    @property
    def path(self) -> str:
        """Document path (identity in the index)."""
        return self.ref.path


@dataclass
class LoadResult:
    """Result of a streaming load operation.

    Attributes:
        documents: Async iterator of loaded documents.
        enumerated_paths: Set of all paths that were enumerated (for stale detection).
            Note: This set is populated during enumeration, before documents are yielded.
        errors: List of (path, error_message) for documents that failed to load.
            Note: Errors accumulate as iteration proceeds.
    """

    documents: AsyncIterator[LoadedDocument]
    enumerated_paths: set[str] = field(default_factory=set)
    errors: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class EagerLoadResult:
    """Result of an eager load operation.

    Attributes:
        documents: List of all loaded documents.
        enumerated_paths: Set of all paths that were enumerated.
        errors: List of (path, error_message) for documents that failed to load.
    """

    documents: list[LoadedDocument]
    enumerated_paths: set[str] = field(default_factory=set)
    errors: list[tuple[str, str]] = field(default_factory=list)


# =============================================================================
# Loading Service
# =============================================================================


class LoadingService:
    """Service for loading documents from sources.

    Responsibilities:
    - Resolve source collection and source (via SourceCollectionRepository + SourceRegistry).
    - Enumerate DocumentReference values from a DocumentSource.
    - Perform change detection using SourceMetadataRepository and DocumentSource.check_modified.
    - Fetch document content (DocumentSource.fetch_content).
    - Extract title if missing.
    - Extract metadata via profile registry, honoring source_collection.source_config["metadata_profile"].
    - Track enumerated paths for stale document detection.
    - Emit normalized LoadedDocument objects.

    Non-responsibilities:
    - No database persistence of content, FTS, or metadata tables.
    - No embeddings.

    Example:
        loader = LoadingService(
            db=db,
            source_collection_repo=source_collection_repo,
            document_repo=document_repo,
            source_metadata_repo=source_metadata_repo,
            source_registry=source_registry,
        )
        result = await loader.load_collection_eager("my-docs")
        for doc in result.documents:
            print(doc.path, doc.title)
    """

    def __init__(
        self,
        db: "DatabaseProtocol",
        source_collection_repo: "SourceCollectionRepositoryProtocol",
        document_repo: "DocumentRepositoryProtocol",
        source_metadata_repo: "SourceMetadataRepository",
        source_registry: "SourceRegistry",
    ):
        """Initialize LoadingService.

        Args:
            db: Database for direct SQL operations.
            source_collection_repo: Repository for source collection operations.
            document_repo: Repository for document operations.
            source_metadata_repo: Repository for source metadata (change detection).
            source_registry: Registry for creating document sources.
        """
        self._db = db
        self._source_collection_repo = source_collection_repo
        self._document_repo = document_repo
        self._source_metadata_repo = source_metadata_repo
        self._source_registry = source_registry

    async def load_collection_eager(
        self,
        collection_name: str,
        source: DocumentSource | None = None,
        force: bool = False,
    ) -> EagerLoadResult:
        """Load all documents from a source collection (materialized).

        Args:
            collection_name: Name of the source collection to load.
            source: Optional source override; resolved from source collection if None.
            force: If True, reload all documents regardless of change detection.

        Returns:
            EagerLoadResult with all documents, enumerated paths, and errors.

        Raises:
            SourceCollectionNotFoundError: If source collection does not exist.
            SourceListError: If the source cannot enumerate documents.
        """
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(f"Source collection '{collection_name}' not found")

        # Resolve source if not provided
        if source is None:
            source = self._source_registry.create_source(source_collection)

        logger.debug(
            f"Loading source collection (eager): name={source_collection.name!r}, force={force}"
        )

        # Enumerate all document references
        enumerated_paths: set[str] = set()
        refs: list[DocumentReference] = []
        for ref in source.list_documents():
            enumerated_paths.add(ref.path)
            refs.append(ref)

        logger.debug(f"Enumerated {len(refs)} documents from source")

        # Load each document
        documents: list[LoadedDocument] = []
        errors: list[tuple[str, str]] = []

        for ref in refs:
            try:
                doc = await self._load_document(
                    source_collection=source_collection,
                    source=source,
                    ref=ref,
                    force=force,
                )
                if doc is not None:
                    documents.append(doc)
            except SourceFetchError as e:
                errors.append((ref.path, str(e)))
                logger.warning(f"Failed to fetch document: {ref.path}: {e}")
            except Exception as e:
                errors.append((ref.path, str(e)))
                logger.warning(f"Failed to load document: {ref.path}: {e}")

        logger.debug(
            f"Loaded {len(documents)} documents, {len(errors)} errors"
        )

        return EagerLoadResult(
            documents=documents,
            enumerated_paths=enumerated_paths,
            errors=errors,
        )

    async def load_collection_stream(
        self,
        collection_name: str,
        source: DocumentSource | None = None,
        force: bool = False,
    ) -> LoadResult:
        """Load documents from a source collection as a stream.

        Args:
            collection_name: Name of the source collection to load.
            source: Optional source override; resolved from source collection if None.
            force: If True, reload all documents regardless of change detection.

        Returns:
            LoadResult with async iterator, enumerated paths, and errors.
            Note: enumerated_paths is populated during enumeration, before
            documents are yielded. Errors accumulate as iteration proceeds.

        Raises:
            SourceCollectionNotFoundError: If source collection does not exist.
            SourceListError: If the source cannot enumerate documents.
        """
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(f"Source collection '{collection_name}' not found")

        # Resolve source if not provided
        if source is None:
            source = self._source_registry.create_source(source_collection)

        logger.debug(
            f"Loading source collection (stream): name={source_collection.name!r}, force={force}"
        )

        # Enumerate all document references first (for stale detection)
        enumerated_paths: set[str] = set()
        refs: list[DocumentReference] = []
        for ref in source.list_documents():
            enumerated_paths.add(ref.path)
            refs.append(ref)

        logger.debug(f"Enumerated {len(refs)} documents from source")

        # Create result with shared error list
        errors: list[tuple[str, str]] = []
        result = LoadResult(
            documents=self._stream_documents(
                source_collection=source_collection,
                source=source,
                refs=refs,
                force=force,
                errors=errors,
            ),
            enumerated_paths=enumerated_paths,
            errors=errors,
        )

        return result

    async def _stream_documents(
        self,
        source_collection: "SourceCollection",
        source: DocumentSource,
        refs: list[DocumentReference],
        force: bool,
        errors: list[tuple[str, str]],
    ) -> AsyncIterator[LoadedDocument]:
        """Stream documents from a list of references.

        Args:
            source_collection: The source collection being loaded.
            source: Document source to fetch from.
            refs: List of document references to load.
            force: If True, reload all documents regardless of change detection.
            errors: Shared list to accumulate errors.

        Yields:
            LoadedDocument for each successfully loaded document.
        """
        for ref in refs:
            try:
                doc = await self._load_document(
                    source_collection=source_collection,
                    source=source,
                    ref=ref,
                    force=force,
                )
                if doc is not None:
                    yield doc
            except SourceFetchError as e:
                errors.append((ref.path, str(e)))
                logger.warning(f"Failed to fetch document: {ref.path}: {e}")
            except Exception as e:
                errors.append((ref.path, str(e)))
                logger.warning(f"Failed to load document: {ref.path}: {e}")

    async def _load_document(
        self,
        source_collection: "SourceCollection",
        source: DocumentSource,
        ref: DocumentReference,
        force: bool,
    ) -> LoadedDocument | None:
        """Load a single document from a source.

        Args:
            source_collection: The source collection being loaded.
            source: Document source to fetch from.
            ref: Reference to the document.
            force: If True, reload even if unchanged.

        Returns:
            LoadedDocument if document should be indexed, None if skipped.
        """
        from pathlib import Path

        from pmd.utils.hashing import sha256_hash

        # Get existing document and metadata for change detection
        existing_doc = self._document_repo.get(source_collection.id, ref.path)
        doc_id = self._get_document_id(source_collection.id, ref.path) if existing_doc else None
        stored_metadata: dict[str, Any] = {}

        if doc_id and not force:
            meta = self._source_metadata_repo.get_by_document(doc_id)
            if meta:
                stored_metadata = meta.extra.copy()
                stored_metadata["etag"] = meta.etag
                stored_metadata["last_modified"] = meta.last_modified

            # Check if source says document is modified
            if not await source.check_modified(ref, stored_metadata):
                return None

        # Fetch content
        fetch_start = time.perf_counter()
        fetch_result = await source.fetch_content(ref)
        fetch_duration_ms = int((time.perf_counter() - fetch_start) * 1000)

        content = fetch_result.content

        # Check content hash if we have existing document
        if existing_doc and not force:
            content_hash = sha256_hash(content)
            if existing_doc.hash == content_hash:
                # Content unchanged - skip this document
                return None

        # Extract title
        title = ref.title or self._extract_title(content, Path(ref.path).stem)

        # Extract metadata
        extracted_metadata = fetch_result.extracted_metadata
        if extracted_metadata is None:
            extracted_metadata = self._extract_metadata_via_profiles(
                content,
                ref.path,
                source_collection,
            )

        return LoadedDocument(
            ref=ref,
            fetch_result=fetch_result,
            title=title,
            fetch_duration_ms=fetch_duration_ms,
            source_collection_id=source_collection.id,
            extracted_metadata=extracted_metadata,
        )

    def _get_document_id(self, source_collection_id: int, path: str) -> int | None:
        """Get document ID for a path.

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
        source_collection: "SourceCollection",
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
            # Don't fail loading if metadata extraction fails
            logger.warning(f"Failed to extract metadata for {path}: {exc}")
            return None

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

    async def load_from_llamaindex(
        self,
        collection_name: str,
        adapter: "LlamaIndexLoaderAdapter",
    ) -> EagerLoadResult:
        """Load documents from a LlamaIndex loader adapter.

        This method bridges LlamaIndex loaders with the LoadingService,
        allowing any LlamaIndex reader/loader to be used as a document source.

        Args:
            collection_name: Name of the source collection to load into.
            adapter: Configured LlamaIndex loader adapter.

        Returns:
            EagerLoadResult with loaded documents.

        Raises:
            SourceCollectionNotFoundError: If source collection does not exist.
            ValueError: If adapter validation fails (multiple docs, duplicate URIs).

        Example:
            from pmd.services.loading_llamaindex import LlamaIndexLoaderAdapter
            from llama_index.readers.web import SimpleWebPageReader

            reader = SimpleWebPageReader()
            adapter = LlamaIndexLoaderAdapter(
                loader=reader,
                content_type="text/html",
                load_kwargs={"urls": ["https://example.com"]},
            )

            result = await loading_service.load_from_llamaindex("web-docs", adapter)
        """
        source_collection = self._source_collection_repo.get_by_name(collection_name)
        if not source_collection:
            raise SourceCollectionNotFoundError(f"Source collection '{collection_name}' not found")

        logger.debug(
            f"Loading from LlamaIndex adapter: source_collection={source_collection.name!r}, "
            f"namespace={adapter.namespace!r}"
        )

        # Load via adapter (does not have source collection context)
        loaded_docs = await adapter.load_eager()

        # Inject source_collection_id into each document
        for doc in loaded_docs:
            doc.source_collection_id = source_collection.id

        logger.debug(f"Loaded {len(loaded_docs)} documents from LlamaIndex adapter")

        return EagerLoadResult(
            documents=loaded_docs,
            enumerated_paths={doc.path for doc in loaded_docs},
            errors=[],
        )
