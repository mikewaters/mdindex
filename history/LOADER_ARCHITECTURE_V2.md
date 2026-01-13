# Loader Service Architecture (v2)

## Goals

- Introduce a new `LoadingService` that abstracts retrieval and preparation of source data for persistence.
- Keep `IndexingService` focused on persistence and indexing responsibilities.
- Support both eager (materialized list) and streaming (lazy iterator) loading modes.
- Perform cleanup when a collection is reprocessed.
- Integrate with the `Application` composition root (not the deprecated `ServiceContainer`).

## Current Architecture (Summary)

- Loading and persistence are coupled in `IndexingService` (`src/pmd/services/indexing.py`).
- `DocumentSource` (in `src/pmd/sources/content/base.py`) provides listing and fetch, plus change detection via `check_modified`.
- Metadata extraction is performed in `IndexingService` via `pmd.metadata` profiles and optional collection configuration (`metadata_profile`).
- The `Application` class in `pmd.app` is the canonical composition root; `ServiceContainer` is deprecated.

## Proposed Architecture

### New Service: `LoadingService`

Location: `src/pmd/services/loading.py`

Responsibilities:
- Resolve collection and source (via `CollectionRepository` + `SourceRegistry`).
- Enumerate `DocumentReference` values from a `DocumentSource`.
- Perform change detection using `SourceMetadataRepository` and `DocumentSource.check_modified`.
- Fetch document content (`DocumentSource.fetch_content`).
- Extract title if missing.
- Extract metadata via profile registry (`pmd.metadata.get_default_profile_registry`), honoring `collection.source_config["metadata_profile"]`.
- Track enumerated paths for stale document detection.
- Emit a normalized `LoadedDocument` object.

Non-responsibilities:
- No database persistence of content, FTS, or metadata tables.
- No embeddings.

### Adjusted Service: `IndexingService`

Location: `src/pmd/services/indexing.py`

Responsibilities:
- Persist documents (`DocumentRepository.add_or_update`).
- Index or remove from FTS (`FTS5SearchRepository`).
- Persist source metadata (`SourceMetadataRepository`).
- Persist document metadata (`DocumentMetadataRepository`).
- Embed (unchanged).
- Mark stale documents inactive after loader completes (using enumerated paths from loader).

## Types

### `LoadedDocument`

Location: `src/pmd/services/loading.py` (or `src/pmd/core/types.py` if reused elsewhere)

Compose with existing `FetchResult` to avoid field duplication:

```python
from dataclasses import dataclass
from pmd.sources.content.base import DocumentReference, FetchResult
from pmd.metadata import ExtractedMetadata


@dataclass
class LoadedDocument:
    """A document that has been fetched and prepared for indexing.

    Attributes:
        ref: Original document reference from the source.
        fetch_result: Result from fetching content (content, metadata, etc.).
        title: Extracted or inferred title.
        fetch_duration_ms: Time taken to fetch the document.
        extracted_metadata: Metadata extracted via profiles (may override fetch_result.extracted_metadata).
    """
    ref: DocumentReference
    fetch_result: FetchResult
    title: str
    fetch_duration_ms: int
    extracted_metadata: ExtractedMetadata | None = None

    # Convenience accessors
    @property
    def content(self) -> str:
        return self.fetch_result.content

    @property
    def content_type(self) -> str:
        return self.fetch_result.content_type

    @property
    def path(self) -> str:
        return self.ref.path
```

### `LoadResult`

Returned by streaming loader to provide both documents and enumeration info:

```python
@dataclass
class LoadResult:
    """Result of a load operation.

    Attributes:
        documents: Iterator of loaded documents.
        enumerated_paths: Set of all paths that were enumerated (for stale detection).
        errors: List of (path, error_message) for documents that failed to load.
    """
    documents: AsyncIterator[LoadedDocument]
    enumerated_paths: set[str]
    errors: list[tuple[str, str]]
```

For eager mode, use a concrete variant:

```python
@dataclass
class EagerLoadResult:
    """Result of an eager load operation.

    Attributes:
        documents: List of all loaded documents.
        enumerated_paths: Set of all paths that were enumerated.
        errors: List of (path, error_message) for documents that failed to load.
    """
    documents: list[LoadedDocument]
    enumerated_paths: set[str]
    errors: list[tuple[str, str]]
```

## Loader Protocol

Location: `src/pmd/app/types.py`

```python
from typing import Protocol, AsyncIterator


class LoadingServiceProtocol(Protocol):
    """Protocol for document loading service."""

    async def load_collection_eager(
        self,
        collection_name: str,
        source: DocumentSource | None = None,
        force: bool = False,
    ) -> EagerLoadResult:
        """Load all documents from a collection (materialized).

        Args:
            collection_name: Name of the collection to load.
            source: Optional source override; resolved from collection if None.
            force: If True, reload all documents regardless of change detection.

        Returns:
            EagerLoadResult with all documents, enumerated paths, and errors.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            SourceListError: If the source cannot enumerate documents.
        """
        ...

    async def load_collection_stream(
        self,
        collection_name: str,
        source: DocumentSource | None = None,
        force: bool = False,
    ) -> LoadResult:
        """Load documents from a collection as a stream.

        Args:
            collection_name: Name of the collection to load.
            source: Optional source override; resolved from collection if None.
            force: If True, reload all documents regardless of change detection.

        Returns:
            LoadResult with async iterator, enumerated paths, and errors.
            Note: enumerated_paths is populated during enumeration, before
            documents are yielded. Errors accumulate as iteration proceeds.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            SourceListError: If the source cannot enumerate documents.
        """
        ...
```

## Loader API Design

### Eager Load

```python
async def load_collection_eager(
    self,
    collection_name: str,
    source: DocumentSource | None = None,
    force: bool = False,
) -> EagerLoadResult:
```

- Materializes all `LoadedDocument` instances before returning.
- Useful for tests and small sources.
- `enumerated_paths` and `errors` are complete when returned.

### Streaming Load

```python
async def load_collection_stream(
    self,
    collection_name: str,
    source: DocumentSource | None = None,
    force: bool = False,
) -> LoadResult:
```

- Yields `LoadedDocument` as soon as each is fetched and prepared.
- Suitable for large sources and CLI indexing.
- `enumerated_paths` is populated during the initial enumeration phase (before first yield).
- `errors` accumulates as iteration proceeds.

### Internal Loader Steps

1. Resolve collection by name.
2. Resolve `DocumentSource` if not provided (via `SourceRegistry`).
3. Enumerate all `DocumentReference` objects from `list_documents()` and populate `enumerated_paths`.
4. For each `DocumentReference`:
   - Look up existing document ID if present.
   - Pull stored metadata from `SourceMetadataRepository` if available.
   - If not forced, call `source.check_modified(ref, stored_metadata)`.
     - If `False`, skip (do not yield).
   - Fetch content via `source.fetch_content(ref)`.
   - If not forced and content hash matches existing document, skip.
   - Extract title (fallback to path stem).
   - Extract metadata via profiles if `FetchResult.extracted_metadata` is empty.
   - Yield or append `LoadedDocument`.
5. Capture any fetch errors in `errors` list (do not halt iteration).

### Error Handling Strategy

Loading errors are captured, not raised:

```python
# In loader implementation
try:
    fetch_result = await source.fetch_content(ref)
except SourceFetchError as e:
    self._errors.append((ref.path, str(e)))
    continue  # Skip this document, continue with next
```

This allows partial success - the indexer receives all loadable documents and a list of failures.

## Indexing Integration

### IndexingService Signature Changes

Updated to accept optional loader and remove required source parameter:

```python
class IndexingService:
    def __init__(
        self,
        db: DatabaseProtocol,
        collection_repo: CollectionRepositoryProtocol,
        document_repo: DocumentRepositoryProtocol,
        fts_repo: FTSRepositoryProtocol,
        embedding_repo: EmbeddingRepositoryProtocol,
        source_metadata_repo: SourceMetadataRepositoryProtocol,  # Injected, not created inline
        document_metadata_repo: DocumentMetadataRepositoryProtocol,  # Injected
        embedding_generator_factory: Callable[[], Awaitable[EmbeddingGeneratorProtocol]] | None = None,
        llm_available_check: Callable[[], Awaitable[bool]] | None = None,
        loader: LoadingServiceProtocol | None = None,
    ):
        ...
```

### Updated index_collection Signature

```python
async def index_collection(
    self,
    collection_name: str,
    source: DocumentSource | None = None,  # Now optional
    force: bool = False,
    embed: bool = False,
    mode: Literal["eager", "stream"] = "stream",
) -> IndexResult:
```

**Migration path for breaking change:**
- Phase 1: Make `source` optional with default `None`. If `None`, loader resolves it.
- Phase 2 (future): Deprecate `source` parameter entirely.

### Indexing Flow (Streaming Mode - Default)

```python
async def index_collection(self, collection_name: str, ...) -> IndexResult:
    # Get loader (injected or create default)
    loader = self._loader or LoadingService(...)

    # Load documents
    load_result = await loader.load_collection_stream(
        collection_name, source=source, force=force
    )

    indexed_count = 0
    skipped_count = 0

    # Process each loaded document
    async for doc in load_result.documents:
        result = await self._persist_document(doc)
        if result == "indexed":
            indexed_count += 1
        else:
            skipped_count += 1

    # Cleanup stale documents using enumerated_paths
    await self._cleanup_stale_documents(
        collection_name,
        load_result.enumerated_paths
    )

    # Combine loader errors with any persistence errors
    all_errors = load_result.errors + self._persist_errors

    if embed:
        await self.embed_collection(collection_name, force=force)

    return IndexResult(
        indexed=indexed_count,
        skipped=skipped_count,
        errors=all_errors,
    )
```

### _persist_document Method

New method focused purely on persistence (extracted from current `_index_document`):

```python
async def _persist_document(self, doc: LoadedDocument) -> str:
    """Persist a loaded document to storage.

    Args:
        doc: Document that has been loaded and prepared.

    Returns:
        "indexed" if persisted, "skipped" if content unchanged.
    """
    # Store document
    doc_result, is_new = self._document_repo.add_or_update(
        doc.collection_id,
        doc.path,
        doc.title,
        doc.content,
    )

    doc_id = doc_result.id

    # Index in FTS5
    if is_indexable(doc.content):
        self._fts_repo.index_document(doc_id, doc.path, doc.content)
    else:
        self._fts_repo.remove_from_index(doc_id)

    # Store source metadata
    self._source_metadata_repo.upsert(
        doc_id,
        etag=doc.fetch_result.metadata.get("etag"),
        last_modified=doc.fetch_result.metadata.get("last_modified"),
        extra=doc.fetch_result.metadata,
    )

    # Store document metadata
    if doc.extracted_metadata:
        self._document_metadata_repo.upsert(doc_id, doc.extracted_metadata)

    return "indexed"
```

### Cleanup of Stale Documents

```python
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
    collection = self._collection_repo.get_by_name(collection_name)
    all_docs = self._document_repo.list_by_collection(collection.id, active_only=True)

    stale_count = 0
    for doc in all_docs:
        if doc.path not in seen_paths:
            self._document_repo.mark_inactive(doc.id)
            self._fts_repo.remove_from_index(doc.id)
            stale_count += 1

    return stale_count
```

## Application Integration

Location: `src/pmd/app/__init__.py`

Update `create_application()` to wire the loader:

```python
async def create_application(config: Config) -> Application:
    # ... existing setup ...

    # Create repositories (inject, don't instantiate inline)
    source_metadata_repo = SourceMetadataRepository(db)
    document_metadata_repo = DocumentMetadataRepository(db)

    # Create loader
    loading = LoadingService(
        db=db,
        collection_repo=collection_repo,
        document_repo=document_repo,
        source_metadata_repo=source_metadata_repo,
        source_registry=get_default_registry(),
    )

    # Create indexing with loader
    indexing = IndexingService(
        db=db,
        collection_repo=collection_repo,
        document_repo=document_repo,
        fts_repo=fts_repo,
        embedding_repo=embedding_repo,
        source_metadata_repo=source_metadata_repo,
        document_metadata_repo=document_metadata_repo,
        embedding_generator_factory=get_embedding_generator,
        llm_available_check=is_llm_available,
        loader=loading,
    )

    # ... rest of setup ...
```

Optionally expose loader on Application:

```python
class Application:
    def __init__(
        self,
        db: Database,
        llm_provider: LLMProvider | None,
        indexing: IndexingService,
        search: SearchService,
        status: StatusService,
        loading: LoadingService,  # New
        config: Config,
    ):
        # ...
        self.loading = loading
```

## Services Module Export

Location: `src/pmd/services/__init__.py`

Add exports:

```python
from .loading import LoadingService, LoadedDocument, LoadResult, EagerLoadResult

__all__ = [
    # ... existing exports ...
    "LoadingService",
    "LoadedDocument",
    "LoadResult",
    "EagerLoadResult",
]
```

## CLI Integration

Location: `src/pmd/cli/commands/index.py`

Simplified call (source resolved internally):

```python
async def handle_index_collection(args, services):
    result = await services.indexing.index_collection(
        args.collection,
        force=args.force,
        embed=args.embed,
        mode="stream",
    )
    # ... display results ...
```

## MCP Integration

Location: `src/pmd/mcp/server.py`

Updated handler:

```python
@mcp.tool()
async def index_collection(
    collection_name: str,
    force: bool = False,
    embed: bool = False,
) -> dict:
    async with create_application(config) as app:
        result = await app.indexing.index_collection(
            collection_name,
            force=force,
            embed=embed,
        )
        return {
            "indexed": result.indexed,
            "skipped": result.skipped,
            "errors": result.errors,
        }
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Change detection semantics differ | Loader reuses exact same logic from current `_index_document`; no behavior change |
| API breaking change (`source` now optional) | Phased deprecation: optional first, then deprecated warning, then removed |
| Cleanup on every reprocess is costly | `cleanup_stale_documents` only marks inactive; heavy cleanup (orphan content/embeddings) remains separate |
| `enumerated_paths` memory for huge collections | For very large collections, consider streaming enumeration with bloom filter approximation |
| Error accumulation unbounded | Consider `max_errors` parameter to halt after N failures |

## Testing Strategy

### LoadingService Unit Tests

Location: `tests/unit/services/test_loading.py`

| Test Case | Description |
|-----------|-------------|
| test_load_eager_returns_all_documents | Eager mode returns complete list |
| test_load_stream_yields_documents | Stream mode yields as iterator |
| test_skip_unmodified_document | Respects `check_modified` returning False |
| test_skip_unchanged_content_hash | Skips when content hash matches |
| test_force_reloads_all | `force=True` ignores change detection |
| test_extracts_title_from_content | Title extraction fallback works |
| test_extracts_metadata_via_profile | Uses collection's `metadata_profile` |
| test_enumerated_paths_complete | All paths in `enumerated_paths` even if skipped |
| test_errors_captured_not_raised | Fetch errors added to errors list |
| test_resolves_source_from_collection | Source created from registry when None |

### IndexingService Integration Tests

Location: `tests/unit/services/test_indexing.py` (additions)

| Test Case | Description |
|-----------|-------------|
| test_index_with_loader_persists_documents | Documents from loader are persisted |
| test_cleanup_marks_stale_inactive | Documents not in enumerated_paths marked inactive |
| test_fts_removed_for_stale_documents | Stale documents removed from FTS |
| test_errors_combined_from_loader_and_persist | Both error sources in result |
| test_source_optional_uses_loader | `source=None` resolves via loader |

### Application Integration Tests

Location: `tests/integration/test_application.py`

| Test Case | Description |
|-----------|-------------|
| test_create_application_wires_loader | Loader accessible on app |
| test_full_index_flow_with_loader | End-to-end index via Application |

## Files to Change (Summary)

| File | Change |
|------|--------|
| `src/pmd/services/loading.py` | **Add**: New service |
| `src/pmd/services/indexing.py` | **Update**: Accept loader, optional source, stale cleanup |
| `src/pmd/services/__init__.py` | **Update**: Export LoadingService and types |
| `src/pmd/app/__init__.py` | **Update**: Wire LoadingService in create_application |
| `src/pmd/app/types.py` | **Update**: Add LoadingServiceProtocol |
| `src/pmd/cli/commands/index.py` | **Update**: Simplify source handling |
| `src/pmd/mcp/server.py` | **Update**: Simplify source handling |
| `tests/unit/services/test_loading.py` | **Add**: Loader unit tests |
| `tests/unit/services/test_indexing.py` | **Update**: Add loader integration tests |

## Migration Checklist

- [ ] Create `LoadingService` with eager and streaming modes
- [ ] Add `LoadingServiceProtocol` to `pmd.app.types`
- [ ] Update `IndexingService` to accept optional loader
- [ ] Make `source` parameter optional in `index_collection`
- [ ] Add `_cleanup_stale_documents` to `IndexingService`
- [ ] Wire loader in `create_application()`
- [ ] Update CLI to use simplified API
- [ ] Update MCP server to use simplified API
- [ ] Add unit tests for `LoadingService`
- [ ] Add integration tests for loader + indexer flow
- [ ] Update `ServiceContainer` for backwards compatibility (deprecated path)
