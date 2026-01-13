# Loader Service Architecture (Option A)

## Goals
- Introduce a new Loader service that abstracts retrieval and preparation of source data for persistence.
- Keep IndexingService focused on persistence and indexing responsibilities.
- Support both eager (materialized list) and greedy (streaming) loading modes.
- Perform cleanup when a collection or document is reprocessed.

## Current Architecture (Summary)
- Loading and persistence are coupled in `IndexingService` (`src/pmd/services/indexing.py`).
- `DocumentSource` (in `src/pmd/sources/content/base.py`) provides listing and fetch, plus change detection via `check_modified`.
- Metadata extraction is performed in IndexingService via `pmd.metadata` profiles and optional collection configuration (`metadata_profile`).

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
- Collection cleanup during reprocess.

## Types

### `LoadedDocument`
Add a small dataclass either in `src/pmd/core/types.py` or in `src/pmd/services/loading.py` (and re-export if needed):

Proposed fields:
- `ref: DocumentReference`
- `content: str`
- `title: str`
- `content_type: str`
- `encoding: str`
- `extracted_metadata: ExtractedMetadata | None`
- `fetch_metadata: dict[str, Any]`
- `fetch_duration_ms: int`

Rationale:
- IndexingService can persist content and metadata without needing to re-query the source.
- `fetch_metadata` captures etag/last_modified/http_status for source metadata persistence.

## Loader API Design

### Eager load
```
async def load_collection_eager(
    self,
    collection_name: str,
    source: DocumentSource | None = None,
    force: bool = False,
) -> list[LoadedDocument]:
```
- Materializes all `LoadedDocument` instances before returning.
- Useful for tests and small sources.

### Greedy load (streaming)
```
async def load_collection_stream(
    self,
    collection_name: str,
    source: DocumentSource | None = None,
    force: bool = False,
) -> AsyncIterator[LoadedDocument]:
```
- Yields `LoadedDocument` as soon as it is fetched and prepared.
- Suitable for large sources and current CLI index behavior.

### Internal loader steps
1) Resolve collection by name.
2) Resolve `DocumentSource` if not provided (via `SourceRegistry`).
3) For each `DocumentReference` from `list_documents()`:
   - Look up existing document ID if present.
   - Pull stored metadata from `SourceMetadataRepository` if available.
   - If not forced, call `source.check_modified(ref, stored_metadata)`.
     - If `False`, skip.
   - Fetch content via `source.fetch_content(ref)`.
   - If not forced and content hash matches existing, skip (but still update source metadata if desired).
   - Extract title (fallback to path stem).
   - Extract metadata via profiles if `FetchResult.extracted_metadata` is empty.
   - Yield or append `LoadedDocument`.

## Indexing Integration

### IndexingService signature changes
Recommended adjustments to reflect loader integration:

```
class IndexingService:
    def __init__(
        self,
        db: DatabaseProtocol | None = None,
        collection_repo: CollectionRepositoryProtocol | None = None,
        document_repo: DocumentRepositoryProtocol | None = None,
        fts_repo: FTSRepositoryProtocol | None = None,
        embedding_repo: EmbeddingRepositoryProtocol | None = None,
        embedding_generator_factory: Callable[[], Awaitable[EmbeddingGeneratorProtocol]] | None = None,
        llm_available_check: Callable[[], Awaitable[bool]] | None = None,
        source_registry: SourceRegistry | None = None,
        loader: LoadingService | None = None,
        container: ServiceContainer | None = None,
    )
```

### Indexing flow (greedy mode default)
- `IndexingService.index_collection()` obtains loader (injected or created).
- Calls `load_collection_stream` and persists each `LoadedDocument`.
- Tracks `seen_paths` for cleanup.

### Cleanup on reprocess (collection)
After consuming the loader stream for a collection:
1) Query all active documents in the collection.
2) Any document path not in `seen_paths` is considered stale and should be marked inactive (`active = 0`).
3) Call `SourceMetadataRepository.remove_orphans()` and `DocumentMetadataRepository.remove_orphans()` to clean metadata for deactivated docs.
4) Optionally reuse the existing `cleanup_orphans()` to remove content/embeddings not referenced by any active doc (may be gated by a flag to avoid heavy work each run).

### Cleanup on reprocess (document)
For each `LoadedDocument`:
- If content hash changes, it will update the document row.
- FTS index updated or removed via `is_indexable` as today.
- Source metadata and document metadata upserted.

## ServiceContainer integration
Location: `src/pmd/services/container.py`

Add:
- Lazy init property `self._loading: LoadingService | None`.
- Accessor `loading` to construct loader with shared repos and registry.

Example:
```
@property
def loading(self) -> LoadingService:
    if self._loading is None:
        self._loading = LoadingService(
            db=self.db,
            collection_repo=self.collection_repo,
            document_repo=self.document_repo,
            source_registry=get_default_registry(),
        )
    return self._loading
```

IndexingService created by the container should receive the loader.

## Services module export
Location: `src/pmd/services/__init__.py`

Add exports:
- `LoadingService`

## CLI integration
Location: `src/pmd/cli/commands/index.py`

- `handle_index_collection` should call the updated `IndexingService.index_collection` without manually instantiating sources.
- The loader will resolve the source from the collection internally.

Proposed call:
```
result = await services.indexing.index_collection(
    args.collection,
    force=args.force,
    embed=args.embed,
    mode="greedy",
)
```

## MCP integration
Location: `src/pmd/mcp/server.py`

- `index_collection` handler should call the updated IndexingService with loader‑backed flow and avoid manual source creation.

## Risks and Mitigations
- Risk: change detection and skipping now move to loader; must preserve semantics (etag/last-modified + content hash fallback).
  - Mitigation: reuse existing logic and avoid double-check.
- Risk: loader mode ambiguity in APIs.
  - Mitigation: use explicit method names for eager vs greedy; or a `mode` argument with two explicit code paths.
- Risk: cleanup on every reprocess may be costly for large collections.
  - Mitigation: allow optional flag to skip cleanup or gate heavy cleanup (content/embeddings) separately.

## Testing Strategy (planned)
- Loader unit tests for:
  - eager vs greedy behavior
  - skip behavior when `check_modified` returns false
  - metadata extraction path using collection `metadata_profile`
- Indexing integration tests for:
  - cleanup of removed documents on reprocess
  - FTS removal for non-indexable documents
  - metadata persistence when loader provides extracted metadata

## Files to change (summary)
- Add: `src/pmd/services/loading.py`
- Update: `src/pmd/services/indexing.py`
- Update: `src/pmd/services/container.py`
- Update: `src/pmd/services/__init__.py`
- Update: `src/pmd/cli/commands/index.py`
- Update: `src/pmd/mcp/server.py`
- Optional: `src/pmd/core/types.py` (if placing `LoadedDocument` there)
- Optional: `src/pmd/app/types.py` (if adding protocol for loader)
