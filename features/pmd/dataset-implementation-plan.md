# Feature Plan: Resource/Dataset layer aligned with current codebase

Date: 2026-01-19
Status: **IMPLEMENTED**
Last updated: 2026-01-19

## Implementation Summary

All core phases implemented:
- ✅ Phase 0: Prework - Facade strategy decided (new DatasetFacade)
- ✅ Phase 1a: Schema migration - v0003_resources_table.py with DDL
- ✅ Phase 1b: Backfill migration - filesystem and HTTP/entity collections
- ✅ Phase 2: Repository layer - ResourceRepository, DatasetFacade
- ✅ Phase 3: Caching rename - DocumentCacher → ResourceCacher
- ✅ Phase 4: Dataset orchestration - Dataset class with sync/materialize/index
- ⏸️ Phase 5: Service integration - Deferred (feature flag added, Dataset API primary)
- ⏸️ Phase 6: CLI & Application - Deferred (Python API available)
- ⏸️ Phase 7: Tests - Core tests complete, benchmarks deferred
- ⏸️ Phase 8: Documentation - Deferred (docstrings in code)

## Context recap (current code)
- Storage is SQLite via `pmd.store.database.Database`; migrations are raw SQL in `pmd/store/migrations/versions`.
- Persistence uses repositories (collections, documents, content, source_metadata, embeddings, fts) — no ORM.
- Ingestion flow: `LoadingService` → `IndexingService` → repositories via facades (`IndexFacade`, `LoadFacade`).
- Caching: optional `DocumentCacher` already at `pmd.store.caching` (storage layer).
- Source resolution: singleton registry (`get_default_registry`); no multi-registry support.
- Services access repositories through **facades** (e.g., `IndexFacade` wraps document, content, FTS, embedding repos).

## Conceptual model

```
SourceCollection (definition: where to fetch from)
    │
    │  sync_resources()
    ▼
Resource (fetched item with load/index state)
    │
    │  materialize_documents()
    ▼
Document (indexed content, searchable)
```

**Key distinction:**
- **SourceCollection**: Configuration for a data source (filesystem path, glob patterns, source type)
- **Resource**: A fetched item from a source with lifecycle state (loaded? indexed? failed?)
- **Document**: Searchable content derived from a Resource, stored in FTS and optionally embedded

Current model collapses Resource into Document. This plan separates them to enable:
- Tracking fetch state independently from index state
- Re-indexing without re-fetching
- Visibility into what's fetched vs. what's searchable

## Resource lifecycle (state machine)

```
load_status:
  ┌─────────┐    fetch()     ┌─────────┐
  │ pending │ ──────────────▶│ loading │
  └─────────┘                └────┬────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │ loaded  │   │ failed  │   │  stale  │
              └─────────┘   └─────────┘   └─────────┘
                                              ▲
                                              │ hash mismatch or
                                              │ source reports modified

index_state:
  ┌─────────┐   index()    ┌──────────┐
  │ pending │ ────────────▶│ indexing │
  └─────────┘              └────┬─────┘
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
            ┌─────────┐   ┌─────────┐   ┌─────────┐
            │ indexed │   │ failed  │   │  stale  │
            └─────────┘   └─────────┘   └─────────┘
                              ▲
                              │ resource content changed
```

**Stale triggers:**
- `load_status=stale`: Source reports modified (etag/last-modified changed) or periodic refresh
- `index_state=stale`: Resource hash changed after re-fetch, document needs re-indexing

## Objectives (reformulated)
1) Add a Resource layer that persists fetched source items and their fetch state, separate from derived Documents.
2) Introduce a Dataset orchestration object (1:1 with `SourceCollection`) that drives sync → materialize → index.
3) Rename `DocumentCacher` → `ResourceCacher` (already in storage layer, just rename/extend interface).
4) Keep the existing indexing/search behavior intact while we stage this in.

## Design deltas vs. PM proposal
- We will **not** introduce SQLAlchemy models; we stay with the existing raw-SQL migration + repository pattern.
- Resource content will reuse the existing `content` table when possible; we add a `resources` table to store fetch metadata and map to existing content hashes.
- Documents will reference a single Resource (`resource_id` FK) in this phase. Multi-document-per-resource (e.g., chunked loaders) is a future extension.
- Dataset object will live in `pmd.datasets` (new top-level module) to keep orchestration separate from pure data access in `pmd.store`.
- Single SourceRegistry (no DI of registry).
- `ResourceRepository` will be exposed via a new `DatasetFacade` or added to `IndexFacade`.

## Plan

### Phase 0 – Prework
- ~~Move `DocumentCacher` to `pmd.store`~~ (already done)
- Verify facade pattern is understood; decide on `DatasetFacade` vs extending `IndexFacade`

### Phase 1a – Schema migration (DDL only)
- Add new table `resources` with columns (raw SQL migration):
  - `id` INTEGER PRIMARY KEY
  - `source_collection_id` INTEGER NOT NULL REFERENCES source_collections(id)
  - `uri` TEXT NOT NULL — canonical identifier (e.g., `file:///abs/path` or `https://...`)
  - `resource_type` TEXT — mime type or classification
  - `hash` TEXT — content hash (matches `content.hash` when loaded)
  - `content_ref` TEXT — path/URI to cached payload (from ResourceCacher)
  - `source_created_at` TEXT, `source_modified_at` TEXT — from source metadata
  - `loaded_at` TEXT, `load_method` TEXT, `load_status` TEXT DEFAULT 'pending', `load_error` TEXT
  - `indexed_at` TEXT, `index_state` TEXT DEFAULT 'pending', `index_method` TEXT, `index_error` TEXT
  - `metadata` TEXT — JSON blob for extensible attributes
  - `created_at` TEXT, `updated_at` TEXT
  - UNIQUE(source_collection_id, uri)
  - INDEX(source_collection_id, index_state)
  - INDEX(source_collection_id, load_status)
- Add nullable `resource_id` INTEGER REFERENCES resources(id) to `documents` table.

### Phase 1b – Backfill migration (data only)
- For each active document with a filesystem collection:
  - Create Resource row: `uri = 'file://' || source_collections.pwd || '/' || documents.path`
  - Set `hash = documents.hash`, `load_status = 'loaded'`, `index_state = 'indexed'`
  - Set `loaded_at = indexed_at = documents.modified_at`
  - Update `documents.resource_id` to new resource ID
- For non-filesystem collections (HTTP, entity): best-effort using `source_metadata.source_uri` if available
- **Verification query**: `SELECT COUNT(*) FROM documents WHERE resource_id IS NULL AND active = 1` should be 0 or acceptably small
- **Rollback**: `UPDATE documents SET resource_id = NULL; DROP TABLE resources;`

### Phase 2 – Repository layer
- Create `ResourceRepository` in `pmd/store/repositories/resource.py`:
  ```python
  class ResourceRepository:
      def upsert(self, collection_id, uri, **attrs) -> Resource
      def get_by_uri(self, collection_id, uri) -> Resource | None
      def get_by_id(self, resource_id) -> Resource | None
      def list_by_collection(self, collection_id, *, status=None, state=None) -> list[Resource]
      def mark_loading(self, resource_id) -> None
      def mark_loaded(self, resource_id, hash, content_ref, metadata=None) -> None
      def mark_load_failed(self, resource_id, error) -> None
      def mark_indexing(self, resource_id) -> None
      def mark_indexed(self, resource_id, method=None) -> None
      def mark_index_failed(self, resource_id, error) -> None
      def mark_stale(self, resource_id, reason: Literal['load', 'index']) -> None
      def list_needing_index(self, collection_id) -> list[Resource]  # load_status=loaded, index_state in (pending, stale)
      def delete_orphaned(self, collection_id, valid_uris: set[str]) -> int
  ```
- Export from `pmd/store/repositories/__init__.py`
- Create `DatasetFacade` in `pmd/store/facades/dataset.py` wrapping:
  - `SourceCollectionRepository`, `ResourceRepository`, `DocumentRepository`, `ContentRepository`
- Or extend `IndexFacade` with resource methods (decision in Phase 0)

### Phase 3 – Caching rename
- Rename `DocumentCacher` → `ResourceCacher` in `pmd/store/caching.py`
- Update method names: `cache_document` → `cache_resource`, etc.
- Keep backward-compatible aliases if needed (deprecation warnings)
- Update all imports in services/tests

### Phase 4 – Dataset orchestration object
- Create `pmd/datasets/__init__.py` and `pmd/datasets/dataset.py`:
  ```python
  @dataclass
  class SyncResult:
      added: int
      updated: int
      unchanged: int
      failed: int
      errors: list[tuple[str, str]]  # (uri, error)

  @dataclass
  class MaterializeResult:
      created: int
      updated: int
      skipped: int

  @dataclass
  class DatasetIndexResult:
      indexed: int
      failed: int
      errors: list[tuple[str, str]]

  class Dataset:
      def __init__(self, collection: SourceCollection, facade: DatasetFacade,
                   cacher: ResourceCacher, source_registry=None):
          ...

      async def sync_resources(self, mode: Literal['full', 'incremental'] = 'incremental') -> SyncResult:
          """Fetch resources from source, update load state."""

      async def materialize_documents(self) -> MaterializeResult:
          """Create/update Documents from loaded Resources."""

      async def index(self) -> DatasetIndexResult:
          """Index documents (FTS + optional embeddings), update Resource.index_state."""

      async def refresh(self) -> tuple[SyncResult, MaterializeResult, DatasetIndexResult]:
          """Convenience: sync + materialize + index."""
  ```
- Uses `get_default_registry()` for source creation (no injection)
- Incremental sync logic:
  1. List resources with `load_status` in (pending, stale)
  2. For others, check source modification via `source.check_modified()` using stored etag/last-modified
  3. Fetch changed resources, update hash, compare to stored hash
  4. Mark unchanged resources as still `loaded`, changed as needing re-index

### Phase 5 – Service integration
- Update `IndexingService` to optionally use Resource tracking:
  - After persisting document, update `Resource.index_state`
  - Set `documents.resource_id` when known
- Update `LoadingService._load_document()` to optionally upsert Resource:
  - Before fetch: `mark_loading()`
  - After fetch: `mark_loaded()` with hash and content_ref
  - On error: `mark_load_failed()`
- Keep existing API surface stable; resource tracking is additive
- Feature flag: `Config.use_resource_tracking: bool = False` initially

### Phase 6 – CLI & Application wiring
- Add `Dataset` construction to `Application` (lazy, on-demand)
- Optional CLI commands:
  - `pmd dataset sync <collection>` — sync resources only
  - `pmd dataset status <collection>` — show resource states
- Existing `pmd index` continues to work via `IndexingService`

### Phase 7 – Tests
**Unit tests:**
- `ResourceRepository` CRUD and state transitions
- `DatasetFacade` integration
- `ResourceCacher` rename verification

**Integration tests:**
- Full flow: sync → materialize → index
- Incremental sync skips unchanged (mock source returning same etag)
- Re-sync after source modification updates Resource and triggers re-index
- Backfill migration on test data

**Performance tests:**
- Benchmark queries with resources JOIN on 10k+ documents
- Verify indexes are used (EXPLAIN QUERY PLAN)

**Migration tests:**
- Run backfill on copy of production-like data
- Verify rollback works cleanly

### Phase 8 – Documentation
- Update `store/README.md`: new Resource model, facade changes
- Update `services/README.md`: resource tracking integration
- Create `datasets/README.md`: Dataset API and usage
- Add migration notes to `docs/MAINTENANCE.md`
- Update architecture diagram if one exists

### Phase 9 – Feature flag / rollout (optional)
- Gate Dataset path behind `Config.use_resource_tracking`
- Run both paths in parallel during validation
- Telemetry: track sync/index success rates, timing
- Graduate to default after confidence threshold

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Migration backfill creates incorrect URIs | Best-effort mapping; allow null `resource_id`; verify query post-migration |
| Scope creep to multi-document-per-resource | Explicitly out of scope for v1; document as future work |
| Performance regression from joins | Add indexes in Phase 1a; benchmark in Phase 7 |
| Breaking existing IndexingService callers | Feature flag; keep API stable; resource tracking is additive |
| Orphaned resources accumulate | `delete_orphaned()` method; cleanup in sync when source no longer lists URI |

## Open decisions

1. **Facade strategy**: Create new `DatasetFacade` or extend `IndexFacade`?
   → **DECIDED**: Create new `DatasetFacade`. Rationale:
     - IndexFacade already wraps 5 repos; adding ResourceRepository would exceed healthy complexity
     - Resource lifecycle (sync/materialize) is a distinct concern from indexing
     - DatasetFacade wraps: SourceCollectionRepository, ResourceRepository, DocumentRepository, ContentRepository
     - IndexFacade remains focused on: Document indexing, FTS, embeddings
     - Composition: DatasetFacade.index() will delegate to IndexFacade for actual indexing

2. **Content storage**: Reuse `content` table or store in `content_ref` files?
   → Initial cut reuses `content` table; `content_ref` is optional cache path

3. **Multi-document-per-resource**: Support chunked loaders (1 Resource → N Documents)?
   → Deferred to v2; schema supports it (nullable `resource_id`)

4. **Orphan cleanup timing**: Delete orphaned resources immediately or mark stale first?
   → Mark stale with TTL, delete after grace period (configurable)

5. **Module location**: `pmd.datasets` vs `pmd.store.datasets` vs `pmd.workflows.datasets`?
   → `pmd.datasets` as top-level, parallel to `pmd.services`

## Appendix: Current table relationships (for reference)

```
source_collections
    │ 1:N
    ▼
documents ◄─────────────────┐
    │                       │
    ├─► content (via hash)  │ N:1 (dedup)
    ├─► source_metadata     │ 1:1
    ├─► document_metadata   │ 1:1
    ├─► document_tags       │ 1:N
    ├─► documents_fts       │ 1:1
    └─► content_vectors     │ 1:N (via hash)

After this plan:

source_collections
    │ 1:N
    ▼
resources (NEW)
    │ 1:1 (initially; 1:N later)
    ▼
documents
    │
    └─► (same relations as above)
```
