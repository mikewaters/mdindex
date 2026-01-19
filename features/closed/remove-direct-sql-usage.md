# Feature: Remove direct SQL usage outside `pmd.store`

## Summary
Several modules outside `src/pmd/store/` execute raw SQL directly via
`Database.execute()` / `Database.transaction()`. This feature inventories those
call sites, categorizes the queries they emit, and proposes minimal additions
to the existing Repository layer to remove those direct SQL joins.

This is intended to support the SQLAlchemy transition by:
- shrinking the “cursor compatibility” surface area, and
- consolidating document/FTS/vector/metadata join logic into repositories.

## Goals
- Enumerate all direct SQL call sites outside `pmd.store`.
- Identify the query patterns (counts, joins, IN-list lookups, cleanup deletes).
- Propose repository methods (or a small new repository) that can replace them.

## Non-goals
- Implementing the changes in this document (this is a design/plan only).
- Changing the semantics of queries (counts, filters, “active” logic, etc.).

## Inventory of call sites and query types

### `src/pmd/services/status.py`
Direct SQL is used for status/diagnostics aggregates and “sync health” joins.

- `src/pmd/services/status.py:74` — `SELECT COUNT(*) FROM documents WHERE active = 1`
  - Type: aggregate count (single table)
  - Proposed repository change:
    - `DocumentRepository.count_active(source_collection_id: int | None = None) -> int`
- `src/pmd/services/status.py:82` — `SELECT COUNT(DISTINCT hash) FROM content_vectors`
  - Type: aggregate count (single table, distinct)
  - Proposed repository change:
    - `EmbeddingRepository.count_distinct_embedded_hashes() -> int`
- `src/pmd/services/status.py:98` — `SELECT COUNT(*) FROM content_vectors`
  - Type: aggregate count (single table)
  - Proposed repository change:
    - `EmbeddingRepository.count_embeddings() -> int`
- `src/pmd/services/status.py:165` — count active docs by collection id
  - Type: aggregate count (filtered)
  - Proposed repository change:
    - `DocumentRepository.count_active(source_collection_id: int | None = None) -> int`
- `src/pmd/services/status.py:174` — join `documents` → `content_vectors` to count embedded docs per collection
  - Type: join + aggregate count
  - Proposed repository change:
    - `DocumentRepository.count_active_with_embeddings(source_collection_id: int | None = None) -> int`
      - (implemented via join to `content_vectors` on `hash`)
- `src/pmd/services/status.py:223`/`:234` — left join `documents` → `documents_fts` to find missing FTS rows + sample paths
  - Type: left join + anti-join (“missing”) + sample list
  - Proposed repository change (choose one):
    - Add to `FTS5SearchRepository`:
      - `count_documents_missing_fts(source_collection_id: int | None = None) -> int`
      - `list_paths_missing_fts(source_collection_id: int | None = None, limit: int = 20) -> list[str]`
    - OR introduce a small `IndexDiagnosticsRepository` (see below).
- `src/pmd/services/status.py:248`/`:259` — left join `documents` → `content_vectors` to find missing embeddings + sample paths
  - Type: left join + anti-join (“missing”) + sample list
  - Proposed repository change:
    - `EmbeddingRepository.count_documents_missing_embeddings(source_collection_id: int | None = None) -> int`
    - `EmbeddingRepository.list_paths_missing_embeddings(source_collection_id: int | None = None, limit: int = 20) -> list[str]`
- `src/pmd/services/status.py:273` — orphan embeddings: `content_vectors` hashes with no active docs
  - Type: left join + anti-join (“orphaned”) + aggregate count
  - Proposed repository change:
    - `EmbeddingRepository.count_orphaned_embeddings() -> int`
- `src/pmd/services/status.py:283` — orphan FTS rows: `documents_fts` rows with no active docs
  - Type: left join + anti-join (“orphaned”) + aggregate count
  - Proposed repository change:
    - `FTS5SearchRepository.count_orphaned_fts_rows() -> int`

### `src/pmd/services/indexing.py`
Direct SQL is used for cleanup operations, ID lookups, and the metadata backfill join.

- `src/pmd/services/indexing.py:350`/`:359` — count + delete orphaned content not referenced by active documents
  - Type: anti-join + aggregate count + delete
  - Proposed repository change:
    - Introduce `ContentRepository`:
      - `count_orphaned_content(active_docs_only: bool = True) -> int`
      - `delete_orphaned_content(active_docs_only: bool = True) -> int`
        - Returns deleted rows/hashes count.
- `src/pmd/services/indexing.py:367`/`:376` — count + delete orphaned `content_vectors` not referenced by active documents
  - Type: anti-join + aggregate count + delete
  - Proposed repository change:
    - `EmbeddingRepository.count_orphaned_embeddings(active_docs_only: bool = True) -> int`
    - `EmbeddingRepository.delete_orphaned_embeddings(active_docs_only: bool = True) -> int`
- `src/pmd/services/indexing.py:404` — `_get_document_id`: lookup document id by `(source_collection_id, path)`
  - Type: point lookup (index-backed)
  - Proposed repository change:
    - `DocumentRepository.get_id(source_collection_id: int, path: str, active_only: bool = False) -> int | None`
      - Keeps `DocumentResult` unchanged while still exposing the internal row id.
- `src/pmd/services/indexing.py:533`–`:576` — metadata backfill selects:
  - joins `documents` → `source_collections` → `content`
  - optional left join to `document_metadata` for “only missing” mode
  - Type: multi-join “report/query” returning denormalized rows
  - Proposed repository change (choose one):
    - Add to `DocumentRepository` (preferred for ownership of doc/content join):
      - `list_for_metadata_backfill(collection_name: str | None, only_missing: bool) -> list[BackfillRow]`
        - where `BackfillRow` includes: `document_id`, `path`, `body`, `collection_name`, `source_config`.
    - OR introduce `MetadataBackfillRepository` under `pmd.store` dedicated to this query.

### `src/pmd/services/loading.py`
- `src/pmd/services/loading.py:419` — `_get_document_id` duplicate point lookup by `(source_collection_id, path)`
  - Type: point lookup
  - Proposed repository change:
    - reuse `DocumentRepository.get_id(...)` above.

### `src/pmd/workflows/pipelines/embedding.py`
- `src/pmd/workflows/pipelines/embedding.py:211` — list embed targets:
  - join `documents` → `content` selecting `path`, `hash`, `doc`
  - Type: join + bulk select
  - Proposed repository change:
    - `DocumentRepository.list_active_with_content(source_collection_id: int) -> list[tuple[path, hash, body]]`
      - Or a dedicated return type `EmbedTargetRow`.

### `src/pmd/workflows/pipelines/ingestion.py`
- `src/pmd/workflows/pipelines/ingestion.py:318` — `_get_document_id` duplicate point lookup by `(source_collection_id, path)`
  - Type: point lookup
  - Proposed repository change:
    - reuse `DocumentRepository.get_id(...)`.

### `src/pmd/search/adapters/boost.py`
- `src/pmd/search/adapters/boost.py:164` — map paths → document IDs:
  - `SELECT id, path FROM documents WHERE path IN (...) AND active = 1`
  - Type: IN-list lookup
  - Proposed repository change:
    - `DocumentRepository.get_ids_by_paths(paths: list[str], active_only: bool = True) -> dict[str, int]`
      - Optional extension: accept `source_collection_id` to disambiguate if paths are not globally unique.

### `src/pmd/metadata/*` and `src/pmd/search/metadata/*`
These modules form the metadata/tag storage + retrieval subsystem and currently
contain the deepest “document ↔ content” join outside `pmd.store`.

- `src/pmd/metadata/store/repository.py:42` — metadata upsert + junction table maintenance
  - Type: transactional writes (`document_metadata` + `document_tags`)
  - Proposed repository change:
    - No new `pmd.store` repo needed; this repository should be treated as part
      of the storage layer (it already depends on `Database`).
    - When moving to SQLAlchemy, this module should either:
      - become an ORM repository (preferred), or
      - be relocated under `pmd.store` so “storage layer only” remains true.
- `src/pmd/metadata/query/retrieval.py:186` (and the mirrored `src/pmd/search/metadata/retrieval.py:186`)
  - Query fetches full document rows + content body by `document_id IN (...)` and optional collection filter.
  - Type: join + IN-list bulk select
  - Proposed repository change:
    - `DocumentRepository.get_search_rows_by_ids(document_ids: list[int], source_collection_id: int | None = None) -> list[DocumentSearchRow]`
      - `DocumentSearchRow` includes fields needed for `SearchResult` construction: `path`, `title`, `hash`, `source_collection_id`, `modified_at`, `body`.
- `src/pmd/metadata/query/scoring.py:329` (and the mirrored `src/pmd/search/metadata/scoring.py:329`)
  - `build_path_to_id_map`: path → id lookup via IN-list
  - Type: IN-list lookup
  - Proposed repository change:
    - reuse `DocumentRepository.get_ids_by_paths(...)` above.

### `src/pmd/cli/commands/collection.py`
- `src/pmd/cli/commands/collection.py:202` — count active docs by collection id
  - Type: aggregate count
  - Proposed repository change:
    - reuse `DocumentRepository.count_active(source_collection_id=...)`.
- `src/pmd/cli/commands/collection.py:209` — count embedded docs by collection
  - Type: join + aggregate count
  - Proposed repository change:
    - reuse `DocumentRepository.count_active_with_embeddings(source_collection_id=...)`.

## Proposed repository API additions (consolidated)

### `DocumentRepository` additions
- `count_active(source_collection_id: int | None = None) -> int`
- `count_active_with_embeddings(source_collection_id: int | None = None) -> int`
- `get_id(source_collection_id: int, path: str, active_only: bool = False) -> int | None`
- `get_ids_by_paths(paths: list[str], active_only: bool = True, source_collection_id: int | None = None) -> dict[str, int]`
- `list_active_with_content(source_collection_id: int) -> list[tuple[str, str, str]]`
  - returns `(path, hash, body)` for embedding pipeline
- `get_search_rows_by_ids(document_ids: list[int], source_collection_id: int | None = None) -> list[DocumentSearchRow]`
- `list_for_metadata_backfill(collection_name: str | None, only_missing: bool) -> list[BackfillRow]`

### `FTS5SearchRepository` additions (diagnostics)
- `count_documents_missing_fts(source_collection_id: int | None = None) -> int`
- `list_paths_missing_fts(source_collection_id: int | None = None, limit: int = 20) -> list[str]`
- `count_orphaned_fts_rows() -> int`

### `EmbeddingRepository` additions (diagnostics/cleanup)
- `count_embeddings() -> int`
- `count_distinct_embedded_hashes() -> int`
- `count_documents_missing_embeddings(source_collection_id: int | None = None) -> int`
- `list_paths_missing_embeddings(source_collection_id: int | None = None, limit: int = 20) -> list[str]`
- `count_orphaned_embeddings(active_docs_only: bool = True) -> int`
- `delete_orphaned_embeddings(active_docs_only: bool = True) -> int`

### New `ContentRepository` (optional, but keeps ownership clear)
- `count_orphaned_content(active_docs_only: bool = True) -> int`
- `delete_orphaned_content(active_docs_only: bool = True) -> int`

## Alternative: one dedicated diagnostics repository
If the above additions feel too “reporting-heavy” for the core repos, add:
- `IndexDiagnosticsRepository`
  - owns all “missing/orphan” and “counts across joins” queries
  - used by `StatusService` and CLI “show collection”
  - keeps `DocumentRepository`/`EmbeddingRepository` focused on CRUD and search

This still satisfies “repository layer” consolidation while avoiding bloated
CRUD repositories.

## Mechanical refactors to remove direct SQL
- `StatusService`:
  - replace all `self._db.execute(...)` with repository calls
- `IndexingService.cleanup_orphans()`:
  - delegate to `ContentRepository` and `EmbeddingRepository` cleanup methods
- `IndexingService._get_document_id`, `LoadingService._get_document_id`, `IngestionPipeline._get_document_id`:
  - replace with `DocumentRepository.get_id(...)`
- `EmbeddingPipeline._list_embed_targets()`:
  - replace join with `DocumentRepository.list_active_with_content(...)`
- `TagRetriever` and metadata scoring helpers:
  - replace document-details join with `DocumentRepository.get_search_rows_by_ids(...)`
  - replace path→id map helper with `DocumentRepository.get_ids_by_paths(...)`
