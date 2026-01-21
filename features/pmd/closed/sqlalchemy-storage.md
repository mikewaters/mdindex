# Feature: SQLAlchemy + Alembic storage layer

## Summary
Replace the current `sqlite3`-based `pmd.store.database.Database` and custom
`PRAGMA user_version` migrations with SQLAlchemy (sync + async) and Alembic,
while keeping the public Repository interfaces in `pmd.store` stable for
upstream callers.

This change also merges `source_metadata` into `documents` so the “document row”
is representable as a single ORM model.

## Goals
- Preserve the existing high-level repository APIs in `src/pmd/store/` so
  upstream clients remain unchanged (imports, method names, return types).
- Introduce SQLAlchemy as the only DB abstraction used by the storage layer
  (no direct `sqlite3` connection management outside SQLAlchemy).
- Support **both sync and async** DB access without mixing them:
  - Sync: CLI and any synchronous clients keep working.
  - Async: enable non-blocking DB access for async pipelines.
- Merge `source_metadata` into `documents` (single-row ownership).
- Add ORM models for:
  - `source_collections`
  - `documents`
  (and optionally `content` for relationships / joins)
- Switch migrations to Alembic, “starting fresh” from the current schema.
- Keep these tables/virtual tables semantically unchanged:
  - `content` table
  - `documents_fts` (FTS5 virtual table)
  - `content_vectors` + `content_vectors_vec` (sqlite-vec virtual table)

## Non-goals
- Changing indexing/search semantics, scoring, or result shaping.
- Migrating to non-SQLite backends (Postgres, etc.).
- Reworking the service layer contracts to be “async-first” in one pass (see
  rollout notes for an incremental approach).

## Current State (as of this change)
- Storage uses `sqlite3` directly via `pmd.store.database.Database`.
- Schema is managed via `pmd.store.migrations` using `PRAGMA user_version`.
- `source_metadata` is a separate table joined to `documents` by `document_id`.
- Direct SQL call sites outside `pmd.store` have already been removed by
  consolidating those queries into repositories (see
  `features/remove-direct-sql-usage.md`).

## Proposed Design

### 1) Keep `pmd.store` Repository surface stable
Keep these modules and class names:
- `pmd.store.database.Database` (sync)
- `pmd.store.collections.SourceCollectionRepository`
- `pmd.store.documents.DocumentRepository`
- `pmd.store.source_metadata.SourceMetadataRepository` (compat wrapper; table removed)

The implementation behind them changes to SQLAlchemy.

### 2) SQLAlchemy-first `Database` (no cursor compatibility requirement)
With direct SQL usage removed from the service layer, `pmd.store.database.Database`
can be a focused SQLAlchemy abstraction:

- Owns a single SQLAlchemy `Engine` (`sqlite+pysqlite`) and `Session` factory.
- Provides a `session()` / `begin()` context manager for repositories.
- Provides a `connection()` / `begin_connection()` escape hatch for the small
  set of store-internal operations that must use SQL (FTS/vec DDL, specialized
  queries that are simpler in Core).

This keeps all raw SQL contained within `pmd.store` while enabling ORM usage for
the “non-FTS/non-vector” tables.

### 3) Add async support without breaking sync callers
Add a parallel async DB abstraction (new public type, does not replace sync):
- `pmd.store.database.AsyncDatabase` (new)
  - wraps `AsyncEngine` (`sqlite+aiosqlite`)
  - exposes `async_session()` / `async_begin()` for async repositories
  - provides `async_connection()` for async Core usage (when needed)

This avoids mixing sync/async sessions and makes the call sites explicit.

Rollout option:
- Keep existing services using sync repos initially (no signature changes),
  and introduce async repos/services incrementally where the event loop
  blocking becomes unacceptable.

### 4) ORM models
Create a new module: `src/pmd/store/models.py`

Models (minimum):
- `SourceCollection` mapped to `source_collections`
- `Document` mapped to `documents` (includes merged source metadata columns)

Optional (recommended for joins / CAS):
- `Content` mapped to `content`

Relationships:
- `SourceCollection.documents` (1→many)
- `Document.source_collection` (many→1)
- `Document.content` (many→1 via `hash`)

We continue to treat FTS and vec tables as “non-ORM” (managed by raw SQL).

### 5) Schema change: merge `source_metadata` → `documents`
New columns added to `documents` (nullable unless otherwise required):
- `source_uri TEXT`
- `etag TEXT`
- `last_modified TEXT`
- `last_fetched_at TEXT`
- `fetch_duration_ms INTEGER`
- `http_status INTEGER`
- `content_type TEXT`
- `extra_metadata TEXT` (JSON-encoded string)

`source_metadata` table is removed.

`SourceMetadataRepository` becomes a compatibility wrapper:
- `upsert(metadata)` updates the corresponding `documents` row by `document_id`
- `get_by_document(document_id)` reads from `documents`
- `get_stale_documents(...)` queries `documents` directly

### 6) Alembic migrations (fresh history)
Add Alembic inside the package so it ships in the wheel:
- `src/pmd/store/alembic/`
  - `env.py` configured for programmatic use
  - `versions/` with revision modules

Migrations to include:
1. `0001_baseline` (single migration; new schema only)
   - Creates tables and indexes for the **new** schema (including the merged
     `documents` columns).
   - Creates `documents_fts` (FTS5) as part of the baseline.
   - Excludes `content_vectors_vec` creation (sqlite-vec requires per-connection
     extension load; keep runtime-managed by `Database.connect()`).

No “upgrade-from-legacy” or data-migration revisions are included. The expected
path for existing local databases is to delete/recreate them.

Runtime migration entrypoint:
- `pmd.store.migrate.upgrade(db_url)` called from `Database.connect()` to
  `upgrade head` before any queries.

### 7) sqlite-vec + pragmas
On connection (sync + async), ensure:
- `PRAGMA foreign_keys = ON`
- Best-effort sqlite-vec extension load (if `sqlite_vec` installed)
- Track `vec_available` on the `Database`/`AsyncDatabase` instance

`content_vectors_vec` creation remains “runtime-managed”:
- If vec is available, run an idempotent `CREATE VIRTUAL TABLE IF NOT EXISTS ...`
  with the configured embedding dimension.

## Implementation Plan
1. Add dependencies: `sqlalchemy`, `alembic`, and `aiosqlite` (plus any typing
   extras if needed).
2. Add `src/pmd/store/models.py` (Declarative Base + ORM models).
3. Implement SQLAlchemy-backed `pmd.store.database.Database`:
   - Engine + Session factory
   - connect-time pragmas + sqlite-vec loading (best-effort)
   - run Alembic upgrade on connect
4. Add `pmd.store.database.AsyncDatabase` (async engine + session factory) with
   the same responsibilities.
5. Add Alembic package scaffold under `src/pmd/store/alembic/` and implement:
   - `0001_baseline` (single baseline migration; no legacy upgrade path)
6. Update `SourceCollectionRepository` and `DocumentRepository` internals to
   use ORM Sessions (keeping public method signatures unchanged).
7. Replace `SourceMetadataRepository` internals to read/write merged columns
   on `documents`.
8. Remove the legacy `pmd.store.migrations` runner and its tests (or keep it
   temporarily behind a “bootstrap” path if needed during transition).
9. Update configuration to support SQLAlchemy URLs:
   - keep `Config.db_path`
   - add derived `db_url_sync` / `db_url_async` (or a single helper that builds
     both when needed)
10. Update tests:
   - add migration smoke test (new/empty DB → upgrade head)
   - add repository tests for merged source metadata behavior
   - remove tests tied to `PRAGMA user_version` migration runner
11. Ensure the remaining test suite passes; keep changes limited to the storage
    layer and the dependency-injection wiring that constructs repositories.

## Risks / Open Questions
- **Async sqlite-vec loading**: confirm `sqlite_vec.load(...)` works with
  SQLAlchemy’s `aiosqlite` driver connections; otherwise fall back to “vec
  disabled in async mode” while still supporting async for non-vec operations.
- **FTS/vec DDL timing**: decide whether FTS is created in migrations vs at
  runtime. sqlite-vec virtual table creation must remain runtime-managed.
- **SQLite DDL limitations**: column drops and constraint changes require batch
  operations; migrations should favor additive/idempotent changes.
- **Local DB reset UX**: without legacy migrations, users with an existing DB
  file need a documented “reset” flow (delete the DB file, or a CLI helper).

## Rollout
- Phase 1 (safe): sync SQLAlchemy `Database` + Alembic migrations + merged
  schema; repositories are the only consumers of DB access.
- Phase 2 (incremental): introduce async DB + async repositories where needed,
  without breaking sync CLI or synchronous integrations.
