**Date:** January 18, 2026 (America/New_York)
**Author:** ChatGPT
**Model:** GPT-5.2 Thinking

# Requirements specification: Dataset + Resource abstractions (store module)

## 1) Problem statement

Today, data flows from `SourceCollection` → many `Document` rows where content is cached and indexed. You want:

* A durable representation of “things in the world” exactly as they exist in the source (URLs, files, Obsidian notes, etc.)
* A first-class orchestration layer that manages collection sync + caching before creating indexable Documents.

## 2) Goals

1. Introduce a **Resource** persistence layer that stores “clean source material” + retrieval metadata, independent of downstream indexing and document shaping.
2. Introduce a **Dataset** domain object (Python `@dataclass`) that is 1:1 with `SourceCollection`, and owns:

   * create/update of the `SourceCollection`
   * retrieval + caching of individual Resources
   * converting Resources → Documents
3. Move “DocumentCacher” responsibilities to the Resource layer so caching is based on source identity, not downstream document variants.

(Using a dataclass for the Dataset aligns with Python’s standard library `dataclasses` module. ([Python documentation][1]))

## 3) Non-goals (for this iteration)

* Replacing or redesigning `IndexingService` (Dataset will call it).
* Redefining how indexing works internally (chunking/embeddings/etc.).
* Multi-source datasets (keep 1:1 Dataset↔SourceCollection for now).
* Perfect deduplication across different SourceCollections (can be added later).

## 4) Concepts and invariants

### 4.1 Definitions

* **Resource**: the canonical cached representation of a single thing in the world (URI-addressable). Examples:

  * `https://...` web page snapshot / extracted content
  * `file://...` file bytes
  * `obsidian://...` note content + intrinsic metadata
* **Document**: an indexable unit produced *from Resources* (may be transformed, normalized, chunked, enriched). Documents are allowed to change as indexing strategy evolves.
* **Dataset**: the orchestrator that syncs a `SourceCollection` into Resources, then generates Documents from Resources.

### 4.2 Invariants

* Dataset is **1:1 with SourceCollection** (same identity boundary).
* Resource identity is **URI-based** within a dataset (at minimum unique on `(source_collection_id, uri)`; global uniqueness can be a later enhancement).
* Documents are derived artifacts; Resources are the “source of truth” cache.

## 5) Data model changes

### 5.1 New SQLAlchemy model: `Resource`

Create a new ORM-mapped table (SQLAlchemy declarative mapping). ([SQLAlchemy][2])

**Minimum required columns**

* `id` (PK)
* `source_collection_id` (FK → SourceCollection)
* `uri` (string, required)
* `resource_type` (enum/string): e.g. `url`, `file`, `obsidian`, `raw_text`, …
* `content_ref` (string/blob/json): pointer to cached payload (see “Storage strategy”)
* `content_hash` (string, nullable): hash of cached payload for change detection
* `source_modified_at` (datetime, nullable): “modified at” as reported by the source (if known)
* `source_created_at` (datetime, nullable): “created at” as reported by the source (if known)
* `loaded_at` (datetime, nullable): last time we successfully loaded
* `load_method` (string, nullable): how it was loaded (e.g., `http_get`, `filesystem`, `obsidian_api`)
* `load_status` (enum/string): `success`, `failed`, `skipped_not_modified`, …
* `load_error` (text, nullable): last error message/trace summary
* `index_state` (enum/string): `not_indexed`, `indexed`, `stale`, `failed`
* `indexed_at` (datetime, nullable)
* `index_method` (string, nullable): e.g. `indexing_service_v1`
* `index_error` (text, nullable)
* `metadata` (json, nullable): source-specific extras (headers, etag, size, mime, etc.)
* Timestamps: `created_at`, `updated_at`

**Constraints / indexes**

* Unique: `(source_collection_id, uri)`
* Index: `(source_collection_id, index_state)`, `(source_collection_id, loaded_at)`

### 5.2 Relationship to Documents

You will want an explicit lineage:

* Add `resource_id` FK on `Document` (or a join table if many-to-many is needed later).
* A Document MUST reference exactly one Resource in this iteration.

### 5.3 Storage strategy (requirement, not implementation)

Resource cached content should be stored either:

* in DB (bytea/blob/text) **only if small**, or
* in object storage / filesystem with `content_ref` pointing to the location.

Requirement: caching must be **idempotent** and **reproducible** given `(dataset, uri)`.

## 6) Dataset dataclass requirements

### 6.1 Dataset identity and construction

Dataset is a domain object that wraps a SourceCollection.

**Fields (suggested)**

* `source_collection_id` (or the `SourceCollection` instance)
* `store`/repositories handles (resource repo, document repo, sourcecollection repo)
* `retriever` (pluggable strategy to list items + fetch content)
* `indexing_service`

(Using dataclasses gives you a clean “domain object” that can be instantiated/tested without needing ORM entanglement. ([Python documentation][1]))

### 6.2 Core operations (public API)

1. `upsert_source_collection(config) -> SourceCollection`

   * Creates or updates the underlying SourceCollection config (name, connector params, filters).
2. `sync_resources(mode=...) -> SyncResult`

   * Lists current items from the source, upserts Resources, and caches content where needed.
   * Modes:

     * `full_refresh`: re-fetch all items
     * `incremental`: fetch only changed/new, using validators when available
3. `materialize_documents(strategy=...) -> MaterializeResult`

   * Creates/updates Documents derived from Resources (including doc variants rules if needed).
4. `index() -> IndexResult`

   * Calls `IndexingService` for Documents (or per-Resource batches), then updates `Resource.index_*` fields.

### 6.3 Sync semantics

For each listed source item (URI):

* Upsert `Resource` row (ensure exists).
* Decide whether to fetch content:

  * If source supports validators (HTTP ETag/Last-Modified), Dataset should store them in `Resource.metadata` and use conditional requests for incremental sync where possible. ([MDN Web Docs][3])
  * If a content hash is available, compare and skip caching if unchanged.
* Update `loaded_at`, `load_status`, `load_method`, `load_error` accordingly.
* If the content changed, mark `index_state = stale` (or `not_indexed`).

(Conditional requests are a well-established way to avoid unnecessary downloads using `ETag` and `Last-Modified`. ([MDN Web Docs][3]))

### 6.4 Failure handling requirements

* A failed load must not delete prior cached content (unless explicitly configured).
* Resource must record failure in `load_status/load_error`, and Dataset should continue syncing other items.
* Provide a summary result object with counts: created/updated/skipped/failed.

## 7) Caching responsibilities (move DocumentCacher → Resource)

### 7.1 New responsibility boundary

* **Resource caching** is the only place where raw content is fetched and stored.
* **Document creation** reads from Resource cached content and performs transformations necessary for indexing.

### 7.2 Required interfaces

Define a `ResourceCacher` (or refactor existing `DocumentCacher`) that supports:

* `fetch(uri, prior_metadata) -> (content, metadata, status)`
* `persist_content(resource, content) -> content_ref + content_hash`
* `load_content(resource) -> content`

HTTP retrievers should support conditional requests (If-None-Match / If-Modified-Since) when stored metadata exists. ([MDN Web Docs][3])

## 8) Indexing requirements

* Dataset `index()` may call `IndexingService` as a single operation after materialization.
* After successful indexing of a Resource’s derived Document(s), update:

  * `Resource.index_state = indexed`
  * `Resource.indexed_at = now`
  * `Resource.index_method = <identifier>`
* If indexing fails for a document, mark Resource `index_state = failed` and store `index_error`.

## 9) Transaction and consistency requirements

* Sync and materialize should be consistent per run:

  * Resource row + cached payload + derived Document updates should commit together per-resource, or in safe batches.
* If you use a Unit of Work approach, Dataset operations become easier to make atomic and testable (single “commit/rollback” boundary). ([cosmicpython.com][4])

## 10) Migration plan (practical requirements)

1. Create `resources` table and ORM model.
2. Add `resource_id` nullable FK to `documents`.
3. Backfill Resources:

   * For each existing Document, create a Resource representing its original source identity if recoverable (best-effort).
   * Set Document.resource_id.
4. Switch pipeline:

   * SourceCollection → Dataset.sync_resources() → Dataset.materialize_documents() → Dataset.index()
5. Deprecate or repurpose DocumentCacher into ResourceCacher.

## 11) Observability requirements

* Every run should emit structured logs/metrics:

  * number of listed items
  * resources created/updated/skipped
  * bytes fetched
  * indexing successes/failures
* Store “how and when”:

  * load_method + loaded_at
  * index_method + indexed_at
    This is explicitly required by your note: “needs to know when it was last loaded (and how) and indexed (and how).”

## 12) Acceptance criteria (testable)

1. **Resource persistence**: syncing a SourceCollection creates one Resource per discovered URI and updates `loaded_at/load_status`.
2. **Idempotency**: running `sync_resources(incremental)` twice with no source changes does not rewrite cached content and does not mark Resources stale.
3. **Document derivation**: `materialize_documents()` creates Documents referencing Resources via `resource_id`.
4. **Index lifecycle**: after `index()`, Resources have `index_state=indexed` and `indexed_at` set; changed Resources become `stale` until reindexed.
5. **Failure isolation**: one failing URI does not prevent other Resources from syncing and indexing; failure is recorded on the Resource.
6. **Backward compatibility**: existing indexing pipeline can still run during migration (feature flag or staged rollout).

---

If you want, I can also draft the concrete table schema (SQLAlchemy model skeleton + Alembic migration outline) and the Dataset method signatures you’ll expose from `store`, but the above is the requirements baseline you can hand to implementation.

[1]: https://docs.python.org/3/library/dataclasses.html?utm_source=chatgpt.com "dataclasses — Data Classes"
[2]: https://docs.sqlalchemy.org/en/latest/orm/declarative_mapping.html?utm_source=chatgpt.com "Mapping Classes with Declarative"
[3]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Conditional_requests?utm_source=chatgpt.com "HTTP conditional requests - MDN Web Docs"
[4]: https://www.cosmicpython.com/book/chapter_06_uow.html?utm_source=chatgpt.com "6. Unit of Work Pattern"
