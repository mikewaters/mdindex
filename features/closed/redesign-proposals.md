# RAG Proposal


Perform a ground-up rewrite of `pmd` using LlamaIndex abstractions over custom code. We will create a new module named `idx`.
We will take some design decisions from the `pmd` project where needed.

## Design decisions inherited from `pmd` (must replicate)
Before we finalize contracts for refresh, chunking, FTS, vector storage, and hybrid retrieval, we will preserve the current `pmd` semantics:

### Refresh (differential indexing)
- Enumerate all document references from the source each run (used for stale detection).
- For each document, skip loading/indexing when unchanged, using **both**:
  - Source-provided metadata (e.g., `etag`, `last_modified`) via a `check_modified()` call when available, and
  - Content hash comparison (SHA256) after fetch as a safety net.
- Treat missing documents as *stale* and **soft-delete** them (mark inactive) and remove them from the FTS index; a “rename” behaves as delete+add (no special rename detection).
- Support a `force` mode that bypasses change detection and reloads everything.
- Provide explicit orphan cleanup for derived indexes (FTS + vector nodes) since documents are soft-deleted.

### Chunking (for embeddings)
- Use deterministic, line-oriented chunking (no overlap) with a byte budget:
  - Accumulate lines until `max_bytes` would be exceeded, then cut a chunk.
  - Avoid tiny fragments by merging chunks smaller than `min_chunk_size`.
  - Track `pos` as the running offset produced by the chunker (used for provenance/snippetting).

### Vector storage + scoring
We will preserve `pmd`’s *chunk-level* semantics (seq/pos, best-chunk-per-document dedupe), but **we will not use `sqlite-vec`**.

- Persist vectors using LlamaIndex’s `VectorStoreIndex` with `SimpleVectorStore` (persisted via LlamaIndex `StorageContext` / `persist_dir`) to a location configured in `idx.core.settings`.
- Use stable per-chunk node IDs: `node_id = "{hash}:{seq}"`, where `hash` is the document content SHA256 and `seq` is the deterministic chunk sequence number.
  - Trade-off (simplicity): any document content change implies deleting and recreating *all* chunk nodes for that document (we do not attempt partial reuse of chunks).
- Store `hash`, `seq`, `pos`, `path`, and `dataset_name` in node metadata so:
  - vector results can be deduped to a document, and
  - refresh can delete+replace deterministically.
- Dataset scoping for vector search is performed client-side (simplicity): retrieve candidates, then filter by `dataset_name` in metadata.
- Prefer rank-based fusion (RRF) in hybrid search; do not assume any specific numeric similarity distribution from the vector store.
- Deduplicate by document path at query time (keep the best-scoring chunk per document).

### Full-text search (FTS5)
- Use a dedicated `documents_fts` FTS5 virtual table (tokenizer: `porter unicode61`) with `(path, body)`.
- Use document `id` as the FTS5 `rowid` to support update/delete by `rowid`.
- Queries are passed through as FTS5 syntax (minimal query rewriting).
- Normalize the raw BM25/rank signal to a comparable range per result set (current behavior uses an abs+max-normalization scheme).

### Hybrid search (RRF) + LLM-as-judge reranking
- Hybrid combines multiple ranked lists via Reciprocal Rank Fusion:
  - `rrf = Σ(weight / (k + rank + 1))` with small bonuses for the top ranks.
- Reranking (LLM-as-judge) is applied to the top `rerank_candidates` after fusion.
- Final scoring blends RRF score with reranker score using **position-aware** weights:
  - Ranks 1–3: 75% RRF / 25% reranker
  - Ranks 4–10: 60% RRF / 40% reranker
  - Ranks 11+: 40% RRF / 60% reranker
- Optionally normalize final scores to 0–1 (max-normalization).

## Problem Statement
We have multiple repositories containing text and multi-modal content, such as Obsidian vaults, email accounts, bookmarks managers, and knowledge management tools.
We want to provide end users with the ability to query their content across these disparate systems, and to do so we will provide a way to index them all in a single workload.  
We also want to associate this content with a user's ontology, which represents their interests and concerns in a structured way, and we can use metadata provided by them in the source material (like tags) or perform our own content classification to augment that.

### Initial narrow scope
We are taking a small slice of this problem and solving that.

- We provide a naive search to start (fulltext, vector, and hybrid), and will include more advanced methods like GraphRAG in the future. 
- We will support markdown-based source repositories (like Obsidian) for now, and skip multi-modal data for indexing
- We will use a stub ontology for now, and perform no content classification nor link to knowledge graphs; we'll use only the metadata present in the source
- We will use local ML models and a local database for now to reduce complexity (object-store caching is deferred for the MVP)

## Goals
- Implement the Design and Features below, bounded by the Architecture choices.

## Non-goals
- Creating a CLI or API frontend. Callers will use a python interpreter for now.
- Cross-platform use; this is designed for MacOS only.

## Design
Library name: `idx` (for "Index")

### Configuration
Module: `idx.core.settings`

Configuration drives the behavior of the library on a given system.

Use `pydantic-settings` for library config. For the MVP we will support environment variables first; config-file support is deferred.

Settings needed:
- database_path aka DB_PATH (env var)
- vector_store_path aka VECTOR_STORE_PATH (env var) (LlamaIndex persist dir; rebuildable cache)
- embedding model
- transformers model 
- any API keys in the future
- logging level
- observability tool configuration
- default performance settings

### Orchestration
Module: `idx.pipelines`

Contains Ingestion Pipeline and well as Retrieval Pipeline. Client entry point, and so interface should be uncomplicated and accept Pydantic model shape.

### Source management
Module: `idx.source`

Contains any abstractions required for reading, parsing, or extracting a dataset source.

### Transformations
Module: `idx.transform`
Custom llamaindex.TransformComponent classes, used by `idx.pipelines`. Used to gather metadata and to transform resource content to be persisted to the relational database and indexed for full-text and vector similarity search.


### Ontology
Module: `idx.ontology`

Mostly a stub for now, but will contain the mappings required to "translate" a resource's metadata and content into structured metadata to be stored. For the MVP, ontology output should be stored as unstructured JSON attached to the Document record.
### Persistence
Module: `idx.store`

The following abstractions participate in content ingestion, storing artifacts into a relational database. The core technology is SQLAlchemy using SQLite, and should use the Database Path from configuration. The storage layer should not be accessed directly, instead it exposes a `DatasetService` which accepts Pydantic models that can be validated and then persisted. All persistence should live here, including database abstractions.

For the MVP we will **not** implement content-addressable storage in the relational database. We will still compute and store a `content_hash` per document for refresh semantics and vector node IDs.

- Dataset (database model): 
  - representation of the source - the thing that's retrieved by the llamaindex.Reader
  - should be globally identified by an URI which includes the dataset name with prefix `dataset:`

- Document (database model):
  - the post-transformation llamaindex.Document is persisted to the database
  - has a reference to its parent dataset
  - stores `path`, `active`, `content_hash`, and the full normalized `body` (text) required for FTS + chunking + reranking
  - stores source change-detection fields when available (`etag`, `last_modified`) to support fast refresh checks
  - stores extracted metadata (e.g., frontmatter/tags/ontology output) as JSON for the MVP

- Each database model has a corresponding Repository class abstracting storage access; repositories accept Database model instances as input.

- There is a single DatasetService that handles access to the repository layer; it accepts Pydantic Models as input, serializes them to database models, and sends those to the repositories.

- The relational database is the single source of truth for content.

#### Indexes
We support both full-text search and vector search, using SQLite FTS5 and LlamaIndex’s `VectorStoreIndex` (persisted via LlamaIndex `StorageContext` at `vector_store_path`). These are treated as derived caches that are trivially rebuildable from relational DB document bodies + deterministic chunking. If we wrap these via LlamaIndex abstractions, those wrappers must preserve the existing scoring and persistence behavior described above. If they require custom persistence wiring, it should reside in `idx.store`. These are also *not* covered by the DatasetService or Repository pattern requirements. 

Indexes are derived from the Documents and deterministic chunking, and must be rebuildable. We can provide this facility, or rely on the client to script or assemble it as needed.

### Search
Module: `idx.search`

Contains abstractions for full-text search, vector search, and hybrid retrieval (RRF) + LLM-as-judge reranking, including query rewriting, chunking, etc. These are expected to be called from the Orchestration layer.
We will use common LlamaIndex abstractions for now, as long as they meet our constraints.

Rank fusion strategy (simplicity trade-off):
- Hybrid search should use RRF across ranked lists, not a weighted average of raw scores.
- This trades off “absolute score interpretability” for “backend independence” and simpler portability (FTS and vector only need to produce a ranked list).

Search criteria should be produced by the client in Pydantic model shape.
Search resuolts should be returned in a Pydantic model shape or shapes.

### LLMs
Module: `idx.llm`

Contains abstractions for normalizing LLM providers. Allows us to leverage MLX (via `mlx-lm` and `mlx-embeddings` libraries) for LLM-as-judge, expected to be called from the Search or Orchestration layers. This abstraction must match the LlamaIndex format, so we can switch to something like OpenAI in the future.

We will **skip `dspy` initially** (simplicity). Prompts should be centrally-managed in `idx.llm.prompts.py`.

### Status
Module: `idx.core.status`

We should report on basic health of the system, such as LLM availability, database availability, vector store, etc. This can be a minimal stub.

## Features
### 1a. Ingest, transform, and persist a local directory
Ingest, transform, persist a local directory given its path and a list of file-matching glob statements.  Applies custom transforms for parsing and metadata gathering. Entry point is the Orchestration layer.

Dataset naming requirement:
- Dataset names are user-provided identifiers (URI-ified) and must be globally unique.
- Dataset names should be normalized into an exact URI-acceptable format (deterministic normalization) before uniqueness checks and persistence.
- Uniqueness must be validated at the top of the call chain (Orchestration layer) before any work begins.
### 1b. Ingest, transform, persist a local Obsidian vault 
Ingest, transform, persist a local Obsidian vault given its path. Applies custom transforms for parsing and metadata gathering. Entry point is the Orchestration layer.
- Creates a dataset representing the Vault
- Kicks off an Ingestion Pipeline containing custom transforms
- Identifies non-text documents by mime type. For now, it skips anything that's not text and emits a log message.
- Subsequent transforms retrieve the document's Obsidian frontmatter, and perform a mapping of tags and properties to a provided ontology (stub for now, just returns the tags as-is); this is stored as JSON metadata on the Document record.
- Chunking initially follows the existing `pmd` behavior (deterministic, line-oriented chunking with a byte budget) to preserve embedding refresh and vector provenance behavior. Trade-off: changes imply full re-chunk+re-embed for the document.
- Each Document is persisted to the database (including the normalized full text body, source metadata, and extracted JSON metadata).
- Both FTS and vector indexes should be populated for the document, and it should be ready for search.

### 2. Refresh a collected dataset
We should be able to repeat Feature 1 and only require the differential. For example, if only one obsidian note was changed and 10 added, we should only process those 11. Entry point is the Orchestration layer.

Refresh semantics replicate `pmd`:
- Use source metadata (`etag`/`last_modified`) + content hash to skip unchanged documents
- Soft-delete documents not present in the current enumeration (mark inactive + remove from FTS)
- Treat renames as delete+add (no special rename detection)
- Support `force` mode and explicit orphan cleanup for content/embeddings

Vector refresh strategy (simplicity trade-off):
- When a document’s content hash changes, delete all vector nodes for the *previous* `(hash, seq)` set, then insert the newly chunked nodes for the new hash.
- Changing the embedding model (or embedding dimension) requires a full rebuild of the vector store.
- Changing chunking configuration requires a full rebuild of the vector store.

### 3. Fulltext search
An end-user should be able to issue an FTS search command, and provide configuration values for behavior. Entry point is the Orchestration layer.
- Requirement: support both global (all datasets) search and in-dataset search (filter by dataset **name**).

### 4. Vector similarity search
An end user should be able to issue a vector search command, and provide configuration values for behavior. The embedding model should be configured in settings. Entry point is the Orchestration layer.
- Requirement: support both global (all datasets) search and in-dataset search (filter by dataset **name**).

### 5. Hybrid search (RRF)
An end user should be able to issue a hybrid search command, which should leverage RRF. Entry point is the Orchestration layer.
- Requirement: support both global (all datasets) search and in-dataset search (filter by dataset **name**).
- De-duplication strategy: results should be unique by `(dataset_name, path)`; within a document, keep only the best-scoring chunk.

Reranking unit (configurable):
- Support reranking based on either (a) the full document text or (b) the best-matching chunk text.
- This should be configurable via the ingestion/retrieval pipeline configuration, since it affects what content is readily available and how much text is passed to the reranker.

### 6. Hybrid search (RRF) + LLM as judge
An end user should be able to issue a hybrid search command, which should leverage RRF, with an additional step of LLM-as-judge. This must use the local MLX LLM Provider, with the model specified in settings. Entry point is the Orchestration layer.
- Requirement: support both global (all datasets) search and in-dataset search (filter by dataset **name**).

## Deferred / Out of scope for MVP
- Object store caching via `fsspec` and a `Resource` persistence model (source->cache copying and URL tracking).
- Relational DB content-addressable storage (CAS) for document/chunk bodies and cross-dataset storage deduplication.
- Persisting Chunks as first-class DB entities (and all related FK/repository/service wiring).
- Persisting Document/Chunk metadata as separate DB entities; for MVP metadata stays as JSON on Document.
- Global-search dedupe by content hash and a “multi-location per hash” result model (dataset+path list).
- Config-file based settings loading (beyond env vars) in `idx.core.settings`.
- Optional score normalization for hybrid + reranked final scores (0–1), beyond what’s required to match `pmd` behavior.
- “Comprehensive” unit/integration tests for every module/file/feature; MVP focuses on end-to-end coverage of ingest, refresh, and search paths.

### (Deferred) Tag retrieval and boosting
The existing `pmd` tag retrieval / metadata boosting approach should not be carried forward as-is. It needs a redesign aligned with the new ontology + metadata model, so it is out-of-scope for this proposal.

## Architecture

### Principles
- We value simplicity over all else.  
- We value strict module boundaries, and only want to export whats necessary.
  
### Constraints/assumptions
- MacOS-only; no cross-platform needed
- No CLIs or APIs needed at this time; we can include cli ability using `__main__` if needed.
- Offline support for now, but use LLM provider abstraction so we can drop in OpenAI later (LlamaIndex supports this)
- No legacy support/backwards compatibility needed for now; we maintain a single database schema version, which is changed at will (and subsequently tested)
- SQLite FTS5 should be used for full-text search

### Layout
- all code should live within the `idx` library, located in `src/idx`.
- tests should reside in the project root in `tests/idx`

### Data flow
- The entry point for external callers should be the Orchestration layer alone.
- Pydantic models should be used liberally anytime a caller has to provide diverse input.

### NFRs
- comprehensive logging using loguru; llamaindex built-in logging should also be used liberally , and these loggers should have the same destination (console for now)
- testing methods should follow LlamaIndex best-practices
- MVP testing focuses on end-to-end coverage for ingestion, refresh, and retrieval paths, with unit tests for high-risk logic (chunking, refresh decisions, scoring/fusion).
- Each python module should export symbols in `__all__`; however it should only export symbols that are **used** but external modules. Keep strict separation of concerns - this will be verified using the `tach` tool in the future.
- Performance considerations should be built into the client-facing layer; for example, the client creating an ingestion pipeline should be able to configure the concurrency.
- Observability should be self-hosted LangFuse, which is supported by LlamaIndex. Use LlamaIndex observability when its available
