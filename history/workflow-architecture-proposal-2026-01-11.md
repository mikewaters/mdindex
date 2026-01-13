# Workflow Architecture Proposal (Ingestion + Search)

Date: 2026-01-11

## Goals

- Evolve the public interface to workflow/DAG pipelines for ingestion and search.
- Keep nodes granular enough for drop-in components (loaders, chunkers, embedding stores, etc.).
- Preserve flexibility on workflow engine and tracing instrumentation until a final choice is made.
- Enable stage-level tracing suitable for a UI and future agentic nodes.

## Current Orchestration Map

### Ingestion

Files: `src/pmd/services/loading.py`, `src/pmd/services/indexing.py`

Primary steps today:

- Resolve collection and source.
- Enumerate document references.
- Change detection via `SourceMetadataRepository` + `DocumentSource.check_modified`.
- Fetch content via `DocumentSource.fetch_content`.
- Hash check to skip unchanged.
- Extract title.
- Extract metadata via profile registry.
- Persist document content.
- FTS index update (index or remove).
- Persist source metadata.
- Persist document metadata.
- Cleanup stale documents.

### Embeddings

File: `src/pmd/services/indexing.py`

- Resolve collection.
- Load content for active docs.
- Skip documents with existing embeddings unless forced.
- `EmbeddingGenerator.embed_document` to create and store embeddings.

### Search

Files: `src/pmd/services/search.py`, `src/pmd/search/pipeline.py`

Primary steps today:

- Resolve collection id.
- Optional query expansion.
- Parallel FTS, vector, and tag search across query variants.
- RRF fusion.
- Optional metadata boost.
- Optional LLM rerank with blending.
- Normalize, filter, limit.

## Proposed Workflow Nodes and Contracts

The goal is to isolate orchestration into `src/pmd/workflows/` and keep services thin.

### Shared Contracts

New module: `src/pmd/workflows/contracts.py`

Suggested dataclasses (reuse existing types when possible):

- `IngestionRequest`: `{ collection_name, force, source_override? }`
- `ResolvedCollection`: `{ collection, source_config }`
- `ResolvedSource`: `{ source, collection }`
- `EnumeratedRefs`: `{ refs: list[DocumentReference], enumerated_paths: set[str] }`
- `LoadInput`: `{ ref, collection, source, force }`
- `LoadedDocument`: reuse `src/pmd/services/loading.py` type
- `PersistedDocument`: `{ doc_id, hash, path, content, is_indexable }`
- `IndexingSummary`: `{ indexed, skipped, errors }`
- `EmbedRequest`: `{ collection_name, force }`
- `EmbedSummary`: reuse `EmbedResult`
- `Chunk`: `{ chunk_id, doc_hash, text, start_char?, end_char?, metadata? }`
- `DocumentChunks`: `{ doc_hash, chunks: list[Chunk] }`
- `EmbeddingVector`: `{ doc_hash, chunk_id, vector: list[float], metadata? }`
- `EmbeddedChunks`: `{ doc_hash, vectors: list[EmbeddingVector] }`
- `StoreResult`: `{ stored, skipped }`

Search:

- `SearchRequest`: `{ query, collection_name?, limit, min_score, flags }`
- `ResolvedSearch`: `{ query, collection_id, config }`
- `SearchQueries`: `{ queries: list[str], weights: list[float] }`
- `ResultBatch`: `{ results: list[list[SearchResult]], weights: list[float] }`
- `FusedCandidates`: `{ candidates: list[RankedResult] }`
- `SearchResponse`: `{ results: list[RankedResult] }`

### Ingestion Pipeline Nodes

Proposed nodes (each corresponds to existing functions):

1) `resolve_collection`
   - Source: `IndexingService.index_collection`
   - Output: `ResolvedCollection`

2) `resolve_source`
   - Source: `LoadingService.load_collection_*` or `IndexingService._index_via_legacy`
   - Output: `ResolvedSource`

3) `enumerate_refs`
   - Source: `LoadingService.load_collection_*` enumeration logic
   - Output: `EnumeratedRefs`

4) `load_document` (map)
   - Source: `LoadingService._load_document`
   - Output: `LoadedDocument | None` (None indicates skip)

5) `persist_document`
   - Source: `IndexingService._persist_document`
   - Output: `PersistedDocument`

6) `index_fts`
   - Source: `IndexingService._persist_document` (indexing block)
   - Output: `FTSIndexResult`

7) `persist_source_metadata`
   - Source: `IndexingService._update_source_metadata`
   - Output: `SourceMetadataResult`

8) `persist_document_metadata`
   - Source: `IndexingService._persist_document_metadata`
   - Output: `DocumentMetadataResult`

9) `cleanup_stale_documents`
   - Source: `IndexingService._cleanup_stale_documents`
   - Output: `StaleCleanupResult`

10) `summarize_index`
   - Source: `IndexResult` aggregation in `index_collection`
   - Output: `IndexingSummary`

### Embedding Pipeline Nodes

11) `list_embed_targets`
    - Source: `IndexingService.embed_collection` SQL query
    - Output: `EmbedTargets { docs: list[{hash, path, doc}] }`

12) `chunk_document` (map)
    - Source: chunker implementation (new drop-in component)
    - Output: `DocumentChunks { doc_hash, chunks: list[Chunk] }`

13) `embed_chunks` (map)
    - Source: `EmbeddingGeneratorProtocol` (query/doc embedder split)
    - Output: `EmbeddedChunks { doc_hash, vectors: list[EmbeddingVector] }`

14) `store_vectors` (map)
    - Source: `EmbeddingRepositoryProtocol` (vector store writer)
    - Output: `StoreResult { stored, skipped }`

15) `summarize_embedding`
    - Source: `EmbedResult` aggregation in `embed_collection`

### Search Pipeline Nodes

1) `resolve_collection`
   - Source: `SearchService._resolve_collection_id`
   - Output: `ResolvedSearch`

2) `maybe_expand_query`
   - Source: `HybridSearchPipeline._expand_query`
   - Output: `SearchQueries`

3) `run_fts`
   - Source: `HybridSearchPipeline._parallel_search` (FTS branch)
   - Output: `ResultBatch`

4) `run_vector`
   - Source: `HybridSearchPipeline._parallel_search` (vector branch)
   - Output: `ResultBatch`

5) `run_tag_search`
   - Source: `HybridSearchPipeline._parallel_search` (tag branch)
   - Output: `ResultBatch`

6) `fuse_rrf`
   - Source: `reciprocal_rank_fusion`
   - Output: `FusedCandidates`

7) `metadata_boost`
   - Source: `HybridSearchPipeline._apply_metadata_boost`
   - Output: `FusedCandidates`

8) `rerank`
   - Source: `HybridSearchPipeline._rerank_with_blending`
   - Output: `FusedCandidates`

9) `normalize_scores`
   - Source: `normalize_scores`
   - Output: `FusedCandidates`

10) `filter_limit`
    - Source: end of `HybridSearchPipeline.search`
    - Output: `SearchResponse`

## Minimal Refactor Plan (Concrete Moves)

### 1) Add workflow package

New files:

- `src/pmd/workflows/contracts.py`
- `src/pmd/workflows/pipelines/ingestion.py`
- `src/pmd/workflows/pipelines/embedding.py`
- `src/pmd/workflows/pipelines/search.py`

### 2) Extract node logic from services

Ingestion:

- Move change detection + fetch + title + metadata extraction into `load_document` in `workflows/pipelines/ingestion.py`.
- Keep persistence as a single `persist_document` node initially (FTS + metadata + source metadata) to avoid over-fragmentation.
- `IndexingService.index_collection` becomes a thin wrapper: build pipeline inputs, run workflow, return summary.

Embeddings:

- Move `embed_collection` logic to `workflows/pipelines/embedding.py`.
- Split embedding into `chunk_document` -> `embed_chunks` -> `store_vectors` nodes.
- `IndexingService.embed_collection` becomes a thin wrapper.

Search:

- Move `HybridSearchPipeline` orchestration into `workflows/pipelines/search.py`.
- Keep `HybridSearchPipeline` as the internal implementation; defer full flattening into nodes until after engine choice.
- `SearchService.hybrid_search` becomes a thin wrapper that constructs workflow inputs and returns results.

### 3) Keep ports and adapters intact

- Keep `pmd.search.ports` and `pmd.search.adapters` as-is.
- The workflow nodes should depend on these ports, not concrete implementations.

### 4) Tracing and UI readiness

- Add a lightweight event interface in `contracts.py` or a separate `events.py`.
- Each node emits structured events; later bind these to OpenTelemetry or the chosen workflow engine.
- Preserve `pmd.core.instrumentation` and add a node-level wrapper after engine selection.

## Integration Points with Existing Services

### `IndexingService`

- Replace `_index_via_loader` and `_index_via_legacy` with workflow execution.
- Preserve `IndexResult`, `EmbedResult`, `CleanupResult` as return types.
- Keep `backfill_metadata` as a separate legacy utility (can be workflowed later).

### `LoadingService`

- `load_collection_stream` and `load_collection_eager` can remain.
- Internally used as node implementations or replaced by node logic directly.

### `SearchService`

- `fts_search` and `vector_search` can remain simple direct calls.
- `hybrid_search` becomes the workflow entry point.

## Decision Matrix (Engine-Agnostic)

This proposal does not choose a workflow engine. It prepares for both by:

- Centralizing node logic and data contracts in `pmd/workflows/`.
- Keeping execution as plain Python functions until you choose an engine.
- Keeping instrumentation independent via node event emission.

## Open Questions

- Should we promote `LoadingService` to be a shared node implementation, or replace it with pure workflow nodes?
- Should embeddings be part of the ingestion pipeline or always a separate pipeline?
- Should `HybridSearchPipeline` remain as the core implementation, or be fully flattened into node functions?
- Should FTS indexing + metadata persistence stay fused in a single node, or split into finer nodes later?
