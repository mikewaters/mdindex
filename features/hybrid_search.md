## Requirements: Hybrid Retrieval (FTS5 + Dense) with RRF + LLM Judge

### 0) Goals and non-goals

**Goals**

* Add **dense semantic retrieval** alongside the existing **SQLite FTS5** lexical retrieval.
* Use **LlamaIndex's `QueryFusionRetriever`** to combine lexical + dense results using **Reciprocal Rank Fusion (RRF)**. ([LlamaIndex][1])
* Use **LlamaIndex's `LLMRerank`** as an optional final LLM-as-judge reranker. ([LlamaIndex][2])
* Keep **SQLite FTS5** as the lexical index (no migration required).
* **Prefer LlamaIndex abstractions** wherever they exist to avoid reimplementing existing functionality.
* Maintain consistency with existing architectural patterns (ambient sessions, Pydantic models, etc.).

**Non-goals**

* Replacing FTS5 with a built-in hybrid vector store (LanceDB hybrid, etc.).
* Building a full response synthesizer; this spec focuses on retrieval + reranking.
* Reimplementing functionality that LlamaIndex already provides.

---

## 1) Storage & ID contract (must-have)

### 1.1 Canonical IDs

Hybrid search operates at **chunk level**. Every chunk/node must have a **stable canonical identifier** (`node_id`) that is:

* **Format**: `{content_hash}:{chunk_seq}` where:
  * `content_hash` is the SHA256 of the source document's normalized content
  * `chunk_seq` is the 0-indexed chunk sequence number within the document
* **Deterministic** across re-ingestion of the same content version
* **Shared** by both the FTS5 chunk index and the dense vector store
* **Example**: `a1b2c3d4e5f6...:0`, `a1b2c3d4e5f6...:1`

Both retrievers must return results referencing the same `node_id` so fusion can **dedupe** correctly. ([LlamaIndex][3])

**Document-level identifiers** (for grouping/filtering):
* `source_doc_id`: Composite key `{dataset_name}:{relative_path}` (e.g., `my-vault:notes/ideas.md`)
* `doc_id`: Integer primary key from `documents` table (internal use)

### 1.2 Node payload

Each chunk/node persisted must include (using LlamaIndex `TextNode` schema):

| Field | Type | Description |
|-------|------|-------------|
| `id_` | str | Canonical ID `{hash}:{seq}` (LlamaIndex node ID) |
| `text` | str | Chunk text content |
| `metadata.source_doc_id` | str | `{dataset_name}:{path}` |
| `metadata.doc_id` | int | Database document ID |
| `metadata.chunk_seq` | int | 0-indexed chunk sequence |
| `metadata.chunk_pos` | int | Character offset in source document |
| `metadata` | dict | Must not be `None`; use `{}` when empty ([GitHub][4]) |

These fields map to the existing `SearchResult` model in `src/idx/search/models.py`.

### 1.3 Persistence

**Vector store:**
* Dense vectors stored in **`SimpleVectorStore`** persisted to disk via LlamaIndex `StorageContext`. ([LlamaIndex][5])
* Location: `settings.vector_store_path` (default: `~/.idx/vector_store`)
* Use LlamaIndex's `StorageContext.persist()` and `StorageContext.from_defaults(persist_dir=...)` for save/load.
* Rationale: Lowest operational burden, sufficient for local deployments, easy to swap later.

**FTS store:**
* Chunks indexed in a **new FTS5 virtual table** (`chunks_fts`) separate from the existing document-level `documents_fts`.
* Location: Same SQLite database (`settings.database_path`)

**Consistency requirement:**
* Both stores must be updated atomically per-document during ingestion.
* On partial failure, roll back both stores to prevent orphaned entries.

### 1.4 Chunk-level FTS indexing (new requirement)

The existing `documents_fts` indexes full document bodies. For hybrid search with chunk-level fusion:

* Create `chunks_fts` virtual table with columns: `node_id`, `text`, `source_doc_id`
* Use `node_id` as the implicit rowid equivalent for lookups
* Wrap FTS queries in a LlamaIndex-compatible `CustomRetriever` that returns `NodeWithScore` objects

This ensures FTS and vector retrievers return results at the same granularity.

---

## 2) Ingestion pipeline changes (must-have)

### 2.1 Current pipeline (preserved)

The existing pipeline handles document-level persistence:

```
Source.documents (LlamaIndex Document)
    ↓
TextNormalizerTransform (normalize whitespace, BOM, etc.)
    ↓
PersistenceTransform (upsert to documents table + documents_fts)
    ↓
MarkdownNodeParser (split into chunks) [LlamaIndex]
    ↓
Nodes (TextNode with text + metadata)
```

### 2.2 Extended pipeline (new stages using LlamaIndex)

Add chunk-level processing after document persistence using **LlamaIndex's `IngestionPipeline`**:

```
Nodes (from MarkdownNodeParser)
    ↓
ChunkPersistenceTransform (new - custom)
    ├→ Assign node.id_ = {hash}:{seq}
    ├→ Upsert to chunks_fts
    └→ Track node_ids for vector indexing
    ↓
SentenceTransformersEmbedding [LlamaIndex built-in]
    ├→ Compute embeddings via settings.embedding_model
    └→ Batch automatically handled by LlamaIndex
    ↓
VectorStoreIndex.from_documents() or index.insert_nodes() [LlamaIndex]
    └→ Persist to SimpleVectorStore
```

**Key LlamaIndex abstractions to use:**
* `SentenceTransformersEmbedding` or `HuggingFaceEmbedding` for embeddings ([LlamaIndex][6])
* `IngestionPipeline` for orchestrating transforms ([LlamaIndex][6])
* `VectorStoreIndex` for vector storage and retrieval ([LlamaIndex][8])

### 2.3 Incremental updates & deletes (must-have)

**Upsert semantics:**
* By `node_id`: Use LlamaIndex's `index.insert_nodes()` with `allow_update=True`
* Change detection: Compare `content_hash`; skip unchanged documents

**Delete semantics:**
* By `source_doc_id`: Use `index.delete_ref_doc(ref_doc_id)` to remove all chunks for a document
* By `node_id`: Use `index.delete_nodes([node_id])` for specific chunks
* Triggered when: Source file deleted, document marked inactive, explicit API call

**Consistency guarantees:**
* Delete from vector store first, then FTS
* Use database transaction for FTS operations
* Log orphaned entries for manual cleanup if partial failure occurs

### 2.4 Performance & batching

LlamaIndex handles batching internally. Expose configuration:

| Setting | Default | Description |
|---------|---------|-------------|
| `IDX_EMBEDDING_BATCH_SIZE` | 32 | Passed to embedding model |
| `IDX_BATCH_SIZE` | 100 | General batch size for DB operations |
| `IDX_CHUNK_MAX_BYTES` | 2048 | Maximum chunk size |
| `IDX_CHUNK_MIN_BYTES` | 128 | Minimum chunk size |

---

## 3) Retrieval system requirements (must-have)

### 3.1 Two retrievers returning LlamaIndex `NodeWithScore`

Both retrievers must return `list[NodeWithScore]` for compatibility with LlamaIndex fusion.

**A) Lexical Retriever (FTS5) - Custom `BaseRetriever`**

Implement a custom retriever extending LlamaIndex's `BaseRetriever`:

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

class FTSChunkRetriever(BaseRetriever):
    """Custom retriever wrapping SQLite FTS5 chunk queries."""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Query chunks_fts, return NodeWithScore objects."""
```

* Query `chunks_fts` using FTS5 MATCH syntax
* BM25 scoring via `bm25(chunks_fts)` function
* Normalize scores to 0-1 range
* Return `NodeWithScore(node=TextNode(...), score=normalized_score)`

**B) Dense Retriever (Vector) - LlamaIndex built-in**

Use LlamaIndex's `VectorIndexRetriever`:

```python
from llama_index.core import VectorStoreIndex, StorageContext

# Load persisted index
storage_context = StorageContext.from_defaults(persist_dir=settings.vector_store_path)
index = load_index_from_storage(storage_context)

# Get retriever
vector_retriever = index.as_retriever(similarity_top_k=k_dense)
```

### 3.2 Fusion via LlamaIndex `QueryFusionRetriever`

Use **LlamaIndex's `QueryFusionRetriever`** with RRF mode ([LlamaIndex][1]):

```python
from llama_index.core.retrievers import QueryFusionRetriever

hybrid_retriever = QueryFusionRetriever(
    retrievers=[fts_chunk_retriever, vector_retriever],
    mode="reciprocal_rerank",  # RRF fusion
    similarity_top_k=k_fused,
    num_queries=1,  # No query expansion
    use_async=False,
)
```

**Fusion behavior:**
* Deduplicates by `node.id_` (chunk-level)
* Produces final ranked list with fused scores
* RRF formula: `score = 1 / (k + rank)` summed across retrievers

### 3.3 LLM-as-judge reranking using LlamaIndex `LLMRerank`

Use **LlamaIndex's `LLMRerank`** node postprocessor ([LlamaIndex][2]):

```python
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.mlx import MLX  # or other LLM provider

reranker = LLMRerank(
    llm=llm,
    choice_batch_size=5,
    top_n=top_k,
)

# Apply after retrieval
reranked_nodes = reranker.postprocess_nodes(
    nodes=fused_nodes,
    query_bundle=query_bundle,
)
```

**Feature flag:** Controlled by `SearchCriteria.rerank` (default `False`)

**Bounds:**
* Input (M): Max chunks sent to LLM (default: 20, max: 50)
* Output (N): `top_n` parameter (default: 10)

**LLM provider options:**
* **Local:** Use `llama_index.llms.mlx.MLX` for Apple Silicon
* **API:** Use `llama_index.llms.openai.OpenAI` or other providers
* Configurable via settings

---

## 4) Query pipeline requirements (must-have)

### 4.1 Query flow using LlamaIndex patterns

```
SearchCriteria(query, mode, rerank, ...)
    ↓
QueryBundle(query_str=query)
    ↓
┌─────────────────────────────────────────────┐
│ mode="fts"    → FTSChunkRetriever._retrieve │
│ mode="vector" → VectorIndexRetriever        │
│ mode="hybrid" → QueryFusionRetriever        │
└─────────────────────────────────────────────┘
    ↓
list[NodeWithScore]
    ↓
[if rerank=True]
    ↓
LLMRerank.postprocess_nodes()
    ↓
Convert to SearchResults (internal model)
```

**SearchCriteria** (existing model, ensure these fields):

```python
class SearchCriteria(BaseModel):
    query: str
    mode: Literal["fts", "vector", "hybrid"] = "hybrid"
    dataset_name: str | None = None
    limit: int = Field(default=10, ge=1, le=100)
    rerank: bool = False
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    # Hybrid tuning:
    k_lex: int = Field(default=20, ge=1, le=100)
    k_dense: int = Field(default=20, ge=1, le=100)
```

### 4.2 Unified search interface

Create `src/idx/search/service.py`:

```python
class SearchService:
    """Unified search interface using LlamaIndex retrievers."""

    def __init__(self):
        self._fts_retriever = FTSChunkRetriever(...)
        self._vector_retriever = None  # Lazy load
        self._fusion_retriever = None  # Lazy load
        self._reranker = None  # Lazy load

    def search(self, criteria: SearchCriteria) -> SearchResults:
        """Execute search based on criteria.mode."""
        query_bundle = QueryBundle(query_str=criteria.query)

        if criteria.mode == "fts":
            nodes = self._fts_retriever.retrieve(query_bundle)
        elif criteria.mode == "vector":
            nodes = self._get_vector_retriever().retrieve(query_bundle)
        else:  # hybrid
            nodes = self._get_fusion_retriever().retrieve(query_bundle)

        if criteria.rerank:
            nodes = self._get_reranker().postprocess_nodes(nodes, query_bundle)

        return self._convert_to_search_results(nodes[:criteria.limit])
```

### 4.3 Determinism & observability

**Debug info** (returned when `debug=True` or via logging):

```python
class SearchDebugInfo(BaseModel):
    k_lex: int
    k_dense: int
    k_fused: int
    k_final: int
    fts_results: list[tuple[str, float]]      # (node_id, score)
    vector_results: list[tuple[str, float]]   # (node_id, score)
    fused_results: list[tuple[str, float]]    # (node_id, rrf_score)
    reranked_results: list[tuple[str, float]] | None
    embedding_model: str
    index_version: str | None
```

**Reproducibility requirements:**
* Log `embedding_model` version with results
* Track `index_version` (hash of last ingestion run) for debugging
* Configuration snapshot available via `settings.model_dump()`

---

## 5) Configuration requirements (must-have)

All settings use `IDX_` environment variable prefix (existing pattern):

| Setting | Default | Description |
|---------|---------|-------------|
| `IDX_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `IDX_EMBEDDING_DIMENSIONS` | 384 | Must match model output |
| `IDX_VECTOR_STORE_PATH` | `~/.idx/vector_store` | SimpleVectorStore persist dir |
| `IDX_DATABASE_PATH` | `~/.idx/idx.db` | SQLite database path |
| `IDX_RERANK_ENABLED` | `false` | Default rerank behavior |
| `IDX_RERANK_CANDIDATES` | 20 | Default M for reranking |
| `IDX_RERANK_TOP_N` | 10 | Default N for reranking output |
| `IDX_LLM_PROVIDER` | `mlx` | LLM provider for reranking (mlx, openai) |

---

## 6) Acceptance criteria (tests)

### 6.1 Correctness

Given a test corpus in `tests/corpus/`:

* **FTS-only:** `mode="fts"` returns expected keyword matches
* **Vector-only:** `mode="vector"` returns semantically similar chunks
* **Hybrid:** `mode="hybrid"` returns superset of both (before truncation)
* **RRF ordering:** Results appearing in both lists rank higher than single-source results
* **Rerank:** When enabled, LLM judgment reorders results appropriately

### 6.2 Consistency

* **Idempotent ingestion:** Re-ingesting same document produces same `node_id`s
* **No duplicates:** Vector store and FTS have exactly one entry per `node_id`
* **Delete propagation:** Removing a document removes all its chunks from both stores

### 6.3 Performance envelopes

Set internal targets (not exposed as requirements):

| Operation | p95 Target | Notes |
|-----------|------------|-------|
| FTS retrieval | < 50ms | For 10K chunks |
| Vector retrieval | < 100ms | For 10K chunks |
| RRF fusion | < 10ms | Merging 40 results |
| LLM rerank | < 2s | 20 candidates, local MLX |

---

## 7) Implementation plan

### 7.1 File organization

```
src/idx/
├── search/
│   ├── __init__.py        # Export SearchService, SearchCriteria, SearchResults
│   ├── models.py          # SearchCriteria, SearchResult, SearchResults (existing)
│   ├── fts.py             # FTSSearch - document-level (existing, keep)
│   ├── fts_chunk.py       # FTSChunkRetriever(BaseRetriever) - chunk-level (new)
│   ├── vector.py          # VectorSearch wrapper (new, thin wrapper around LlamaIndex)
│   ├── hybrid.py          # HybridSearch using QueryFusionRetriever (new)
│   └── service.py         # SearchService unified interface (new)
├── store/
│   ├── fts.py             # FTSManager (existing)
│   ├── fts_chunk.py       # FTSChunkManager (new)
│   └── vector.py          # VectorStoreManager using LlamaIndex StorageContext (new)
├── transform/
│   └── llama.py           # Add ChunkPersistenceTransform (EmbeddingsTransform via LlamaIndex)
└── ingest/
    └── pipelines.py       # Extend pipeline with chunk + vector stages
```

### 7.2 Implementation order

1. **Storage layer:** `FTSChunkManager`, `VectorStoreManager` (wrapping LlamaIndex `StorageContext`)
2. **Ingestion transforms:** `ChunkPersistenceTransform` (embeddings via LlamaIndex built-in)
3. **Pipeline integration:** Extend `IngestPipeline` with new stages
4. **Retrievers:** `FTSChunkRetriever(BaseRetriever)`, thin `VectorSearch` wrapper
5. **Fusion:** `HybridSearch` using `QueryFusionRetriever`
6. **Rerank integration:** Wire `LLMRerank` into search pipeline
7. **Unified interface:** `SearchService`
8. **Tests:** Integration tests with `tests/corpus`

### 7.3 LlamaIndex abstractions to use

| Component | LlamaIndex Class | Custom Code Needed |
|-----------|-----------------|-------------------|
| Embeddings | `HuggingFaceEmbedding` | None |
| Vector Store | `SimpleVectorStore` | None |
| Vector Index | `VectorStoreIndex` | None |
| Storage | `StorageContext` | None |
| Fusion | `QueryFusionRetriever` | None |
| Reranking | `LLMRerank` | None |
| FTS Retriever | `BaseRetriever` | Custom subclass |
| Chunk Persistence | `TransformComponent` | Custom subclass |

### 7.4 Architectural guardrails

* **Ambient sessions:** All DB operations use `current_session()` pattern
* **Pydantic models:** All inputs/outputs are validated models
* **LlamaIndex-first:** Use LlamaIndex abstractions wherever they exist
* **Existing code preserved:** `FTSSearch` (document-level) remains for backwards compatibility

---

## References

[1]: https://developers.llamaindex.ai/python/examples/retrievers/reciprocal_rerank_fusion/ "Reciprocal Rerank Fusion Retriever"
[2]: https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/ "Node Postprocessor Modules"
[3]: https://developers.llamaindex.ai/python/examples/low_level/fusion_retriever/ "Building an Advanced Fusion Retriever from Scratch"
[4]: https://github.com/run-llama/llama_index/issues/12311 "[Question]: error when I run the llmrerank #12311"
[5]: https://developers.llamaindex.ai/python/framework/module_guides/storing/save_load/ "Persisting & Loading Data"
[6]: https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/ "Ingestion Pipeline | LlamaIndex Python Documentation"
[7]: https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/transformations/ "Transformations | LlamaIndex Python Documentation"
[8]: https://developers.llamaindex.ai/python/framework/module_guides/indexing/vector_store_index/ "Using VectorStoreIndex | LlamaIndex Python Documentation"
[9]: https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/ "Node Postprocessor | LlamaIndex Python Documentation"
