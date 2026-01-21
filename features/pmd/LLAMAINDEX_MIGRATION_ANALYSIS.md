# LlamaIndex Migration Analysis

## Executive Summary

The current PMD codebase implements a custom RAG (Retrieval-Augmented Generation) system built directly on top of SQLite (using `sqlite-vec` for vectors and FTS5 for text). While LlamaIndex offers powerful abstractions that could theoretically replace 60-70% of the custom code, a full migration represents a **system rewrite** rather than a simple refactor due to the tight coupling between the storage layer, custom database schema, and the hybrid search pipeline.

We recommend a **phased adoption strategy**, starting with peripheral components (LLM integration, Data Loaders) before considering the core storage and retrieval engines.

## Component Analysis

### 1. Data Loading (Sources)
**Current Status:**
- Uses a custom `DocumentSource` protocol.
- Already has experimental `LlamaIndexSource` and `LlamaIndexLoaderAdapter`.

**LlamaIndex Equivalent:**
- `BaseReader` / `SimpleDirectoryReader`.

**Recommendation:**
- **Adopt.** The current `DocumentSource` protocol is compatible with LlamaIndex readers.
- Continue using `LlamaIndexSource` to wrap LlamaIndex readers.
- This allows access to LlamaIndex's massive library of data loaders (Hub) without changing the core system.

### 2. LLM Integration
**Current Status:**
- Custom `LLMProvider` abstract base class.
- Wrappers for embeddings and completion.

**LlamaIndex Equivalent:**
- `LLM` class (OpenAI, Anthropic, vLLM, etc.).
- `BaseEmbedding` class.

**Recommendation:**
- **Adopt.** Replace `LLMProvider` with LlamaIndex's `LLM` and `BaseEmbedding` abstractions.
- **Benefit:** Immediate support for dozens of providers, simplified retry logic, and standard callbacks.
- **Effort:** Low. Create an adapter that exposes LlamaIndex LLMs to the rest of PMD, or update PMD to use them directly.

### 3. Storage (Indexing)
**Current Status:**
- **Vector:** `EmbeddingRepository` manually manages `sqlite-vec` serialization, storage, and querying via raw SQL.
- **Text:** `FTS5SearchRepository` manages SQLite FTS5 tables.
- **Metadata:** Custom `documents` and `content` tables.

**LlamaIndex Equivalent:**
- `VectorStoreIndex`.
- `StorageContext`.
- `VectorStore` implementations (Chroma, Qdrant, PGVector, etc.).

**Recommendation:**
- **Hold (High Effort/Risk).**
- The current implementation is highly optimized for a single-file SQLite deployment (using `sqlite-vec`).
- LlamaIndex does support `SQLiteVectorStore`, but migrating would mean:
    1. Abandoning the specific `content_vectors` schema.
    2. Migrating existing data to LlamaIndex's node structure.
    3. Losing the tight integration between the `documents` table and the vector store.
- **Alternative:** Write a custom `LlamaIndex VectorStore` that wraps the *existing* `EmbeddingRepository`. This allows using LlamaIndex's higher-level features (agents, query engines) while keeping the efficient SQLite backend.

### 4. Search & Retrieval
**Current Status:**
- `HybridSearchPipeline`: A sophisticated, custom implementation featuring:
    - Reciprocal Rank Fusion (RRF).
    - Position-aware score blending.
    - Metadata-based boosting.
    - LLM Query Expansion.

**LlamaIndex Equivalent:**
- `RetrieverQueryEngine`.
- `QueryFusionRetriever` (supports RRF).
- `RouterRetriever`.
- `NodePostProcessor` (for reranking/boosting).

**Recommendation:**
- **Hold / Long-term Goal.**
- The `HybridSearchPipeline` contains specific domain logic (e.g., "Rank 1-3: 75% RRF + 25% reranker") that would need to be rewritten as custom `NodePostProcessors`.
- `QueryFusionRetriever` can replace the RRF logic, but the fine-tuned control of the current pipeline is valuable.
- Moving to LlamaIndex Retrievers makes sense only if the Storage layer is also migrated.

## Proposed Roadmap

### Phase 1: Integration (Low Hanging Fruit)
1.  **Unified LLM Interface:** Replace `src/pmd/llm` internals with LlamaIndex's `LLM` and `Embedding` classes. This simplifies model management.
2.  **Expanded Sources:** Finalize and promote `LlamaIndexSource` to allow users to ingest data from any LlamaIndex-supported source (Notion, Slack, etc.).

### Phase 2: Abstraction (Bridge Building)
1.  **Custom Vector Store:** Implement a `PMDSQLiteVectorStore` that adheres to LlamaIndex's `VectorStore` interface but writes to the existing `content_vectors` tables.
2.  **Retriever Adapter:** Create a LlamaIndex `BaseRetriever` that calls the existing `HybridSearchPipeline`.
    *   *Why?* This allows PMD to be used *as a tool* within LlamaIndex Agents, enabling complex workflows (e.g., "Search your notes and then summarize") while keeping the highly-tuned retrieval logic.

### Phase 3: Full Migration (Optional)
Only necessary if the custom SQLite architecture becomes a bottleneck.
1.  Replace `HybridSearchPipeline` with `QueryFusionRetriever`.
2.  Replace `EmbeddingRepository` with standard `VectorStoreIndex`.

## Conclusion
The PMD codebase has "reinvented the wheel" regarding RAG storage and retrieval, but it has done so with a high degree of specificity for SQLite. A complete rip-and-replace is risky. The best path forward is to **wrap** the existing robust retrieval logic in LlamaIndex interfaces, allowing the system to act as a powerful engine within the broader LlamaIndex ecosystem.
