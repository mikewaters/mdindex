# Requirements Specification

## Local MLX Embeddings + SimpleVectorStore + Markdown Chunking for Hybrid Reranked Search

---

## 1. Overview

Requirements define a fully local ingestion architecture where:

* Markdown documents are parsed into structurally meaningful nodes
* Embeddings are generated locally using **MLX** on Apple Silicon
* Embedded nodes are stored directly into **LlamaIndex `SimpleVectorStore`**
* Hybrid retrieval (lexical + vector) and reranking operate over high-quality chunks

This design enables offline vector search as part of an existing **hybrid reranked retrieval pipeline**.

Embedding MUST occur inside the ingestion pipeline when a vector store is attached.

---

## 2. Functional Requirements

---

### FR-1 — Pipeline SHALL parse Markdown documents into structured nodes

The ingestion pipeline SHALL ingest Markdown documents and chunk them using `MarkdownNodeParser`.

Chunking MUST preserve Markdown structural boundaries such as:

* headings
* sections
* nested header context

This provides semantically coherent topic units and useful metadata for retrieval.

---

### FR-2 — Chunking MUST support hybrid + reranked retrieval quality

The chunking strategy SHALL be designed to maximize:

* lexical recall (keyword/BM25 retrieval)
* semantic recall (vector similarity retrieval)
* reranker effectiveness over candidate chunks

Hybrid search is explicitly motivated by the complementary strengths of keyword and embedding retrieval.

---

### FR-3 — Pipeline SHALL enforce size-aware chunk boundaries

Markdown structural chunking alone MAY produce chunks that are:

* too large (topic dilution)
* too small (low embedding signal)

Therefore, the pipeline SHALL implement a two-stage chunking strategy:

1. Primary split via `MarkdownNodeParser`
2. Secondary fallback split for oversized nodes using a token/character-based splitter

This ensures embedding chunks remain retrieval-optimal.

Chunk-size sensitivity is a known driver of retrieval performance.

---

### FR-4 — Pipeline SHALL generate embeddings locally using MLX

When a vector store is attached, the pipeline SHALL include an embedding transformation stage.

Embedding MUST occur inside the pipeline before insertion.

Embeddings SHALL be generated locally using:

* `mlx-embeddings`
* MLX-compatible embedding checkpoints (e.g., E5 family)

The MLX library provides `load()` and `generate()` for embedding computation.

---

### FR-5 — Pipeline SHALL store vectors into `SimpleVectorStore`

The ingestion pipeline SHALL write embedded nodes directly into LlamaIndex’s built-in `SimpleVectorStore`.

This store is appropriate for:

* offline Mac deployments
* development and prototyping
* lightweight local hybrid search workflows

Persistence MAY be enabled through the storage context APIs.

---

### FR-6 — System SHALL support index creation from the populated vector store

After ingestion, the system SHALL instantiate a `VectorStoreIndex` directly from the populated vector store.

No re-processing or secondary embedding pass SHALL be required.

---

### FR-7 — Documents SHOULD include stable identifiers

Each ingested document SHOULD provide a stable `doc_id` for:

* provenance
* refresh workflows
* future deduplication and upserts

Stable identity is critical for scalable ingestion management.

---

## 3. Non-Functional Requirements

---

### NFR-1 — System MUST run fully offline on Apple Silicon

The embedding workflow MUST NOT depend on network calls.

All embedding computation SHALL run locally via MLX.

---

### NFR-2 — Chunking SHOULD preserve retrieval precision

Chunk boundaries SHOULD align with semantic units while remaining size-constrained.

Oversized chunks reduce embedding specificity; undersized chunks reduce signal.

Chunking MUST be evaluated using retrieval metrics such as Recall@K and MRR.

---

### NFR-3 — Vector storage SHOULD support persistence

Although `SimpleVectorStore` is in-memory by default, persistence SHOULD be supported via:

```python
storage_context.persist(...)
```

Persistence is the documented local storage mechanism.

---

## 4. Architecture Requirements

---

### AR-1 — Pipeline SHALL enforce correct transformation ordering

The ingestion pipeline MUST apply transformations in this order:

1. Markdown parsing + structural chunking
2. Size-aware fallback splitting
3. Local MLX embedding
4. Vector insertion into SimpleVectorStore

Embedding must precede vector insertion.

---

### AR-2 — Embedding backend SHALL integrate via `BaseEmbedding`

MLX embedding integration SHALL be implemented by subclassing `BaseEmbedding`.

Custom embedding backends are explicitly supported in LlamaIndex.

---

## 5. EXAMPLE Chunking Implementation (Markdown + Size Guardrail)

```python
# Date: January 25, 2026
# Name: ChatGPT
# Model: GPT-5.2 Thinking

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

# Primary: Markdown structure-aware splitting
markdown_parser = MarkdownNodeParser()

# Secondary: Size-aware fallback splitter
fallback_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

def chunk_markdown_nodes(doc):
    nodes = markdown_parser.get_nodes_from_documents([doc])

    final_nodes = []
    for node in nodes:
        if len(node.text) > 2000:  # guardrail threshold
            subnodes = fallback_splitter.get_nodes_from_documents([node])
            final_nodes.extend(subnodes)
        else:
            final_nodes.append(node)

    return final_nodes
```

MarkdownNodeParser provides structure-based sectioning; fallback splitting ensures embedding-sized chunks.

---

## 6. Acceptance Criteria

Option A SHALL be accepted as complete when:

* Markdown documents are chunked structurally via `MarkdownNodeParser`
* Oversized chunks are further split via fallback chunking
* Embeddings are generated locally via MLX
* Vectors are inserted into SimpleVectorStore
* Hybrid + reranked search achieves strong Recall@K before rerank
* No external API calls are required

---

## 7. Integration into existing implementation

Existing IngestionPipeline is **added to** only.  The only changes we will make is to the Vector embedding and vector indexing implementation; the current ingestion, persistence, and chunking is correct and must remain in place.

Changes that are required:
1. Vector embedding must be moved to the IngestionPipeline
2. `config.enable_vector_indexing` removed; Vector Indexing is *required* for this application to function.

Changes that are acceptable:
- Modification to persistence to use a StorageContext, effectively covering both pipeline persistence (current state) and vector persistence (target state) into a single storage context.
- Additional transforms in the INgestionPipeline, to perform additional splitting (if needed) to facilitate vector search. **The relational database structure shall not change.** The IngestionPipeline will have new initialization parameters, like a `SimpleVectorStore`.
- Refactoring of vector embedding or vector search modules (idx.store.vector, idx.search.vector).
- Update of transform module (idx.transform) to add new pipeline elements.

Changes that are not acceptable without approval:
1. Refactoring of the PersistenceTransform, MarkdownNodeParser, ChunkPersistenceTransform

Changes that are not acceptable
1. You shall not make any backward-compatible affordances - they are not necessary or desired.
