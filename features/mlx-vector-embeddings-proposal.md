# Implementation Proposal: Local MLX Embeddings Integration

**Status**: Draft
**Date**: 2026-01-25
**Requirements**: [mlx-vector-embeddings.md](./mlx-vector-embeddings.md)

---

## Executive Summary

This proposal implements local MLX-based vector embeddings integrated directly into the IngestionPipeline. The key architectural change is moving embedding computation from post-pipeline processing into a proper pipeline transform, making vector indexing a required (non-optional) component.

---

## 1. Current State Analysis

### What Works Today

| Component | Location | Status |
|-----------|----------|--------|
| MarkdownNodeParser | `ingest/pipelines.py:13-17` | Structural chunking in place |
| SimpleVectorStore | `store/vector.py` | VectorStoreManager wraps LlamaIndex |
| Batch Embedding | `ingest/pipelines.py:154-208` | HuggingFaceEmbedding post-pipeline |
| Hybrid Search | `search/hybrid.py` | FTS5 + vector with RRF fusion |
| StorageContext | `store/vector.py` | JSON persistence to persist_dir |

### What Must Change

1. **Embedding Location**: Currently post-pipeline, must move into pipeline as transform
2. **Optional → Required**: `enable_vector_indexing` removed; vector indexing always happens
3. **Embedding Backend**: HuggingFace → MLX for Apple Silicon local inference

---

## 2. Architecture Changes

### 2.1 New Pipeline Transform Order

```
IngestionPipeline.run(documents)
├── TextNormalizerTransform       # Whitespace/BOM cleanup
├── PersistenceTransform          # documents → SQLite + documents_fts
├── MarkdownNodeParser            # Structural chunking
├── ChunkPersistenceTransform     # chunks → chunks_fts
├── SizeAwareChunkSplitter [NEW]  # Fallback split oversized nodes
└── EmbeddingTransform [NEW]      # MLX embedding + vector insertion
```

### 2.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `SizeAwareChunkSplitter` | Detect oversized nodes (>2000 chars), split via SentenceSplitter |
| `MLXEmbedding` | LlamaIndex BaseEmbedding implementation using mlx-embeddings |
| `EmbeddingTransform` | Pipeline transform that embeds nodes and inserts into vector store |

---

## 3. Detailed Design

### 3.1 MLXEmbedding Class

**Location**: `src/idx/embedding/mlx.py`

```python
from llama_index.core.embeddings import BaseEmbedding
from mlx_embeddings import load, generate

class MLXEmbedding(BaseEmbedding):
    """Local embedding generation using MLX on Apple Silicon.

    Implements LlamaIndex BaseEmbedding interface using mlx-embeddings
    library for fully offline inference.
    """

    model_name: str = "mlx-community/e5-small-v2-mlx"

    def __init__(
        self,
        model_name: str = "mlx-community/e5-small-v2-mlx",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            self._model, self._tokenizer = load(self.model_name)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        self._load_model()
        output = generate(self._model, self._tokenizer, [text])
        return output[0].tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        self._load_model()
        outputs = generate(self._model, self._tokenizer, texts)
        return [emb.tolist() for emb in outputs]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async wrapper (MLX is sync-only)."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper (MLX is sync-only)."""
        return self._get_text_embeddings(texts)
```

**Design Notes**:
- Lazy model loading to avoid startup cost
- Implements all required BaseEmbedding methods
- Async methods wrap sync (MLX doesn't support async)
- Model name configurable, defaults to E5-small MLX variant

### 3.2 SizeAwareChunkSplitter Transform

**Location**: `src/idx/transform/splitter.py`

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent

class SizeAwareChunkSplitter(TransformComponent):
    """Fallback splitter for oversized nodes.

    Nodes exceeding max_chars are split using SentenceSplitter while
    preserving semantic boundaries. Smaller nodes pass through unchanged.
    """

    max_chars: int = 2000
    fallback_chunk_size: int = 512
    fallback_chunk_overlap: int = 50

    def __init__(
        self,
        max_chars: int = 2000,
        fallback_chunk_size: int = 512,
        fallback_chunk_overlap: int = 50,
    ):
        self.max_chars = max_chars
        self._splitter = SentenceSplitter(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
        )

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs,
    ) -> list[BaseNode]:
        """Split oversized nodes, pass through others unchanged."""
        result = []
        for node in nodes:
            if len(node.text) > self.max_chars:
                # Fallback split preserves metadata
                subnodes = self._splitter.get_nodes_from_documents([node])
                result.extend(subnodes)
            else:
                result.append(node)
        return result
```

**Design Notes**:
- Only affects oversized nodes (configurable threshold)
- Uses LlamaIndex SentenceSplitter for semantic splitting
- Preserves node metadata during splitting

### 3.3 EmbeddingTransform

**Location**: `src/idx/transform/embedding.py`

```python
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.embeddings import BaseEmbedding

from idx.store.vector import VectorStoreManager

class EmbeddingTransform(TransformComponent):
    """Pipeline transform that computes embeddings and inserts into vector store.

    This transform:
    1. Computes embeddings for all input nodes
    2. Inserts embedded nodes into SimpleVectorStore
    3. Returns nodes unchanged (for pipeline continuation)
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        vector_store_manager: VectorStoreManager,
        batch_size: int = 32,
    ):
        self._embed_model = embed_model
        self._vector_store_manager = vector_store_manager
        self._batch_size = batch_size

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs,
    ) -> list[BaseNode]:
        """Embed nodes and insert into vector store."""
        if not nodes:
            return nodes

        # Batch embed
        texts = [node.get_content() for node in nodes]
        embeddings = self._embed_model.get_text_embedding_batch(
            texts,
            show_progress=True,
        )

        # Attach embeddings to nodes
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        # Insert into vector store
        self._vector_store_manager.insert_nodes(nodes)

        return nodes
```

**Design Notes**:
- Transform persists vectors during pipeline execution
- Batch processing for efficiency
- Returns nodes unchanged for potential downstream transforms
- Uses VectorStoreManager abstraction (already exists)

### 3.4 Settings Changes

**Location**: `src/idx/core/settings.py`

```python
class EmbeddingSettings(BaseSettings):
    """Embedding configuration."""

    backend: Literal["mlx", "huggingface"] = "mlx"
    model_name: str = "mlx-community/e5-small-v2-mlx"
    batch_size: int = 32


class Settings(BaseSettings):
    # ... existing fields ...

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
```

**Remove from IngestDirectoryConfig/IngestObsidianConfig**:
```python
# DELETE: enable_vector_indexing: bool = False
```

### 3.5 Pipeline Factory Updates

**Location**: `src/idx/ingest/pipelines.py`

```python
def _build_pipeline(self, config: IngestConfig) -> IngestionPipeline:
    """Build ingestion pipeline with embedding transform."""

    # Existing transforms (unchanged)
    normalizer = TextNormalizerTransform()
    doc_persist = PersistenceTransform(...)
    md_parser = MarkdownNodeParser(
        include_metadata=True,
        include_prev_next_rel=True,
        header_path_separator=" / ",
    )
    chunk_persist = ChunkPersistenceTransform(...)

    # NEW: Size-aware splitting
    size_splitter = SizeAwareChunkSplitter(
        max_chars=2000,
        fallback_chunk_size=512,
        fallback_chunk_overlap=50,
    )

    # NEW: Embedding transform
    embed_model = self._get_embed_model()
    embedding_transform = EmbeddingTransform(
        embed_model=embed_model,
        vector_store_manager=self._get_vector_store_manager(),
        batch_size=self._settings.embedding.batch_size,
    )

    return IngestionPipeline(
        transformations=[
            normalizer,
            doc_persist,
            md_parser,
            chunk_persist,
            size_splitter,       # NEW
            embedding_transform,  # NEW
        ],
        # ... existing params ...
    )


def _get_embed_model(self) -> BaseEmbedding:
    """Get embedding model based on settings."""
    if self._embed_model is None:
        if self._settings.embedding.backend == "mlx":
            from idx.embedding.mlx import MLXEmbedding
            self._embed_model = MLXEmbedding(
                model_name=self._settings.embedding.model_name,
            )
        else:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._settings.embedding.model_name,
            )
    return self._embed_model
```

**Remove from `ingest()` method**:
```python
# DELETE: All code in the `if config.enable_vector_indexing:` block
# This logic is now handled by EmbeddingTransform
```

---

## 4. File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `src/idx/embedding/__init__.py` | Embedding module init |
| `src/idx/embedding/mlx.py` | MLXEmbedding implementation |
| `src/idx/transform/splitter.py` | SizeAwareChunkSplitter |
| `src/idx/transform/embedding.py` | EmbeddingTransform |

### Modified Files

| File | Changes |
|------|---------|
| `src/idx/core/settings.py` | Add EmbeddingSettings, remove enable_vector_indexing |
| `src/idx/ingest/pipelines.py` | New pipeline transforms, remove post-pipeline embedding |
| `src/idx/ingest/config.py` | Remove enable_vector_indexing from configs |
| `pyproject.toml` | Add mlx-embeddings dependency |

### Unchanged Files (Per Requirements)

| File | Reason |
|------|--------|
| `src/idx/transform/llama.py` | Contains PersistenceTransform, ChunkPersistenceTransform |
| `src/idx/store/vector.py` | VectorStoreManager remains unchanged |
| `src/idx/store/models.py` | Relational schema unchanged |

---

## 5. Dependencies

### New Dependencies

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "mlx-embeddings>=0.1.0",
]
```

### MLX Model Selection

Recommended models from mlx-community (Hugging Face):

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `mlx-community/e5-small-v2-mlx` | 384 | Default, good balance |
| `mlx-community/e5-base-v2-mlx` | 768 | Better quality, larger |
| `mlx-community/bge-small-en-v1.5-mlx` | 384 | Alternative family |

---

## 6. Migration Path

### Breaking Change Notice

This change removes `enable_vector_indexing`. Existing users must:

1. Remove `enable_vector_indexing=True` from their ingest calls
2. Vector indexing happens automatically on all ingests

### Data Migration

No data migration required:
- Existing SQLite data (documents, chunks, FTS) unchanged
- Existing vector store files (if any) remain compatible
- Re-ingest with `force=True` to regenerate embeddings with MLX

---

## 7. Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| `test_mlx_embedding.py` | MLXEmbedding interface compliance |
| `test_size_aware_splitter.py` | Oversized node splitting |
| `test_embedding_transform.py` | Embedding + vector insertion |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_pipeline_with_mlx.py` | Full pipeline with MLX embeddings |
| `test_hybrid_search_mlx.py` | Hybrid search over MLX-embedded nodes |

### Acceptance Criteria Verification

Per requirements section 6:

- [ ] Markdown documents chunked via MarkdownNodeParser
- [ ] Oversized chunks split via SizeAwareChunkSplitter
- [ ] Embeddings generated locally via MLX (no network)
- [ ] Vectors inserted into SimpleVectorStore
- [ ] Hybrid + reranked search achieves strong Recall@K
- [ ] No external API calls during embedding

---

## 8. Implementation Order

### Phase 1: MLX Embedding Module

1. Create `src/idx/embedding/` module
2. Implement `MLXEmbedding` class
3. Add unit tests for embedding interface
4. Add mlx-embeddings to dependencies

### Phase 2: Pipeline Transforms

1. Create `SizeAwareChunkSplitter` in `src/idx/transform/splitter.py`
2. Create `EmbeddingTransform` in `src/idx/transform/embedding.py`
3. Add unit tests for both transforms

### Phase 3: Pipeline Integration

1. Update `Settings` with `EmbeddingSettings`
2. Remove `enable_vector_indexing` from config classes
3. Update `IngestPipeline._build_pipeline()` with new transforms
4. Remove post-pipeline embedding code
5. Add integration tests

### Phase 4: Verification

1. Run full ingestion on test corpus
2. Verify hybrid search quality
3. Measure embedding throughput
4. Document any performance tuning needed

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| MLX model download on first use | Document that first run downloads model; add model preload CLI |
| Memory pressure on large batches | Configurable batch_size, default conservative |
| Non-Apple Silicon platforms | Fallback to HuggingFace via settings.embedding.backend |
| Embedding dimension mismatch | Validate dimensions match existing vectors on startup |

---

## 10. Open Questions

1. **Model Preloading**: Should we add a CLI command `idx model download` to preload the MLX model before first ingest?

2. **HuggingFace Fallback**: Should we auto-detect non-Apple Silicon and fallback, or require explicit configuration?

3. **Embedding Dimension Validation**: If existing vectors have different dimensions than new model, should we warn/error/clear?

---

## 11. Appendix: Code Locations Reference

| Current Code | Lines | Relevance |
|--------------|-------|-----------|
| `ingest/pipelines.py:_compute_embeddings` | 179-208 | Code to remove |
| `ingest/pipelines.py:_insert_vectors` | 210-230 | Code to remove |
| `ingest/pipelines.py:154-176` | Post-pipeline embedding block | Code to remove |
| `store/vector.py:VectorStoreManager` | All | Reused unchanged |
| `core/settings.py:Settings` | All | Add EmbeddingSettings |
| `transform/llama.py` | All | Not modified |
