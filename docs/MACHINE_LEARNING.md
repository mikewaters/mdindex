# Machine Learning

This document provides technical documentation on how machine learning is leveraged in the PMD (Prompt Management Database) codebase. It is written for ML practitioners who want to understand the implementation details.

---

## Table of Contents

1. [BM25 Full-Text Search](#1-bm25-full-text-search)
2. [Vector Store and Embeddings](#2-vector-store-and-embeddings)
3. [LLM Integration](#3-llm-integration)
4. [Hybrid Search Pipeline](#4-hybrid-search-pipeline)

---

## 1. BM25 Full-Text Search

### 1.1 Implementation Overview

PMD uses **SQLite FTS5** (Full-Text Search 5) for BM25 ranking rather than a custom implementation. This leverages the mature, battle-tested FTS5 module built into SQLite.

**Core Files:**
- `src/pmd/store/search.py` - FTS5SearchRepository
- `src/pmd/store/schema.py` - Database schema
- `src/pmd/search/fusion.py` - Reciprocal Rank Fusion

### 1.2 BM25 Parameters

The implementation uses FTS5's default BM25 parameters (not exposed for tuning):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| k1 | ~1.2 | Term frequency saturation |
| b | ~0.75 | Document length normalization |

### 1.3 Tokenization

FTS5 index definition (`src/pmd/store/schema.py:43-46`):

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, body,
    tokenize='porter unicode61'
);
```

**Tokenizer characteristics:**
- **Porter stemming**: Aggressive term unification (running → run)
- **Unicode61 collation**: Case-insensitive, handles 61 Unicode character categories
- **Indexed fields**: `path` (filepath) and `body` (document content)

### 1.4 Score Normalization

BM25 scores undergo two-pass max-normalization (`src/pmd/store/search.py:112-227`):

```python
# FTS5 returns negative BM25 scores via .rank
# First pass: Find maximum absolute rank
max_rank = min(abs(row["fts_rank"]) for row in rows)

# Second pass: Normalize to [0, 1]
raw_score = abs(row["fts_rank"])
normalized_score = raw_score / max_rank if max_rank else 0.0
```

This produces scores in the range [0, 1] where 1.0 represents the best match.

### 1.5 Query Processing

Queries are passed directly to FTS5 with minimal preprocessing (`src/pmd/store/search.py:318-332`):

```python
@staticmethod
def _prepare_fts_query(query: str) -> str:
    return query  # Pass-through to FTS5
```

**Supported FTS5 operators:**
- `AND`, `OR`, `NOT` for boolean logic
- Double quotes for phrase matching
- Porter stemming applied automatically at query time

### 1.6 Limitations

1. **Fixed BM25 parameters** - Cannot tune k1/b for specific domains
2. **No field weighting** - Path and body fields treated equally
3. **Simple query preparation** - No spelling correction or fuzzy matching

---

## 2. Vector Store and Embeddings

### 2.1 Embedding Model Architecture

**Primary Model:** MLX community's NomicAI ModernBERT Embed Base (4-bit quantized)

| Property | Value |
|----------|-------|
| Model ID | `mlx-community/nomicai-modernbert-embed-base-4bit` |
| Embedding Dimension | 768 |
| Quantization | 4-bit for Apple Silicon |
| Alternative | `multilingual-e5-small` (384 dimensions) |

**Asymmetric Embeddings:**

The model uses query/document prefix specialization per NomicAI spec:

```python
# Query embedding
query_embedding = embed("search_query: machine learning clustering")

# Document embedding
doc_embedding = embed("search_document: k-means algorithm clustering")
```

Configuration (`src/pmd/core/config.py`):
```python
MLXConfig:
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
```

### 2.2 Vector Storage

**Backend:** SQLite with optional `sqlite-vec` extension for native vector operations.

**Schema** (`src/pmd/store/schema.py`):

```sql
-- Metadata storage (always available)
CREATE TABLE content_vectors (
    hash TEXT NOT NULL,           -- SHA256 of content
    seq INTEGER NOT NULL,         -- Chunk sequence (0, 1, 2, ...)
    pos INTEGER NOT NULL,         -- Character position in document
    model TEXT NOT NULL,          -- Model name
    embedded_at TEXT NOT NULL,    -- ISO timestamp
    PRIMARY KEY (hash, seq)
);

-- Vector storage (sqlite-vec virtual table, optional)
CREATE VIRTUAL TABLE content_vectors_vec USING vec0(
    hash TEXT PRIMARY KEY,        -- Composite key: "{hash}:{seq}"
    seq INTEGER,
    embedding FLOAT[768]          -- 768-d vector
);
```

**Design decisions:**
- **Dual storage** - Metadata always stored; vectors only if sqlite-vec available
- **Content-addressed** - Embeddings keyed by content hash, enabling deduplication
- **Graceful degradation** - Vector search disabled if extension unavailable

### 2.3 Similarity Search

**Distance metric:** L2 (Euclidean) distance via sqlite-vec

**Score conversion:**
```python
# L2 distance → Similarity score
score = 1 / (1 + distance)
# Distance range: [0, ∞) → Similarity range: [0, 1)
```

**Query structure** (`src/pmd/store/embeddings.py`):
```sql
SELECT ... FROM content_vectors_vec v
WHERE v.embedding MATCH ? AND k = ?
ORDER BY v.distance ASC
```

### 2.4 Document Chunking

Documents are chunked before embedding (`src/pmd/llm/embeddings.py`):

```python
ChunkConfig:
    max_bytes: int = 6 * 1024   # ~2000 tokens per chunk
    min_chunk_size: int = 100   # Minimum chunk size
```

Each chunk produces an independent embedding stored with position metadata:
```python
store_embedding(
    hash_value=hash,
    seq=seq,           # Chunk sequence number
    pos=chunk.pos,     # Character position in original
    embedding=embedding,
    model=model,
)
```

### 2.5 Embedding Extraction

MLX handles multiple model output formats (`src/pmd/llm/mlx_provider.py`):

```python
# Priority hierarchy:
# 1. text_embeds: Mean-pooled, normalized (ModernBERT - best for similarity)
if hasattr(result, "text_embeds"):
    embedding = result.text_embeds.tolist()[0]

# 2. pooler_output: CLS token (BERT variants)
elif hasattr(result, "pooler_output"):
    embedding = result.pooler_output.tolist()[0]

# 3. Fallback: Mean pooling over last_hidden_state
elif hasattr(result, "last_hidden_state"):
    embedding = result.last_hidden_state.mean(axis=1).tolist()[0]
```

---

## 3. LLM Integration

### 3.1 Provider Architecture

PMD implements a provider abstraction supporting three backends:

**Abstract Interface** (`src/pmd/llm/base.py`):
```python
class LLMProvider(ABC):
    async def embed(text, model, is_query) -> EmbeddingResult | None
    async def generate(prompt, model, max_tokens, temperature) -> str | None
    async def rerank(query, documents, model) -> RerankResult
    async def model_exists(model) -> bool
    async def is_available() -> bool
```

### 3.2 Provider Implementations

| Provider | Platform | Default Model | Latency |
|----------|----------|---------------|---------|
| **MLX** | macOS Apple Silicon | Qwen2.5-1.5B-Instruct-4bit | 50-200ms |
| **LM Studio** | Cross-platform | Configurable | 100-500ms |
| **OpenRouter** | Cloud | qwen/qwen-1.5-0.5b | 500ms-2s |

**MLX Provider** (`src/pmd/llm/mlx_provider.py`):
- Uses `mlx-lm` and `mlx-embeddings` packages
- Models auto-downloaded from HuggingFace
- Applies chat templates for instruction-tuned models:
```python
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

**LM Studio Provider** (`src/pmd/llm/lm_studio.py`):
- OpenAI-compatible API at `localhost:1234`
- Supports any model loaded in LM Studio

**OpenRouter Provider** (`src/pmd/llm/openrouter.py`):
- Cloud API gateway to 200+ models
- Requires `OPENROUTER_API_KEY` environment variable

### 3.3 Query Expansion

Query expansion generates semantic variations to improve recall (`src/pmd/llm/query_expansion.py`):

```python
prompt = f"""Generate {num_variations} alternative phrasings for this search query
that capture the same intent but use different wording. Return ONLY the variations,
one per line, with no numbering, bullets, or explanations.

Original Query: {query}

Alternative Phrasings:"""

response = await llm.generate(prompt, max_tokens=100, temperature=0.7)
```

**Parsing logic:**
- Removes numbering (1., 2., 3.)
- Strips bullets (-, *, •)
- Removes trailing punctuation
- Returns `[original_query, variation1, variation2, ...]`

### 3.4 Document Reranking

Binary relevance judgment via LLM (`src/pmd/llm/reranker.py`):

```python
prompt = """You are a relevance judge. Given a query and a document,
respond with ONLY 'Yes' if the document is relevant to the query,
or 'No' if it is not relevant.

Query: {query}

Document: {doc['body'][:1000]}

Is this document relevant? Answer with Yes or No only:"""
```

**Generation parameters:**
- `max_tokens`: 1 (minimal output)
- `temperature`: 0.0 (deterministic)

**Score mapping:**
```python
answer = response.strip().lower()
relevant = answer.startswith("yes")
confidence = 0.9 if relevant else 0.1
score = 0.5 + 0.5 * confidence if relevant else 0.5 * (1 - confidence)
# Yes → ~0.95, No → ~0.05, Error → 0.5
```

### 3.5 Model Loading

MLX supports lazy loading to defer model initialization:

```python
MLXConfig:
    lazy_load: bool = True  # Load on first use (default)
```

Memory management methods:
```python
provider.unload_model()           # Free text generation model
provider.unload_embedding_model() # Free embedding model
provider.unload_all()             # Free all models
```

### 3.6 Error Handling

All LLM operations return `None` on failure, enabling graceful degradation:

```python
async def embed(self, text, ...) -> EmbeddingResult | None:
    try:
        return EmbeddingResult(embedding=embedding, model=model)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None
```

Query expansion falls back to original query on failure. Reranking returns neutral scores (0.5) on error.

---

## 4. Hybrid Search Pipeline

### 4.1 Architecture Overview

The hybrid search pipeline (`src/pmd/search/pipeline.py`) orchestrates BM25 and vector search:

```
Query Text
    ↓
[Optional] Query Expansion (LLM)
    ├─ FTS Search (all query variants)
    └─ Vector Search (original embedding)
    ↓
Reciprocal Rank Fusion (RRF)
    ↓
[Optional] LLM Reranking
    ↓
Position-Aware Score Blending
    ↓
Final Score Normalization → [0, 1]
```

### 4.2 Reciprocal Rank Fusion

RRF combines multiple ranked lists by rank position (`src/pmd/search/fusion.py`):

```python
RRF_Score = Σ(weight / (k + rank + 1)) + bonuses

# Parameters:
# k = 60 (smoothing constant)
# rank = 0-indexed position
# weight = 2.0 for original query, 1.0 for expanded queries
# bonuses: +0.05 for rank 0, +0.02 for ranks 1-2
```

### 4.3 Position-Aware Score Blending

When reranking is enabled, scores are blended based on result position (`src/pmd/search/scoring.py`):

| Rank | RRF Weight | Reranker Weight | Rationale |
|------|------------|-----------------|-----------|
| 1-3 | 75% | 25% | Trust initial retrieval for top results |
| 4-10 | 60% | 40% | Balanced weighting |
| 11+ | 50% | 50% | Rely more on reranker for edge cases |

### 4.4 Result Data Structure

Final results include full provenance (`src/pmd/core/types.py`):

```python
@dataclass
class RankedResult:
    file: str
    score: float                   # Final blended score [0, 1]
    fts_score: Optional[float]     # Original BM25 score
    vec_score: Optional[float]     # Original vector score
    rerank_score: Optional[float]  # LLM reranker score
    fts_rank: Optional[int]        # Position in FTS results
    vec_rank: Optional[int]        # Position in vector results
    sources_count: int             # Sources that found this doc
    blend_weight: Optional[float]  # Position-aware weight used
```

### 4.5 Configuration

Pipeline configuration (`src/pmd/search/pipeline.py`):

```python
@dataclass
class SearchPipelineConfig:
    fts_weight: float = 1.0              # RRF weight for FTS
    vec_weight: float = 1.0              # RRF weight for vector
    rrf_k: int = 60                      # RRF smoothing constant
    top_rank_bonus: float = 0.05         # Bonus for rank 0
    expansion_weight: float = 0.5        # Weight for expanded queries
    rerank_candidates: int = 30          # Top-k for reranking
    enable_query_expansion: bool = False
    enable_reranking: bool = False
    normalize_final_scores: bool = True
```

### 4.6 Observability

Optional OpenTelemetry tracing (`src/pmd/core/instrumentation.py`):

```python
TracingConfig:
    enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    service_name: str = "pmd"
    sample_rate: float = 1.0
```

Traced operations include:
- `mlx_lm.generate` - Text generation with latency, token counts
- `mlx_embedding.embed` - Embedding generation with dimension, pooling method

---

## File Reference

| Component | File |
|-----------|------|
| BM25 Search | `src/pmd/store/search.py` |
| Vector Storage | `src/pmd/store/embeddings.py` |
| Vector Search | `src/pmd/store/vector_search.py` |
| Embedding Generation | `src/pmd/llm/embeddings.py` |
| LLM Base Interface | `src/pmd/llm/base.py` |
| MLX Provider | `src/pmd/llm/mlx_provider.py` |
| LM Studio Provider | `src/pmd/llm/lm_studio.py` |
| OpenRouter Provider | `src/pmd/llm/openrouter.py` |
| Query Expansion | `src/pmd/llm/query_expansion.py` |
| Reranking | `src/pmd/llm/reranker.py` |
| Hybrid Pipeline | `src/pmd/search/pipeline.py` |
| RRF Fusion | `src/pmd/search/fusion.py` |
| Score Normalization | `src/pmd/search/scoring.py` |
| Database Schema | `src/pmd/store/schema.py` |
| Configuration | `src/pmd/core/config.py` |
| Type Definitions | `src/pmd/core/types.py` |
| Instrumentation | `src/pmd/core/instrumentation.py` |
