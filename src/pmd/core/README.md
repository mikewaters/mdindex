# Core Module Architecture

**Location:** `src/pmd/core/`

The foundation layer providing types, configuration, and cross-cutting concerns.

## Files and Key Abstractions

### `__init__.py`
Re-exports all public symbols from the core module (45 exports total).

### `config.py`

**`Config`** - Top-level application configuration
- Composite of all sub-configurations
- Database path with XDG_CACHE_HOME awareness
- Factory methods: `from_env()`, `from_file()`, `from_env_or_file()`

**`MLXConfig`** - Apple Silicon local inference configuration
- Model paths, embedding dimensions, prefixes
- Lazy loading control, temperature settings

**`OpenRouterConfig`** - Cloud API configuration
- API key, base URL, model selections

**`LMStudioConfig`** - Local OpenAI-compatible API configuration
- Base URL, model selections

**`SearchConfig`** - Hybrid search tuning
- FTS/vector weights, RRF k parameter, rerank candidates

**`ChunkConfig`** - Document chunking parameters
- max_bytes (6KB default), min_chunk_size

**`TracingConfig`** - OpenTelemetry/Phoenix observability
- Endpoint, sampling, batch export settings

**Invariants:**
- Environment variables override file configuration
- Unknown config keys are ignored (forward compatibility)
- Default database path: `~/.cache/pmd/index.db`

### `types.py`

**`VirtualPath`** (frozen dataclass) - pmd:// URI value object
- Immutable, hashable, equality by value
- Encapsulates collection_name + path

**`Collection`** - Document collection with multi-source support
- Supports: filesystem, HTTP, entity sources
- Contains source configuration and metadata

**`DocumentResult`** - Retrieved document representation
- Core fields: filepath, title, hash, collection_id

**`SearchResult`** (extends DocumentResult) - Search result with scoring
- Adds: score, source (FTS/VECTOR/HYBRID), snippet

**`RankedResult`** - Post-fusion result with full provenance
- Tracks: fts_score, vec_score, rerank_score
- Tracks: fts_rank, vec_rank, sources_count, blend_weight

**`EmbeddingResult`** - Embedding vector with model metadata

**`RerankResult`** - Batch reranking results container

**`IndexStatus`** - System health metrics

### `exceptions.py`

Exception hierarchy for domain-specific error handling:

```
PMDError (root)
├── DatabaseError
├── CollectionError
│   ├── CollectionNotFoundError
│   └── CollectionExistsError
├── DocumentError
│   └── DocumentNotFoundError (with suggestions)
├── LLMError
│   └── ModelNotFoundError
├── SearchError
├── EmbeddingError
├── FormatError
└── VirtualPathError
```

### `instrumentation.py`

OpenTelemetry/Phoenix tracing integration:

**`configure_phoenix_tracing()`** - Bootstrap OTLP provider
- Configurable sampler (ALWAYS_ON or ratio-based)
- Batch vs simple span processor

**`traced_mlx_generate()`** - Context manager for generation tracing
- Records: model_id, prompt, tokens, latency

**`traced_mlx_embed()`** - Context manager for embedding tracing
- Records: model_id, input_length, dimension, pooling

**`traced_request()`** - Parent span for high-level operations
