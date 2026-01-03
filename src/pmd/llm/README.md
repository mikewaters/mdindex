# LLM Module Architecture

**Location:** `src/pmd/llm/`

LLM provider abstraction with multiple backend support.

## Files and Key Abstractions

### `base.py`

**`LLMProvider`** (ABC) - Provider contract

Abstract methods:
- `embed()` - Generate embeddings
- `generate()` - Text completion
- `rerank()` - Document relevance scoring
- `model_exists()` - Model availability check
- `is_available()` - Service health check
- `close()` - Resource cleanup

### `factory.py`

**`create_llm_provider()`** - Factory function
- Routes config to appropriate provider
- Validates platform for MLX (macOS only)

### `mlx_provider.py`

**`MLXProvider`** - Apple Silicon local inference

Features:
- Lazy model loading
- Query/document prefix support
- Multiple embedding extraction strategies
- HuggingFace authentication

**Invariants:**
- Raises RuntimeError on non-macOS platforms
- Supports model unloading for memory management

### `openrouter.py`

**`OpenRouterProvider`** - Cloud API provider

Features:
- HTTP-based via httpx AsyncClient
- Model enumeration endpoint
- Requires API key

### `lm_studio.py`

**`LMStudioProvider`** - Local OpenAI-compatible API

Features:
- Communicates with LM Studio server
- OpenAI-compatible endpoints

### `embeddings.py`

**`EmbeddingGenerator`** - Document embedding pipeline

Key methods:
- `embed_document()` - Chunk and embed document
- `embed_query()` - Embed search query
- `clear_embeddings_by_model()` - Cleanup

### `reranker.py`

**`DocumentReranker`** - LLM-based relevance scoring

Key methods:
- `get_rerank_scores()` - Raw LLM scores
- `rerank()` - With 60/40 blending

### `query_expansion.py`

**`QueryExpander`** - Query semantic variations

Key methods:
- `expand()` - Simple alternative phrasings
- `expand_with_semantics()` - Deeper semantic understanding
