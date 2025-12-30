# PMD Architecture

This document describes the technical architecture of PMD (Python Markdown Search), a hybrid search engine for markdown documents.

## Overview

PMD is designed as a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / MCP Server                      │
├─────────────────────────────────────────────────────────────┤
│                      Search Pipeline                         │
│         (Query Expansion → Search → RRF → Reranking)        │
├──────────────────────┬──────────────────────────────────────┤
│     LLM Layer        │           Store Layer                │
│  (Providers, etc.)   │    (Database, Repositories)          │
├──────────────────────┴──────────────────────────────────────┤
│                         Core Layer                           │
│              (Types, Config, Exceptions)                     │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/pmd/
├── __init__.py              # Package init, version export
├── __main__.py              # Entry point: python -m pmd
├── py.typed                 # PEP 561 marker
│
├── cli/                     # Command-line interface
│   ├── __init__.py
│   ├── main.py              # CLI entry point, argument parsing
│   └── commands/            # Command implementations
│       ├── collection.py    # Collection CRUD commands
│       ├── index.py         # Indexing commands
│       ├── search.py        # Search commands
│       └── status.py        # Status command
│
├── core/                    # Core business logic
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── exceptions.py        # Custom exception hierarchy
│   └── types.py             # Type definitions
│
├── store/                   # Data access layer
│   ├── __init__.py
│   ├── database.py          # SQLite connection management
│   ├── schema.py            # Schema definitions
│   ├── collections.py       # Collection CRUD operations
│   ├── documents.py         # Document storage
│   ├── embeddings.py        # Vector storage
│   ├── search.py            # FTS5 search
│   └── vector_search.py     # Vector search adapter
│
├── llm/                     # LLM abstraction layer
│   ├── __init__.py
│   ├── base.py              # Abstract LLM interface
│   ├── lm_studio.py         # LM Studio implementation
│   ├── openrouter.py        # OpenRouter implementation
│   ├── factory.py           # Provider factory
│   ├── embeddings.py        # Embedding generation
│   ├── query_expansion.py   # Query expansion
│   └── reranker.py          # Document reranking
│
├── search/                  # Search orchestration
│   ├── __init__.py
│   ├── pipeline.py          # Hybrid search pipeline
│   ├── fusion.py            # Reciprocal Rank Fusion
│   ├── scoring.py           # Score normalization
│   └── chunking.py          # Document chunking
│
├── formatters/              # Output formatting
│   └── __init__.py
│
├── mcp/                     # MCP server
│   ├── __init__.py
│   └── server.py            # MCP server setup
│
└── utils/                   # Shared utilities
    ├── __init__.py
    └── hashing.py           # SHA256 content hashing
```

## Core Components

### 1. Configuration System (`core/config.py`)

The configuration system uses dataclasses with environment variable support:

```python
@dataclass
class Config:
    db_path: Path                    # Database file location
    llm_provider: str                # Active LLM provider
    lm_studio: LMStudioConfig        # LM Studio settings
    openrouter: OpenRouterConfig     # OpenRouter settings
    search: SearchConfig             # Search parameters
    chunk: ChunkConfig               # Chunking parameters
```

**Key Classes:**
- `Config`: Main application configuration
- `LMStudioConfig`: Local LLM endpoint settings
- `OpenRouterConfig`: Cloud API settings
- `SearchConfig`: RRF parameters, weights, limits
- `ChunkConfig`: Document chunking settings

### 2. Type System (`core/types.py`)

Domain types as frozen dataclasses and TypedDicts:

```python
@dataclass
class Collection:
    id: int
    name: str
    pwd: str              # Base directory path
    glob_pattern: str
    created_at: str
    updated_at: str

@dataclass
class SearchResult(DocumentResult):
    score: float
    source: SearchSource  # FTS or VECTOR
    chunk_pos: Optional[int]
    snippet: Optional[str]

@dataclass
class RankedResult:
    file: str
    title: str
    score: float
    fts_score: Optional[float]
    vec_score: Optional[float]
    rerank_score: Optional[float]
```

### 3. Exception Hierarchy (`core/exceptions.py`)

```
PMDError (base)
├── DatabaseError
├── CollectionError
│   ├── CollectionNotFoundError
│   └── CollectionExistsError
├── DocumentError
│   └── DocumentNotFoundError
├── LLMError
│   ├── OllamaConnectionError
│   └── ModelNotFoundError
├── SearchError
├── EmbeddingError
├── FormatError
└── VirtualPathError
```

## Data Layer

### Database Schema

```sql
-- Content-addressable storage
CREATE TABLE content (
    hash TEXT PRIMARY KEY,      -- SHA256 of content
    doc TEXT NOT NULL,          -- Full document text
    created_at TEXT NOT NULL
);

-- Collections (indexed directories)
CREATE TABLE collections (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,           -- Collection identifier
    pwd TEXT NOT NULL,          -- Filesystem path
    glob_pattern TEXT,          -- File pattern
    created_at TEXT,
    updated_at TEXT
);

-- Documents (file-to-content mappings)
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    collection_id INTEGER REFERENCES collections(id),
    path TEXT NOT NULL,         -- Relative path
    title TEXT NOT NULL,
    hash TEXT REFERENCES content(hash),
    active INTEGER DEFAULT 1,   -- Soft delete flag
    modified_at TEXT,
    UNIQUE(collection_id, path)
);

-- Full-text search index
CREATE VIRTUAL TABLE documents_fts USING fts5(
    path, body,
    content='',
    tokenize='porter unicode61'
);

-- Vector embeddings metadata
CREATE TABLE content_vectors (
    hash TEXT,
    seq INTEGER,                -- Chunk sequence
    pos INTEGER,                -- Character position
    model TEXT,
    embedded_at TEXT,
    PRIMARY KEY (hash, seq)
);
```

### Content-Addressable Storage

Documents are stored using content-addressable storage:

1. **Content Table**: Stores actual text, keyed by SHA256 hash
2. **Documents Table**: Maps (collection, path) → content hash
3. **Benefits**:
   - Automatic deduplication of identical files
   - Efficient change detection via hash comparison
   - Clean garbage collection of orphaned content

```
documents table:
  └── (collection_id, path) → hash reference

content table:
  └── hash (SHA256) → actual content
```

### Repository Pattern

Each domain entity has a repository:

```python
class CollectionRepository:
    def list_all(self) -> list[Collection]
    def get_by_name(self, name: str) -> Collection | None
    def create(self, name, pwd, glob_pattern) -> Collection
    def remove(self, collection_id: int) -> tuple[int, int]
    def rename(self, collection_id: int, new_name: str) -> None

class DocumentRepository:
    def add_or_update(self, collection_id, path, title, content) -> tuple[DocumentResult, bool]
    def get(self, collection_id, path) -> DocumentResult | None
    def get_by_hash(self, hash_value) -> str | None
    def delete(self, collection_id, path) -> bool
    def check_if_modified(self, collection_id, path, new_hash) -> bool
```

## Search Architecture

### Hybrid Search Pipeline

The search pipeline combines multiple retrieval strategies:

```
User Query
    │
    ├─→ Query Expansion (LLM)
    │   • Original query weighted ×2
    │   • Generate 2 semantic variations
    │
    ├─→ Parallel Search (for each query variant)
    │   ├─→ FTS5 BM25 Search
    │   │   • tokenize='porter unicode61'
    │   │   • Score normalized: abs(score) / max_score
    │   │
    │   └─→ Vector Search (sqlite-vec)
    │       • Cosine distance → 1 / (1 + distance)
    │       • Top-K with best distance per document
    │
    ├─→ Reciprocal Rank Fusion (RRF)
    │   • k = 60 (smoothing constant)
    │   • Score = Σ(weight / (k + rank + 1))
    │   • Top-rank bonus: +0.05 for #1, +0.02 for #2-3
    │   • Select top 30 candidates
    │
    ├─→ LLM Re-ranking
    │   • Yes/No classification with confidence
    │   • Yes: score = 0.5 + 0.5 × confidence
    │   • No:  score = 0.5 × (1 - confidence)
    │
    └─→ Position-Aware Blending
        • Rank 1-3:   75% RRF + 25% reranker
        • Rank 4-10:  60% RRF + 40% reranker
        • Rank 11+:   40% RRF + 60% reranker
```

### Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from different retrieval methods:

```python
def reciprocal_rank_fusion(result_lists, k=60):
    scores = defaultdict(float)

    for list_idx, results in enumerate(result_lists):
        weight = 2.0 if list_idx < 2 else 1.0  # Original query gets 2x

        for rank, result in enumerate(results):
            # RRF formula
            rrf_score = weight / (k + rank + 1)

            # Top-rank bonuses
            if rank == 0:
                rrf_score += 0.05
            elif rank <= 2:
                rrf_score += 0.02

            scores[result.filepath] += rrf_score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Document Chunking

Documents are chunked for embedding:

```python
CHUNK_MAX_BYTES = 6 * 1024  # ~2000 tokens

Algorithm:
1. If len(content) <= max_bytes: return single chunk
2. Split preferring: \n\n → sentence end → \n → space
3. Each chunk stored with:
   - hash: SHA256(full_document)
   - seq: 0, 1, 2, ...
   - pos: character offset in original
```

## LLM Integration

### Provider Abstraction

```python
class LLMProvider(ABC):
    @abstractmethod
    async def embed(self, text, model, is_query) -> EmbeddingResult | None

    @abstractmethod
    async def generate(self, prompt, model, max_tokens, temperature) -> str | None

    @abstractmethod
    async def rerank(self, query, documents, model) -> RerankResult

    @abstractmethod
    async def is_available(self) -> bool
```

### LM Studio Provider

- **API**: OpenAI-compatible (`/v1/embeddings`, `/v1/chat/completions`)
- **Endpoint**: `http://localhost:1234` (default)
- **Authentication**: None (local server)
- **Models**: User-loaded models in LM Studio

### OpenRouter Provider

- **API**: OpenAI-compatible with custom headers
- **Endpoint**: `https://openrouter.ai/api/v1`
- **Authentication**: Bearer token (`OPENROUTER_API_KEY`)
- **Models**: Access to 100+ models

### Provider Factory

```python
def create_llm_provider(config: Config) -> LLMProvider:
    if config.llm_provider == "lm-studio":
        return LMStudioProvider(config.lm_studio)
    elif config.llm_provider == "openrouter":
        return OpenRouterProvider(config.openrouter)
```

### Query Expansion

```python
class QueryExpander:
    async def expand(self, query: str, num_variations: int = 2) -> list[str]:
        prompt = f"""Generate {num_variations} alternative phrasings...
        Original Query: {query}
        Alternative Phrasings:"""

        response = await self.llm.generate(prompt)
        return [query] + self._parse_variations(response)
```

### Document Reranking

```python
class DocumentReranker:
    async def rerank(self, query: str, candidates: list[RankedResult]) -> list[RankedResult]:
        for doc in candidates:
            # Ask LLM: "Is this document relevant to the query?"
            result = await self.llm.rerank(query, [doc])

            # Blend scores: 60% RRF + 40% reranker
            doc.score = 0.6 * doc.score + 0.4 * result.score

        return sorted(candidates, key=lambda x: x.score, reverse=True)
```

## MCP Server

The MCP server exposes PMD functionality to AI agents:

```python
class PMDMCPServer:
    async def search(self, query, limit, collection) -> dict:
        """Full hybrid search with LLM enhancement."""

    async def get_document(self, collection, path) -> dict:
        """Retrieve document content."""

    async def list_collections(self) -> dict:
        """List all indexed collections."""

    async def get_status(self) -> dict:
        """Server health and statistics."""
```

## Data Flow

### Indexing Flow

```
Filesystem                    Database
    │                            │
    ├─ Read markdown file        │
    │                            │
    ├─ Extract title (H1 or     │
    │  filename)                 │
    │                            │
    ├─ Compute SHA256 hash ─────→├─ Store in content table
    │                            │  (deduplicated)
    │                            │
    ├─ Create/update document ──→├─ Store in documents table
    │  record                    │
    │                            │
    └─ Index in FTS5 ───────────→└─ Insert in documents_fts
```

### Search Flow

```
Query                         Results
  │                              ▲
  ├─→ Query Expansion (LLM)      │
  │                              │
  ├─→ FTS5 Search ───────────────┤
  │                              │
  ├─→ Vector Search ─────────────┤
  │   (embeddings)               │
  │                              │
  ├─→ RRF Fusion ────────────────┤
  │                              │
  └─→ LLM Reranking ─────────────┘
```

## Performance Considerations

### Database Indexes

```sql
CREATE INDEX idx_documents_collection ON documents(collection_id);
CREATE INDEX idx_documents_hash ON documents(hash);
CREATE INDEX idx_content_vectors_hash ON content_vectors(hash);
CREATE INDEX idx_path_contexts_collection ON path_contexts(collection_id);
```

### Async I/O

- LLM calls are async (`httpx.AsyncClient`)
- Database operations are synchronous (SQLite limitation)
- Search pipeline is async to allow concurrent LLM calls

### Caching

- **Content**: Deduplicated via content-addressable storage
- **Embeddings**: Stored in database, regenerated only when content changes
- **LLM Responses**: `ollama_cache` table for response caching (future)

## Extensibility

### Adding a New LLM Provider

1. Create `src/pmd/llm/new_provider.py`
2. Implement `LLMProvider` abstract class
3. Add configuration in `core/config.py`
4. Register in `llm/factory.py`

### Adding a New Output Format

1. Create `src/pmd/formatters/new_format.py`
2. Implement `Formatter` abstract class
3. Register in CLI command handlers

### Adding a New Search Strategy

1. Extend `SearchRepository` in `store/search.py` (or add an adapter like `store/vector_search.py`)
2. Add fusion logic in `search/fusion.py`
3. Integrate in `search/pipeline.py`

## Security Considerations

- API keys stored in environment variables (not in code)
- Database file permissions follow system defaults
- No network exposure by default (CLI tool)
- MCP server runs locally (Claude integration)

## Testing Strategy

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for pipelines
├── e2e/            # End-to-end workflow tests
└── fixtures/       # Test data files
    └── markdown/   # Sample markdown documents
```

Key test areas:
- Repository CRUD operations
- Search ranking accuracy
- RRF fusion correctness
- LLM provider mocking
- CLI command behavior
