# High-Level System Architecture

PMD (Python Markdown Search) is a hybrid search engine for markdown documents combining full-text search, vector semantic search, and LLM-based re-ranking. This document provides comprehensive architectural documentation for the system.

---

## Table of Contents

1. [System Layering](#1-system-layering)
2. [System Components](#2-system-components)
3. [Module Overview](#3-module-overview)
4. [Software Patterns](#4-software-patterns)
5. [Module-Level Architecture](#5-module-level-architecture)
   - [Core Module](#51-core-module)
   - [Store Module](#52-store-module)
   - [Services Module](#53-services-module)
   - [CLI Module](#54-cli-module)
   - [MCP Module](#55-mcp-module)
   - [LLM Module](#56-llm-module)
   - [Search Module](#57-search-module)
   - [Sources Module](#58-sources-module)
   - [Utils Module](#59-utils-module)
   - [Formatters Module](#510-formatters-module)

---

## 1. System Layering

PMD follows a classic layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│         CLI Interface (cli/)    MCP Server (mcp/)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     SERVICE LAYER                            │
│    ServiceContainer → IndexingService, SearchService,        │
│                       StatusService                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    REPOSITORY LAYER                          │
│  DocumentRepository, CollectionRepository, EmbeddingRepository│
│  FTS5SearchRepository, VectorSearchRepository,               │
│  SourceMetadataRepository                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      DATA LAYER                              │
│            SQLite Database + FTS5 + sqlite-vec               │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | Key Components |
|-------|---------------|----------------|
| **Presentation** | User interface, command parsing, output formatting | CLI commands, MCP server |
| **Service** | Business logic orchestration, transaction coordination | IndexingService, SearchService, StatusService |
| **Repository** | Data access abstraction, query execution | All `*Repository` classes |
| **Data** | Persistence, indexing, vector storage | SQLite, FTS5, sqlite-vec |

### Cross-Cutting Concerns

Several components span multiple layers:
- **LLM Module**: Provides embeddings, reranking, and query expansion to services
- **Search Module**: Provides pipeline orchestration consumed by services
- **Sources Module**: Document ingestion abstraction used by indexing
- **Core Module**: Types, config, and exceptions used everywhere

---

## 2. System Components

### Component Interaction Diagram

```
┌─────────────┐     ┌─────────────┐
│   CLI       │     │  MCP Server │
│  (argparse) │     │  (Protocol) │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
       ┌─────────▼─────────┐
       │  ServiceContainer  │
       │  (DI Container)    │
       └─────────┬─────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐   ┌────▼────┐   ┌───▼───┐
│Indexing│   │ Search  │   │Status │
│Service │   │ Service │   │Service│
└───┬───┘   └────┬────┘   └───┬───┘
    │            │            │
    │     ┌──────▼──────┐     │
    │     │   Hybrid    │     │
    │     │  Pipeline   │     │
    │     └──────┬──────┘     │
    │            │            │
┌───▼────────────▼────────────▼───┐
│         REPOSITORIES             │
│  Document, Collection, Embedding │
│  FTS5Search, SourceMetadata      │
└───────────────┬─────────────────┘
                │
       ┌────────▼────────┐
       │    Database     │
       │    (SQLite)     │
       │  FTS5 + sqlite-vec │
       └─────────────────┘

External Dependencies:
┌─────────────────────────────────┐
│         LLM Providers           │
│  MLX │ OpenRouter │ LM Studio   │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│       Document Sources          │
│  Filesystem │ HTTP │ Entity     │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│        Observability            │
│  OpenTelemetry + Phoenix        │
└─────────────────────────────────┘
```

### External Integration Points

| Component | Purpose | Protocol |
|-----------|---------|----------|
| **LLM Providers** | Embeddings, generation, reranking | MLX (local), HTTP API |
| **Document Sources** | Document ingestion | File I/O, HTTP |
| **OpenTelemetry** | Distributed tracing | OTLP/gRPC |
| **Phoenix** | LLM observability | OTLP HTTP |

---

## 3. Module Overview

The `pmd` package (`src/pmd/`) is organized into the following modules:

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `core` | Configuration, types, exceptions, instrumentation | `Config`, `Collection`, `SearchResult`, `RankedResult` |
| `store` | Database access, repositories | `DocumentRepository`, `EmbeddingRepository`, `CollectionRepository` |
| `services` | Business logic orchestration | `ServiceContainer`, `IndexingService`, `SearchService` |
| `cli` | Command-line interface | `main()`, command handlers |
| `mcp` | Model Context Protocol server | `PMDMCPServer` |
| `llm` | LLM provider abstraction | `LLMProvider`, `EmbeddingGenerator`, `DocumentReranker` |
| `search` | Search pipeline and algorithms | `HybridSearchPipeline`, `reciprocal_rank_fusion` |
| `sources` | Document source abstraction | `DocumentSource`, `SourceRegistry` |
| `utils` | Utility functions | `sha256_hash` |
| `formatters` | Output formatting | (placeholder) |

### Module Dependency Graph

```
cli, mcp
    ↓
services (ServiceContainer)
    ↓
├── llm (LLMProvider, EmbeddingGenerator)
├── search (HybridSearchPipeline)
├── sources (DocumentSource)
└── store (Repositories)
        ↓
    core (Config, Types, Exceptions)
        ↓
    utils (Hashing)
```

---

## 4. Software Patterns

### 4.1 Repository Pattern

All data access is encapsulated in Repository classes that abstract SQL operations:

```python
class DocumentRepository:
    def add_or_update(self, ...) -> tuple[DocumentResult, bool]: ...
    def get(self, collection_id, path) -> DocumentResult | None: ...
    def delete(self, collection_id, path) -> bool: ...
```

**Benefits:**
- Centralized SQL in one location per entity
- Easy to mock for testing
- Clean separation from business logic

### 4.2 Dependency Injection Container

`ServiceContainer` manages all service and repository lifecycles:

```python
async with ServiceContainer(config) as services:
    await services.indexing.index_collection("docs")
    results = await services.search.hybrid_search("query")
```

**Characteristics:**
- Lazy initialization of services and LLM components
- Resource cleanup via async context manager
- Single point of dependency resolution

### 4.3 Factory Pattern

LLM provider creation uses a factory function:

```python
def create_llm_provider(config: Config) -> LLMProvider:
    if config.llm_provider == "mlx":
        return MLXProvider(config.mlx)
    elif config.llm_provider == "openrouter":
        return OpenRouterProvider(config.openrouter)
    # ...
```

### 4.4 Strategy Pattern

Multiple implementations share common interfaces:

- **LLM Providers**: MLX, OpenRouter, LM Studio implement `LLMProvider`
- **Document Sources**: Filesystem, HTTP, Entity implement `DocumentSource`
- **Search Repositories**: FTS5, Vector implement `SearchRepository[QueryT]`

### 4.5 Protocol-Based Abstraction

Document sources use Python `Protocol` for duck typing:

```python
@runtime_checkable
class DocumentSource(Protocol):
    def list_documents(self) -> Iterator[DocumentReference]: ...
    async def fetch_content(self, ref: DocumentReference) -> FetchResult: ...
```

### 4.6 Content-Addressable Storage

Documents reference content by SHA256 hash:

```python
content_hash = sha256_hash(document_body)
# Same content = same hash = deduplication across documents
```

### 4.7 Pipeline Pattern

Search uses a multi-stage pipeline:

```
Query → Expansion → Parallel Search → RRF Fusion → Reranking → Blending → Results
```

### 4.8 Registry Pattern

Sources are registered by URI scheme:

```python
registry = SourceRegistry()
registry.register("file", FileSystemSource)
registry.register("http", HTTPSource)
source = registry.resolve("file:///path/to/docs")
```

---

## 5. Module-Level Architecture

### 5.1 Core Module

**Location:** `src/pmd/core/`

The foundation layer providing types, configuration, and cross-cutting concerns.

#### Files and Key Abstractions

##### `__init__.py`
Re-exports all public symbols from the core module (45 exports total).

##### `config.py`

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

##### `types.py`

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

##### `exceptions.py`

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

##### `instrumentation.py`

OpenTelemetry/Phoenix tracing integration:

**`configure_phoenix_tracing()`** - Bootstrap OTLP provider
- Configurable sampler (ALWAYS_ON or ratio-based)
- Batch vs simple span processor

**`traced_mlx_generate()`** - Context manager for generation tracing
- Records: model_id, prompt, tokens, latency

**`traced_mlx_embed()`** - Context manager for embedding tracing
- Records: model_id, input_length, dimension, pooling

**`traced_request()`** - Parent span for high-level operations

---

### 5.2 Store Module

**Location:** `src/pmd/store/`

Data access layer implementing the Repository pattern with SQLite.

#### Files and Key Abstractions

##### `database.py`

**`Database`** - Connection manager
- Lazy connection initialization
- FTS5 and sqlite-vec extension loading
- Transaction support via context manager
- Graceful degradation if sqlite-vec unavailable

**Invariants:**
- Foreign keys enabled (PRAGMA foreign_keys = ON)
- Row factory set to sqlite3.Row
- `vec_available` property indicates vector search capability

##### `schema.py`

Database schema definitions:
- `SCHEMA_VERSION = 1`
- `EMBEDDING_DIMENSION = 768`
- `SCHEMA_SQL` - Relational tables
- `VECTOR_TABLE_SQL` - sqlite-vec virtual table

**Tables:**
- `content` - Content-addressable storage (hash-based)
- `collections` - Named document collections
- `documents` - File-to-content mappings
- `documents_fts` - FTS5 virtual table
- `content_vectors` - Embedding metadata
- `content_vectors_vec` - Vector storage (sqlite-vec)
- `path_contexts` - Directory-level context
- `source_metadata` - Remote document metadata

##### `documents.py`

**`DocumentRepository`** - Document CRUD with CAS

Key methods:
- `add_or_update()` - Insert/update with content deduplication
- `get()` - Retrieve by collection_id + path
- `delete()` - Soft-delete (mark inactive)
- `check_if_modified()` - Hash comparison for change detection

**Invariants:**
- Documents reference content by SHA256 hash
- Soft deletes via `active` flag (no hard deletion)
- Transactions ensure atomicity of content + document

##### `embeddings.py`

**`EmbeddingRepository`** - Vector storage and similarity search

Key methods:
- `store_embedding()` - Store vector + metadata
- `search_vectors()` - L2-based similarity search
- `has_embeddings()` - Check embedding existence

**Invariants:**
- Separates metadata (content_vectors) from vectors (content_vectors_vec)
- Score = 1 / (1 + distance) for L2 conversion
- Deduplicates by document path (best chunk per doc)

##### `search.py`

**`SearchRepository[QueryT]`** (ABC, Generic) - Abstract search interface
- Type parameter allows different query types

**`FTS5SearchRepository`** - BM25 full-text search
- Porter stemming + Unicode61 tokenization
- Score normalization to [0, 1]

**Invariants:**
- FTS5 returns negative BM25 scores (converted to positive)
- Max-normalization applied to result set

##### `vector_search.py`

**`VectorSearchRepository`** - Adapter wrapping EmbeddingRepository
- Implements `SearchRepository[list[float]]`
- Enables polymorphic treatment with FTS5

##### `collections.py`

**`CollectionRepository`** - Collection management

Key methods:
- `create()` - Create with validation
- `remove()` - Delete with cascading cleanup
- `rename()` - Rename with uniqueness check

**Invariants:**
- Cascading deletion removes documents, orphaned content, embeddings
- Collection names are unique

##### `source_metadata.py`

**`SourceMetadataRepository`** - Remote document tracking

Key methods:
- `upsert()` - Insert or update metadata
- `needs_refresh()` - Check if document exceeds max_age
- `get_stale_documents()` - Find documents needing refresh

---

### 5.3 Services Module

**Location:** `src/pmd/services/`

Business logic orchestration layer coordinating repositories and LLM components.

#### Files and Key Abstractions

##### `container.py`

**`ServiceContainer`** - Dependency injection container

Responsibilities:
- Repository lifecycle management
- LLM component lazy initialization
- Service accessor properties
- Resource cleanup via async context manager

Key properties:
- `collection_repo`, `document_repo`, `embedding_repo`, `fts_repo`
- `indexing`, `search`, `status` (service accessors)

Key methods:
- `get_llm_provider()` - Async factory for LLM
- `get_embedding_generator()` - Async factory for embeddings
- `vec_available` - Check sqlite-vec availability
- `is_llm_available()` - Async LLM health check

##### `indexing.py`

**`IndexingService`** - Document indexing and embedding

**`IndexResult`** - Indexing operation result
- indexed, skipped, errors fields

**`EmbedResult`** - Embedding operation result
- embedded, skipped, chunks_total fields

**`CleanupResult`** - Cleanup operation result
- orphaned_content, orphaned_embeddings fields

Key methods:
- `index_collection()` - Enumerate, fetch, store, index documents
- `embed_collection()` - Generate embeddings for indexed docs
- `update_all_collections()` - Batch update all collections
- `cleanup_orphans()` - Remove orphaned content and embeddings

**Invariants:**
- Indexing continues on individual document errors
- Source metadata (ETags) enables incremental updates
- Content hash comparison for change detection

##### `search.py`

**`SearchService`** - Search operation orchestration

Key methods:
- `fts_search()` - Synchronous BM25 search
- `vector_search()` - Async semantic search
- `hybrid_search()` - Combined FTS + vector with optional LLM

**Invariants:**
- Vector search requires sqlite-vec and LLM provider
- Hybrid search delegates to HybridSearchPipeline

##### `status.py`

**`StatusService`** - Index health monitoring

Key methods:
- `get_index_status()` - Quick status summary
- `get_full_status()` - Comprehensive async status
- `get_collection_stats()` - Per-collection statistics
- `get_index_sync_report()` - FTS/vector sync status

---

### 5.4 CLI Module

**Location:** `src/pmd/cli/`

Command-line interface using argparse with hierarchical commands.

#### Files and Key Abstractions

##### `main.py`

**`create_parser()`** - Build argparse parser with subcommands
**`main()`** - Entry point with logging, tracing, command routing

Global arguments:
- `-v/--version` - Version display
- `-L/--log-level` - Logging level
- `-c/--config` - Config file path
- `--phoenix-tracing` - Enable tracing

##### `commands/__init__.py`

Command handler exports:
- `handle_collection`, `handle_search`, `handle_vsearch`
- `handle_query`, `handle_index_collection`, `handle_embed`
- `handle_update_all`, `handle_cleanup`, `handle_status`

##### `commands/collection.py`

Collection management sub-commands:
- `collection add` - Create from filesystem, HTTP, or entity
- `collection list` - List all collections
- `collection show` - Show collection details
- `collection remove` - Delete collection
- `collection rename` - Rename collection

##### `commands/search.py`

Three search variants:
- `search` - FTS5 BM25 search
- `vsearch` - Vector semantic search
- `query` - Hybrid search with LLM enhancement

##### `commands/index.py`

Indexing commands:
- `index <collection>` - Index documents
- `embed <collection>` - Generate embeddings
- `update-all` - Update all collections
- `cleanup` - Remove orphaned data

##### `commands/status.py`

Status reporting:
- `status` - Index status summary
- `status --check-sync` - FTS/vector sync report

---

### 5.5 MCP Module

**Location:** `src/pmd/mcp/`

Model Context Protocol server for LLM integration.

#### Files and Key Abstractions

##### `server.py`

**`PMDMCPServer`** - MCP protocol implementation

Lifecycle:
- `initialize()` - Connect to database
- `shutdown()` - Release resources

API methods (all async, return dicts):
- `search(query, limit, collection)` - Hybrid search
- `get_document(collection, path)` - Retrieve document
- `list_collections()` - Enumerate collections
- `get_status()` - Index status
- `index_collection(name, force, embed)` - Index collection
- `embed_collection(name, force)` - Generate embeddings

**Invariants:**
- All methods return structured dicts
- LLM features auto-enabled if provider available

---

### 5.6 LLM Module

**Location:** `src/pmd/llm/`

LLM provider abstraction with multiple backend support.

#### Files and Key Abstractions

##### `base.py`

**`LLMProvider`** (ABC) - Provider contract

Abstract methods:
- `embed()` - Generate embeddings
- `generate()` - Text completion
- `rerank()` - Document relevance scoring
- `model_exists()` - Model availability check
- `is_available()` - Service health check
- `close()` - Resource cleanup

##### `factory.py`

**`create_llm_provider()`** - Factory function
- Routes config to appropriate provider
- Validates platform for MLX (macOS only)

##### `mlx_provider.py`

**`MLXProvider`** - Apple Silicon local inference

Features:
- Lazy model loading
- Query/document prefix support
- Multiple embedding extraction strategies
- HuggingFace authentication

**Invariants:**
- Raises RuntimeError on non-macOS platforms
- Supports model unloading for memory management

##### `openrouter.py`

**`OpenRouterProvider`** - Cloud API provider

Features:
- HTTP-based via httpx AsyncClient
- Model enumeration endpoint
- Requires API key

##### `lm_studio.py`

**`LMStudioProvider`** - Local OpenAI-compatible API

Features:
- Communicates with LM Studio server
- OpenAI-compatible endpoints

##### `embeddings.py`

**`EmbeddingGenerator`** - Document embedding pipeline

Key methods:
- `embed_document()` - Chunk and embed document
- `embed_query()` - Embed search query
- `clear_embeddings_by_model()` - Cleanup

##### `reranker.py`

**`DocumentReranker`** - LLM-based relevance scoring

Key methods:
- `get_rerank_scores()` - Raw LLM scores
- `rerank()` - With 60/40 blending

##### `query_expansion.py`

**`QueryExpander`** - Query semantic variations

Key methods:
- `expand()` - Simple alternative phrasings
- `expand_with_semantics()` - Deeper semantic understanding

---

### 5.7 Search Module

**Location:** `src/pmd/search/`

Search pipeline algorithms and orchestration.

#### Files and Key Abstractions

##### `pipeline.py`

**`SearchPipelineConfig`** - Pipeline parameters
- Weights, RRF k, rerank candidates
- Feature flags for expansion/reranking

**`HybridSearchPipeline`** - Multi-stage search orchestration

Pipeline stages:
1. Query expansion (optional)
2. Parallel FTS + vector search
3. Reciprocal Rank Fusion
4. LLM reranking (optional)
5. Position-aware blending
6. Score normalization

##### `fusion.py`

**`reciprocal_rank_fusion()`** - Combine ranked lists

Formula: `RRF = Σ(weight / (k + rank + 1)) + bonuses`

Features:
- Top-rank bonuses (+0.05 for rank 1, +0.02 for ranks 2-3)
- Provenance tracking (fts_rank, vec_rank, sources_count)
- Weighted result lists

##### `scoring.py`

**`normalize_scores()`** - Max-normalization to [0, 1]

**`blend_scores()`** - Position-aware blending

| Rank | RRF Weight | Reranker Weight |
|------|------------|-----------------|
| 1-3 | 75% | 25% |
| 4-10 | 60% | 40% |
| 11+ | 40% | 60% |

##### `chunking.py`

**`chunk_document()`** - Split documents for embedding

Splitting preference:
1. Paragraph breaks (`\n\n`)
2. Sentence ends
3. Line breaks (`\n`)
4. Spaces

##### `text.py`

**`is_indexable()`** - Document quality filter
- Rejects empty documents
- Rejects title-only documents

---

### 5.8 Sources Module

**Location:** `src/pmd/sources/`

Document source abstraction for multi-source ingestion.

#### Files and Key Abstractions

##### `base.py`

**`DocumentSource`** (Protocol) - Source contract

Methods:
- `list_documents()` - Enumerate documents
- `fetch_content()` - Fetch document content
- `capabilities()` - Describe source features
- `check_modified()` - Change detection

**`DocumentReference`** - Document metadata
- uri, path, title, metadata

**`FetchResult`** - Fetch operation result
- content, content_type, encoding, metadata

**`SourceCapabilities`** - Feature flags
- supports_incremental, supports_etag, etc.

##### `registry.py`

**`SourceRegistry`** - URI scheme routing

Key methods:
- `register(scheme, factory)` - Register source type
- `resolve(uri)` - Create source for URI

Built-in schemes: `file`, `http`, `https`, `entity`

##### `filesystem.py`

**`FileSystemSource`** - Local filesystem source

Features:
- Glob pattern matching
- Nanosecond mtime comparison
- Content type detection

##### `http.py`

**`HTTPSource`** - HTTP/HTTPS source

Features:
- Sitemap parsing
- ETag/Last-Modified support
- Retry with exponential backoff
- HTML to text conversion

##### `entity.py`

**`EntitySource`** - Pluggable custom backends

URI format: `entity://<resolver>/<resource-type>[/<id>]`

**`EntityResolver`** (Protocol) - Custom backend contract
**`EntityResolverRegistry`** - Resolver management

##### `auth.py`

**`AuthConfig`** - Authentication configuration
- Types: none, bearer, basic, api_key, custom

**`CredentialResolver`** - Credential reference resolution

Reference formats:
- `$ENV:VAR_NAME` - Environment variable
- `$KEYRING:key_name` - System keyring
- `$STATIC:key_name` - Static store (testing)

---

### 5.9 Utils Module

**Location:** `src/pmd/utils/`

Utility functions used across the application.

#### Files and Key Abstractions

##### `hashing.py`

**`sha256_hash(content: str)`** - SHA256 hash of text
**`sha256_hash_bytes(content: bytes)`** - SHA256 hash of bytes

Used for content-addressable storage in documents.

---

### 5.10 Formatters Module

**Location:** `src/pmd/formatters/`

Output formatting utilities (placeholder for future expansion).

---

## Entry Points

### CLI Entry Point

```
python -m pmd [command] [options]
```

Implemented in `src/pmd/__main__.py`, delegates to `cli.main.main()`.

### MCP Entry Point

```python
from pmd.mcp import PMDMCPServer
from pmd.core.config import Config

server = PMDMCPServer(Config.from_env())
await server.initialize()
# Use server methods
await server.shutdown()
```

---

## Configuration Flow

```
Environment Variables
        ↓
    Override
        ↓
TOML Config File
        ↓
    Override
        ↓
Dataclass Defaults
```

Key environment variables:
- `PMD_CONFIG` - Config file path
- `LLM_PROVIDER` - Provider selection (mlx/openrouter/lm-studio)
- `OPENROUTER_API_KEY` - OpenRouter API key
- `HF_TOKEN` - HuggingFace token for MLX

---

## Data Flow Examples

### Document Indexing

```
CLI: pmd index my-docs
    ↓
IndexingService.index_collection("my-docs")
    ↓
CollectionRepository.get_by_name("my-docs") → Collection
    ↓
SourceRegistry.resolve(collection.source_uri) → DocumentSource
    ↓
DocumentSource.list_documents() → [DocumentReference, ...]
    ↓
For each reference:
├── DocumentSource.fetch_content(ref) → FetchResult
├── sha256_hash(content) → hash
├── DocumentRepository.add_or_update(path, hash, content)
├── FTS5SearchRepository.index_document(doc_id, path, body)
└── SourceMetadataRepository.upsert(doc_id, metadata)
    ↓
Return IndexResult(indexed=N, skipped=M, errors=[])
```

### Hybrid Search

```
CLI: pmd query "machine learning" --limit 10
    ↓
SearchService.hybrid_search("machine learning", limit=10)
    ↓
HybridSearchPipeline.search()
├── QueryExpander.expand() → ["machine learning", "ML algorithms", ...]
├── FTS5SearchRepository.search() → [SearchResult, ...]
├── EmbeddingGenerator.embed_query() → vector
├── EmbeddingRepository.search_vectors(vector) → [SearchResult, ...]
├── reciprocal_rank_fusion() → [RankedResult, ...]
├── DocumentReranker.get_rerank_scores() → [RerankDocumentResult, ...]
├── blend_scores() → [RankedResult, ...]
└── normalize_scores() → [RankedResult, ...]
    ↓
Return ranked results with full provenance
```
