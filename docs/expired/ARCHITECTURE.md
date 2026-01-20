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
   - [LLM Module](#55-llm-module)
   - [Search Module](#56-search-module)
   - [Sources Module](#57-sources-module)
   - [Utils Module](#58-utils-module)
   - [Formatters Module](#59-formatters-module)

---

## 1. System Layering

PMD follows a classic layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│                    CLI Interface (cli/)                      │
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
| **Presentation** | User interface, command parsing, output formatting | CLI commands |
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
┌─────────────┐
│   CLI       │
│  (argparse) │
└──────┬──────┘
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
| `llm` | LLM provider abstraction | `LLMProvider`, `EmbeddingGenerator`, `DocumentReranker` |
| `search` | Search pipeline and algorithms | `HybridSearchPipeline`, `reciprocal_rank_fusion` |
| `sources` | Document source abstraction | `DocumentSource`, `SourceRegistry` |
| `utils` | Utility functions | `sha256_hash` |
| `formatters` | Output formatting | (placeholder) |

### Module Dependency Graph

```
cli
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
    collection = services.collection_repo.get_by_name("docs")
    source = FileSystemSource(
        SourceConfig(
            uri=collection.get_source_uri(),
            extra=collection.get_source_config_dict(),
        )
    )
    await services.indexing.index_collection("docs", source=source)
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

Detailed module documentation now lives alongside the implementation for easier maintenance:

- Core Module: [src/pmd/core/README.md](src/pmd/core/README.md)
- Store Module: [src/pmd/store/README.md](src/pmd/store/README.md)
- Services Module: [src/pmd/services/README.md](src/pmd/services/README.md)
- CLI Module: [src/pmd/cli/README.md](src/pmd/cli/README.md)
- LLM Module: [src/pmd/llm/README.md](src/pmd/llm/README.md)
- Search Module: [src/pmd/search/README.md](src/pmd/search/README.md)
- Sources Module: [src/pmd/sources/README.md](src/pmd/sources/README.md)
- Utils Module: [src/pmd/utils/README.md](src/pmd/utils/README.md)
- Formatters Module: [src/pmd/formatters/README.md](src/pmd/formatters/README.md)

## Entry Points

### CLI Entry Point

```
python -m pmd [command] [options]
```

Implemented in `src/pmd/__main__.py`, delegates to `cli.main.main()`.

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
IndexingService.index_collection("my-docs", source)
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
