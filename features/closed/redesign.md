# Redesign - Substrate

## Layering
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



## Configuration
Pydantic settings

---

## Software Patterns

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
## 1. Database Overview

PMD uses SQLite as its primary data store, with two optional extensions:

| Component | Purpose | Required |
|-----------|---------|----------|
| **SQLite** | Relational data storage | Yes |
| **FTS5** | Full-text search with BM25 ranking | Yes (built-in) |
| **sqlite-vec** | Vector similarity search | Optional |

## 2. Database Entities

xxxx
## 7. Content-Addressable Storage

PMD uses a content-addressable storage (CAS) pattern for document content:

### How It Works

1. **Content Hashing**: Each document's content is hashed using SHA256
2. **Deduplication**: Content is stored once in the `content` table, keyed by hash
3. **Reference**: Documents point to content via the `hash` foreign key

### Benefits

- **Storage Efficiency**: Identical documents share storage
- **Fast Change Detection**: Compare hashes instead of content
- **Version Tracking**: Different versions have different hashes

# Technical Features
### Incremental Update Flow