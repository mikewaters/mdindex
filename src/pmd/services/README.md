# Services Module Architecture

**Location:** `src/pmd/services/`

Business logic orchestration layer coordinating repositories and LLM components.

## Files and Key Abstractions

### `container.py`

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

### `indexing.py`

**`IndexingService`** - Document indexing and embedding

#### Component Architecture

![Indexing Service Architecture](../../../docs/assets/indexing_service.svg)

#### Sequence Diagram

![Indexing Sequence](../../../docs/assets/indexing_sequence.svg)

```mermaid
graph TD
    IndexingService -- manages collections --> CollectionRepository
    IndexingService -- stores content --> DocumentRepository
    IndexingService -- stores vectors --> EmbeddingRepository
    IndexingService -- updates FTS index --> FTS5SearchRepository
    IndexingService -- tracks source state --> SourceMetadataRepository
    IndexingService -- stores doc metadata --> DocumentMetadataRepository
    IndexingService -- fetches documents --> DocumentSource
    IndexingService -- generates embeddings --> EmbeddingGenerator
```

**`IndexResult`** - Indexing operation result
- indexed, skipped, errors fields

**`EmbedResult`** - Embedding operation result
- embedded, skipped, chunks_total fields

**`CleanupResult`** - Cleanup operation result
- orphaned_content, orphaned_embeddings fields

Key methods:
- `index_collection()` - Enumerate, fetch, store, index documents (requires DocumentSource)
- `embed_collection()` - Generate embeddings for indexed docs
- `update_all_collections()` - Batch update all collections
- `cleanup_orphans()` - Remove orphaned content and embeddings

**Invariants:**
- Indexing continues on individual document errors
- Source metadata (ETags) enables incremental updates
- Content hash comparison for change detection

### `search.py`

**`SearchService`** - Search operation orchestration

#### Component Architecture

![Search Service Architecture](../../../docs/assets/search_service.svg)

#### Sequence Diagram

![Search Sequence](../../../docs/assets/search_sequence.svg)

```mermaid
graph TD
    SearchService -- lexical search --> FTS5SearchRepository
    SearchService -- semantic search --> EmbeddingRepository
    SearchService -- resolves names --> CollectionRepository
    SearchService -- embeds queries --> EmbeddingGenerator
    SearchService -- orchestrates search --> HybridSearchPipeline
    SearchService -- expands queries --> QueryExpander
    SearchService -- reranks results --> Reranker
```

Key methods:
- `fts_search()` - Synchronous BM25 search
- `vector_search()` - Async semantic search
- `hybrid_search()` - Combined FTS + vector with optional LLM

**Invariants:**
- Vector search requires sqlite-vec and LLM provider
- Hybrid search delegates to HybridSearchPipeline

### `status.py`

**`StatusService`** - Index health monitoring

#### Component Architecture

![Status Service Architecture](../../../docs/assets/status_service.svg)

```mermaid
graph TD
    StatusService -- lists collections --> CollectionRepository
    StatusService -- accesses db path --> Config
```

Key methods:
- `get_index_status()` - Quick status summary
- `get_full_status()` - Comprehensive async status
- `get_collection_stats()` - Per-collection statistics
- `get_index_sync_report()` - FTS/vector sync status
