# PMD Library Class Diagrams

This document provides class diagrams for each module in the PMD library, showing class structures, inheritance hierarchies, and inter-module relationships.

## Module Overview

The PMD library consists of the following modules (excluding CLI and MCP):

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **core** | Types, configuration, exceptions | Config, Collection, SearchResult, RankedResult |
| **llm** | LLM provider abstraction | LLMProvider, EmbeddingGenerator, QueryExpander, DocumentReranker |
| **search** | Search pipeline and algorithms | HybridSearchPipeline, SearchPipelineConfig |
| **services** | Business logic orchestration | ServiceContainer, IndexingService, SearchService, StatusService |
| **sources** | Document source abstraction | DocumentSource, FileSystemSource, HTTPSource, EntitySource |
| **store** | Data access layer | Database, *Repository classes |
| **utils** | Utility functions | (No classes - only functions) |

---

## Core

The core module provides foundational types, configuration, and exceptions. It has **zero dependencies** on other PMD modules.

### Exception Hierarchy

```mermaid
classDiagram
    direction TB

    Exception <|-- PMDError
    PMDError <|-- DatabaseError
    PMDError <|-- CollectionError
    PMDError <|-- DocumentError
    PMDError <|-- LLMError
    PMDError <|-- SearchError
    PMDError <|-- EmbeddingError
    PMDError <|-- FormatError
    PMDError <|-- VirtualPathError

    CollectionError <|-- CollectionNotFoundError
    CollectionError <|-- CollectionExistsError
    DocumentError <|-- DocumentNotFoundError
    LLMError <|-- ModelNotFoundError

    class PMDError {
        +message: str
    }
    class DocumentNotFoundError {
        +path: str
        +suggestions: list~str~
    }
```

### Configuration Classes

```mermaid
classDiagram
    direction TB

    Config *-- LMStudioConfig
    Config *-- OpenRouterConfig
    Config *-- MLXConfig
    Config *-- SearchConfig
    Config *-- ChunkConfig
    Config *-- TracingConfig

    class Config {
        +db_path: Path
        +llm_provider: str
        +lm_studio: LMStudioConfig
        +openrouter: OpenRouterConfig
        +mlx: MLXConfig
        +search: SearchConfig
        +chunk: ChunkConfig
        +tracing: TracingConfig
        +from_env()$ Config
        +from_file(path)$ Config
        +from_env_or_file()$ Config
    }

    class LMStudioConfig {
        +base_url: str
        +embedding_model: str
        +expansion_model: str
        +reranker_model: str
        +timeout: int
    }

    class OpenRouterConfig {
        +api_key: str
        +base_url: str
        +embedding_model: str
        +expansion_model: str
        +reranker_model: str
        +timeout: int
    }

    class MLXConfig {
        +model: str
        +embedding_model: str
        +embedding_dimension: int
        +query_prefix: str
        +document_prefix: str
        +max_tokens: int
        +temperature: float
        +lazy_load: bool
    }

    class SearchConfig {
        +default_limit: int
        +fts_weight: float
        +vec_weight: float
        +rrf_k: int
        +top_rank_bonus: float
        +expansion_weight: float
        +rerank_candidates: int
    }

    class ChunkConfig {
        +max_bytes: int
        +min_chunk_size: int
    }

    class TracingConfig {
        +enabled: bool
        +phoenix_endpoint: str
        +service_name: str
        +service_version: str
        +sample_rate: float
        +batch_export: bool
    }
```

### Data Types

```mermaid
classDiagram
    direction TB

    DocumentResult <|-- SearchResult

    class VirtualPath {
        +collection_name: str
        +path: str
        +__str__() str
    }

    class Collection {
        +id: int
        +name: str
        +pwd: str
        +glob_pattern: str
        +created_at: datetime
        +updated_at: datetime
        +source_type: str
        +source_config: dict
        +get_source_uri() str
        +get_source_config_dict() dict
    }

    class DocumentResult {
        +filepath: str
        +display_path: str
        +title: str
        +context: str
        +hash: str
        +collection_id: int
        +modified_at: datetime
        +body_length: int
        +body: str
    }

    class SearchResult {
        +score: float
        +source: SearchSource
        +chunk_pos: int
        +snippet: str
    }

    class RankedResult {
        +file: str
        +display_path: str
        +title: str
        +body: str
        +score: float
        +fts_score: float
        +vec_score: float
        +rerank_score: float
        +fts_rank: int
        +vec_rank: int
        +sources_count: int
        +relevant: bool
        +rerank_confidence: float
        +rerank_raw_token: str
        +blend_weight: float
    }

    class EmbeddingResult {
        +embedding: list~float~
        +model: str
    }

    class RerankResult {
        +results: list~RerankDocumentResult~
        +model: str
    }

    class RerankDocumentResult {
        +file: str
        +relevant: bool
        +confidence: float
        +score: float
        +raw_token: str
        +logprob: float
    }

    class Chunk {
        +text: str
        +pos: int
    }

    class IndexStatus {
        +collections: list~Collection~
        +total_documents: int
        +embedded_documents: int
        +index_size_bytes: int
        +cache_entries: int
        +ollama_available: bool
        +models_available: list~str~
    }

    class SearchSource {
        <<enumeration>>
        FTS
        VECTOR
        HYBRID
    }

    class OutputFormat {
        <<enumeration>>
        JSON
        CSV
        XML
        MARKDOWN
        FILES
        CLI
    }

    RerankResult *-- RerankDocumentResult
    IndexStatus *-- Collection
```

### Cross-Module Interactions

```mermaid
flowchart LR
    subgraph core[Core Module]
        Config
        Types[Types & Results]
        Exceptions
    end

    llm[LLM] --> Config
    llm --> Types
    llm --> Exceptions

    search[Search] --> Config
    search --> Types

    services[Services] --> Config
    services --> Types
    services --> Exceptions

    store[Store] --> Types
    store --> Exceptions

    sources[Sources] --> Exceptions
```

---

## LLM

The LLM module provides an abstraction layer for multiple LLM providers with support for embeddings, text generation, and document reranking.

### Provider Hierarchy

```mermaid
classDiagram
    direction TB

    ABC <|-- LLMProvider
    LLMProvider <|-- LMStudioProvider
    LLMProvider <|-- OpenRouterProvider
    LLMProvider <|-- MLXProvider

    class LLMProvider {
        <<abstract>>
        +embed(texts, model)* EmbeddingResult
        +generate(prompt, model)* str
        +rerank(query, docs, model)* RerankResult
        +model_exists(model)* bool
        +is_available()* bool
        +close()* None
        +get_default_embedding_model() str
        +get_default_expansion_model() str
        +get_default_reranker_model() str
    }

    class LMStudioProvider {
        -config: LMStudioConfig
        -client: httpx.AsyncClient
        +embed(texts, model) EmbeddingResult
        +generate(prompt, model) str
        +rerank(query, docs, model) RerankResult
    }

    class OpenRouterProvider {
        -config: OpenRouterConfig
        -client: httpx.AsyncClient
        +embed(texts, model) EmbeddingResult
        +generate(prompt, model) str
        +rerank(query, docs, model) RerankResult
    }

    class MLXProvider {
        -config: MLXConfig
        -model: Any
        -tokenizer: Any
        -embed_model: Any
        +embed(texts, model) EmbeddingResult
        +generate(prompt, model) str
        +rerank(query, docs, model) RerankResult
        +unload_models() None
    }
```

### Service Classes

```mermaid
classDiagram
    direction TB

    EmbeddingGenerator --> LLMProvider : uses
    EmbeddingGenerator --> EmbeddingRepository : uses
    EmbeddingGenerator --> Config : uses

    QueryExpander --> LLMProvider : uses

    DocumentReranker --> LLMProvider : uses

    class EmbeddingGenerator {
        -provider: LLMProvider
        -repo: EmbeddingRepository
        -config: Config
        +embed_document(doc_hash, content) int
        +embed_query(query) list~float~
        +get_embeddings_for_content(hash) list~list~float~~
        +clear_embeddings_by_model(model) int
        +get_embedding_count() int
    }

    class QueryExpander {
        -provider: LLMProvider
        +expand(query) list~str~
        +expand_with_semantics(query) list~str~
    }

    class DocumentReranker {
        -provider: LLMProvider
        +get_rerank_scores(query, docs) RerankResult
        +rerank(query, results, top_n) list~RankedResult~
        +score_document(query, doc) RerankDocumentResult
    }
```

### Cross-Module Interactions

```mermaid
flowchart LR
    subgraph llm[LLM Module]
        LLMProvider
        EmbeddingGenerator
        QueryExpander
        DocumentReranker
    end

    subgraph core[Core]
        Config
        Types[EmbeddingResult, RerankResult, RankedResult]
    end

    subgraph store[Store]
        EmbeddingRepository
    end

    subgraph search[Search]
        chunking[chunk_document]
        text[is_indexable]
    end

    LLMProvider --> Config
    EmbeddingGenerator --> EmbeddingRepository
    EmbeddingGenerator --> chunking
    EmbeddingGenerator --> text
    DocumentReranker --> Types
```

---

## Search

The search module implements the hybrid search pipeline combining full-text search (FTS5) and vector search with RRF fusion and LLM reranking.

### Class Diagram

```mermaid
classDiagram
    direction TB

    HybridSearchPipeline --> SearchPipelineConfig : configured by
    HybridSearchPipeline --> FTS5SearchRepository : uses
    HybridSearchPipeline --> EmbeddingGenerator : uses
    HybridSearchPipeline --> QueryExpander : uses
    HybridSearchPipeline --> DocumentReranker : uses

    class SearchPipelineConfig {
        +fts_weight: float
        +vec_weight: float
        +rrf_k: int
        +top_rank_bonus: float
        +expansion_weight: float
        +rerank_candidates: int
        +enable_query_expansion: bool
        +enable_reranking: bool
        +normalize_final_scores: bool
    }

    class HybridSearchPipeline {
        -fts_repo: FTS5SearchRepository
        -config: SearchPipelineConfig
        -query_expander: QueryExpander
        -reranker: DocumentReranker
        -embedding_gen: EmbeddingGenerator
        +search(query, collection_id, limit) list~RankedResult~
        -_expand_query(query) list~str~
        -_parallel_search(queries, collection_id) tuple
        -_rerank_with_blending(query, results) list~RankedResult~
    }

    class ChunkingResult {
        +chunks: list~Chunk~
        +total_bytes: int
    }

    class _DocProvenance {
        <<internal>>
        +doc: SearchResult
        +fts_score: float
        +vec_score: float
        +fts_rank: int
        +vec_rank: int
        +sources: set
    }
```

### Pipeline Functions

```mermaid
flowchart TB
    subgraph pipeline[Search Pipeline]
        direction TB
        query[Query] --> expand[Query Expansion]
        expand --> parallel[Parallel Search]
        parallel --> fts[FTS5 Search]
        parallel --> vec[Vector Search]
        fts --> fusion[RRF Fusion]
        vec --> fusion
        fusion --> rerank[LLM Reranking]
        rerank --> blend[Score Blending]
        blend --> normalize[Normalization]
        normalize --> results[RankedResult List]
    end
```

### Cross-Module Interactions

```mermaid
flowchart LR
    subgraph search[Search Module]
        HybridSearchPipeline
        SearchPipelineConfig
        fusion[fusion.py]
        scoring[scoring.py]
        chunking[chunking.py]
    end

    subgraph core[Core]
        RankedResult
        SearchResult
        Chunk
        ChunkConfig
    end

    subgraph store[Store]
        FTS5SearchRepository
    end

    subgraph llm[LLM]
        EmbeddingGenerator
        QueryExpander
        DocumentReranker
    end

    HybridSearchPipeline --> FTS5SearchRepository
    HybridSearchPipeline -.-> EmbeddingGenerator
    HybridSearchPipeline -.-> QueryExpander
    HybridSearchPipeline -.-> DocumentReranker
    fusion --> SearchResult
    fusion --> RankedResult
    chunking --> Chunk
    chunking --> ChunkConfig
```

---

## Services

The services module provides business logic orchestration using a dependency injection container pattern.

### Service Container

```mermaid
classDiagram
    direction TB

    ServiceContainer *-- IndexingService
    ServiceContainer *-- SearchService
    ServiceContainer *-- StatusService
    ServiceContainer --> Database : manages
    ServiceContainer --> Config : uses

    class ServiceContainer {
        -_db: Database
        -_config: Config
        -_llm_provider: LLMProvider
        -_indexing: IndexingService
        -_search: SearchService
        -_status: StatusService
        +indexing: IndexingService
        +search: SearchService
        +status: StatusService
        +collection_repo: CollectionRepository
        +document_repo: DocumentRepository
        +embedding_repo: EmbeddingRepository
        +fts_repo: FTS5SearchRepository
        +embedding_generator: EmbeddingGenerator
        +query_expander: QueryExpander
        +reranker: DocumentReranker
        +__aenter__() ServiceContainer
        +__aexit__() None
    }

    class IndexingService {
        -_container: ServiceContainer
        +index_collection(name) IndexResult
        +embed_collection(name) EmbedResult
        +cleanup_orphans() CleanupResult
    }

    class SearchService {
        -_container: ServiceContainer
        +fts_search(query, collection, limit) list~SearchResult~
        +vector_search(query, collection, limit) list~SearchResult~
        +hybrid_search(query, collection, limit) list~RankedResult~
    }

    class StatusService {
        -_container: ServiceContainer
        +get_status() IndexStatus
    }
```

### Data Classes

```mermaid
classDiagram
    class IndexResult {
        +indexed: int
        +skipped: int
        +errors: list~tuple~
    }

    class EmbedResult {
        +embedded: int
        +skipped: int
        +chunks_total: int
    }

    class CleanupResult {
        +orphaned_content: int
        +orphaned_embeddings: int
    }
```

### Cross-Module Interactions

```mermaid
flowchart LR
    subgraph services[Services Module]
        ServiceContainer
        IndexingService
        SearchService
        StatusService
    end

    subgraph core[Core]
        Config
        Types[Collection, RankedResult, IndexStatus]
        Exceptions
    end

    subgraph store[Store]
        Database
        Repos[*Repository Classes]
    end

    subgraph llm[LLM]
        LLMProvider
        EmbeddingGenerator
        QueryExpander
        DocumentReranker
    end

    subgraph search[Search]
        HybridSearchPipeline
    end

    subgraph sources[Sources]
        DocumentSource
    end

    subgraph utils[Utils]
        sha256_hash
    end

    ServiceContainer --> Database
    ServiceContainer --> Config
    ServiceContainer --> Repos
    ServiceContainer --> LLMProvider
    IndexingService --> DocumentSource
    IndexingService --> sha256_hash
    SearchService --> HybridSearchPipeline
```

---

## Sources

The sources module provides a document source abstraction supporting filesystem, HTTP, and custom entity resolvers.

### Protocol and Implementations

```mermaid
classDiagram
    direction TB

    DocumentSource <|.. BaseDocumentSource
    BaseDocumentSource <|-- FileSystemSource
    BaseDocumentSource <|-- HTTPSource
    BaseDocumentSource <|-- EntitySource

    class DocumentSource {
        <<protocol>>
        +list_documents()* AsyncIterator~DocumentReference~
        +fetch_content(ref)* FetchResult
        +capabilities()* SourceCapabilities
        +check_modified(ref)* bool
    }

    class BaseDocumentSource {
        +capabilities() SourceCapabilities
        +check_modified(ref) bool
    }

    class FileSystemSource {
        -config: FileSystemConfig
        +list_documents() AsyncIterator~DocumentReference~
        +fetch_content(ref) FetchResult
        +capabilities() SourceCapabilities
        +check_modified(ref) bool
    }

    class HTTPSource {
        -config: HTTPConfig
        -credential_resolver: CredentialResolver
        -client: httpx.AsyncClient
        +list_documents() AsyncIterator~DocumentReference~
        +fetch_content(ref) FetchResult
        +capabilities() SourceCapabilities
        +check_modified(ref) bool
    }

    class EntitySource {
        -registry: EntityResolverRegistry
        -parsed_uri: ParsedEntityURI
        +list_documents() AsyncIterator~DocumentReference~
        +fetch_content(ref) FetchResult
        +capabilities() SourceCapabilities
    }
```

### Configuration Classes

```mermaid
classDiagram
    class SourceConfig {
        +uri: str
        +extra: dict
    }

    class FileSystemConfig {
        +base_path: Path
        +glob_pattern: str
        +encoding: str
        +follow_symlinks: bool
        +from_source_config(config)$ FileSystemConfig
    }

    class HTTPConfig {
        +base_url: str
        +urls: list~str~
        +sitemap_url: str
        +auth: AuthConfig
        +timeout_seconds: int
        +max_retries: int
        +follow_redirects: bool
        +allowed_content_types: list~str~
        +user_agent: str
        +from_source_config(config)$ HTTPConfig
    }

    class AuthConfig {
        +auth_type: str
        +token: str
        +username: str
        +api_key_header: str
        +custom_headers: dict
        +from_dict(d)$ AuthConfig
        +get_headers(resolver) dict
    }
```

### Credential Providers

```mermaid
classDiagram
    direction TB

    CredentialProvider <|.. EnvironmentCredentials
    CredentialProvider <|.. KeyringCredentials
    CredentialProvider <|.. StaticCredentials

    CredentialResolver *-- EnvironmentCredentials
    CredentialResolver *-- KeyringCredentials
    CredentialResolver *-- StaticCredentials

    class CredentialProvider {
        <<protocol>>
        +name: str
        +get_credential(key)* str
    }

    class EnvironmentCredentials {
        +name: str
        +get_credential(key) str
    }

    class KeyringCredentials {
        +name: str
        +get_credential(key) str
    }

    class StaticCredentials {
        +name: str
        -credentials: dict
        +get_credential(key) str
    }

    class CredentialResolver {
        -env: EnvironmentCredentials
        -keyring: KeyringCredentials
        -static: StaticCredentials
        +resolve(reference) str
    }
```

### Entity Resolver System

```mermaid
classDiagram
    direction TB

    EntityResolver <|.. CustomResolver
    EntityResolverRegistry o-- EntityResolver
    EntitySource --> EntityResolverRegistry : uses

    class EntityResolver {
        <<protocol>>
        +list_entities(resource_type)* AsyncIterator~EntityInfo~
        +fetch_entity(resource_type, id)* EntityContent
    }

    class EntityResolverRegistry {
        -resolvers: dict
        +register(name, resolver) None
        +unregister(name) None
        +get(name) EntityResolver
        +list_resolvers() list~str~
    }

    class EntityInfo {
        +id: str
        +title: str
        +path: str
        +metadata: dict
        +get_path() str
    }

    class EntityContent {
        +content: str
        +title: str
        +content_type: str
        +metadata: dict
    }

    class ParsedEntityURI {
        +resolver: str
        +resource_type: str
        +entity_id: str
        +query: dict
    }
```

### Registry Pattern

```mermaid
classDiagram
    direction TB

    SourceRegistry --> FileSystemSource : creates
    SourceRegistry --> HTTPSource : creates
    SourceRegistry --> EntitySource : creates

    class SourceRegistry {
        -factories: dict~str, SourceFactory~
        +register(scheme, factory) None
        +unregister(scheme) None
        +resolve(uri) DocumentSource
        +is_registered(scheme) bool
        +supported_schemes() list~str~
    }

    class SourceFactory {
        <<type alias>>
        Callable[[SourceConfig], DocumentSource]
    }
```

### Exception Hierarchy

```mermaid
classDiagram
    direction TB

    PMDError <|-- SourceError
    PMDError <|-- CredentialError
    PMDError <|-- EntityResolverError
    PMDError <|-- RegistryError

    SourceError <|-- SourceListError
    SourceError <|-- SourceFetchError
    CredentialError <|-- CredentialNotFoundError
    EntityResolverError <|-- ResolverNotFoundError
    RegistryError <|-- UnknownSchemeError
    RegistryError <|-- SourceCreationError
```

### Cross-Module Interactions

```mermaid
flowchart LR
    subgraph sources[Sources Module]
        DocumentSource
        FileSystemSource
        HTTPSource
        EntitySource
        SourceRegistry
    end

    subgraph core[Core]
        PMDError
    end

    sources --> PMDError

    services[Services] --> sources
```

---

## Store

The store module provides the data access layer using SQLite with content-addressable storage and optional vector search.

### Repository Pattern

```mermaid
classDiagram
    direction TB

    Database <-- CollectionRepository : uses
    Database <-- DocumentRepository : uses
    Database <-- EmbeddingRepository : uses
    Database <-- FTS5SearchRepository : uses
    Database <-- SourceMetadataRepository : uses

    SearchRepository~T~ <|-- FTS5SearchRepository
    SearchRepository~T~ <|-- VectorSearchRepository
    VectorSearchRepository --> EmbeddingRepository : uses

    class Database {
        -conn: aiosqlite.Connection
        -path: Path
        +vec_available: bool
        +connect() None
        +close() None
        +transaction() AsyncContextManager
        +execute(sql, params) Cursor
        +executescript(sql) None
    }

    class SearchRepository~T~ {
        <<abstract>>
        +search(query, collection_id, limit)* list~SearchResult~
    }

    class CollectionRepository {
        -db: Database
        +list_all() list~Collection~
        +get_by_name(name) Collection
        +get_by_id(id) Collection
        +create(name, pwd, pattern, source_type, config) Collection
        +remove(name) None
        +rename(old_name, new_name) None
        +update_collection_path(id, pwd) None
    }

    class DocumentRepository {
        -db: Database
        +add_or_update(coll_id, path, title, body, context) DocumentResult
        +get(coll_id, path) DocumentResult
        +get_by_hash(hash) DocumentResult
        +list_by_collection(coll_id) list~DocumentResult~
        +delete(coll_id, path) None
        +check_if_modified(coll_id, path, hash) bool
        +get_content_length(hash) int
        +count_by_collection(coll_id) int
    }

    class EmbeddingRepository {
        -db: Database
        +store_embedding(hash, pos, embedding, model) None
        +get_embeddings_for_content(hash) list~list~float~~
        +has_embeddings(hash, model) bool
        +delete_embeddings(hash) None
        +clear_embeddings_by_model(model) int
        +count_embeddings() int
        +search_vectors(embedding, coll_id, limit) list~SearchResult~
    }

    class FTS5SearchRepository {
        -db: Database
        +search(query, coll_id, limit) list~SearchResult~
        +index_document(coll_id, path, title, body) None
        +reindex_collection(coll_id) None
        +remove_from_index(coll_id, path) None
        +clear_index() None
    }

    class VectorSearchRepository {
        -embedding_repo: EmbeddingRepository
        +search(embedding, coll_id, limit) list~SearchResult~
    }

    class SourceMetadataRepository {
        -db: Database
        +upsert(doc_id, metadata) None
        +get_by_document(doc_id) SourceMetadata
        +get_by_uri(uri) SourceMetadata
        +delete_by_document(doc_id) None
        +needs_refresh(doc_id, current_etag) bool
        +get_stale_documents(coll_id) list~int~
        +cleanup_orphans() int
    }
```

### Data Classes

```mermaid
classDiagram
    class SourceMetadata {
        +document_id: int
        +source_uri: str
        +etag: str
        +last_modified: datetime
        +content_type: str
        +fetch_timestamp: datetime
    }
```

### Cross-Module Interactions

```mermaid
flowchart LR
    subgraph store[Store Module]
        Database
        CollectionRepository
        DocumentRepository
        EmbeddingRepository
        FTS5SearchRepository
        VectorSearchRepository
        SourceMetadataRepository
    end

    subgraph core[Core]
        Collection
        DocumentResult
        SearchResult
        Exceptions[DatabaseError, DocumentNotFoundError, CollectionErrors]
    end

    subgraph utils[Utils]
        sha256_hash
    end

    store --> Collection
    store --> DocumentResult
    store --> SearchResult
    store --> Exceptions
    DocumentRepository --> sha256_hash

    services[Services] --> store
    llm[LLM] --> EmbeddingRepository
    search[Search] --> FTS5SearchRepository
```

---

## Utils

The utils module contains utility functions with no classes. It serves as a shared utility layer.

### Functions

```mermaid
flowchart TB
    subgraph utils[Utils Module]
        hashing[hashing.py]
    end

    hashing --> sha256_hash["sha256_hash(content: str) -> str"]
    hashing --> sha256_hash_bytes["sha256_hash_bytes(content: bytes) -> str"]

    subgraph consumers[Consumers]
        store_docs[store.documents.DocumentRepository]
        services_idx[services.indexing.IndexingService]
    end

    store_docs --> sha256_hash
    services_idx --> sha256_hash
```

---

## Module Dependency Graph

This diagram shows the overall dependency relationships between all modules:

```mermaid
flowchart TB
    subgraph foundation[Foundation Layer]
        core[Core]
        utils[Utils]
    end

    subgraph data[Data Layer]
        store[Store]
    end

    subgraph integration[Integration Layer]
        sources[Sources]
        llm[LLM]
    end

    subgraph processing[Processing Layer]
        search[Search]
    end

    subgraph orchestration[Orchestration Layer]
        services[Services]
    end

    %% Foundation dependencies
    store --> core
    store --> utils

    sources --> core

    llm --> core
    llm --> store
    llm --> search

    search --> core
    search --> store

    services --> core
    services --> store
    services --> llm
    services --> search
    services --> sources
    services --> utils
```

### Dependency Matrix

| Module | core | utils | store | llm | search | sources | services |
|--------|:----:|:-----:|:-----:|:---:|:------:|:-------:|:--------:|
| **core** | - | - | - | - | - | - | - |
| **utils** | - | - | - | - | - | - | - |
| **store** | Y | Y | - | - | - | - | - |
| **llm** | Y | - | Y | - | Y | - | - |
| **search** | Y | - | Y | Y* | - | - | - |
| **sources** | Y | - | - | - | - | - | - |
| **services** | Y | Y | Y | Y | Y | Y | - |

*Y = Direct dependency, Y* = TYPE_CHECKING only (optional runtime dependency)
