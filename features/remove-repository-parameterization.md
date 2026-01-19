# Remove Repository Parameterization from Upstream Abstractions

## Summary

Remove the ability for services, workflows, and the Application container to accept custom repository implementations as constructor parameters. Only the layer directly above `pmd.store` should import and instantiate repositories.

## Current State

### Where Repositories Are Parameterized

| Layer | Module | Repositories Accepted |
|-------|--------|----------------------|
| Application | `pmd.app.Application` | `source_collection_repo`, `document_repo`, `embedding_repo` |
| Services | `pmd.services.indexing.IndexingService` | `source_collection_repo`, `document_repo`, `fts_repo`, `content_repo`, `embedding_repo` |
| Services | `pmd.services.loading.LoadingService` | `source_collection_repo`, `document_repo`, `source_metadata_repo` |
| Services | `pmd.services.search.SearchService` | `source_collection_repo`, `fts_repo` |
| Services | `pmd.services.status.StatusService` | `source_collection_repo`, `document_repo`, `fts_repo`, `embedding_repo` |
| Workflows | `pmd.workflows.pipelines.ingestion.IngestionPipeline` | `source_collection_repo`, `document_repo`, `fts_repo` |
| Workflows | `pmd.workflows.pipelines.embedding.EmbeddingPipeline` | `source_collection_repo`, `embedding_repo` |
| Adapters | `pmd.search.adapters.*` | Various repositories |

### Supporting Infrastructure

- **`tests/pmd/fakes/repos.py`**: In-memory repository implementations for testing
- **`pmd.app.factory.create_application()`**: Composition root that creates and wires repositories

## Problems with Current Approach

1. **Leaky Abstraction**: Services know about storage implementation details (repositories, Database)
2. **Over-parameterization**: 6+ repository parameters per service constructor
3. **Unused Flexibility**: The parameterization allows custom implementations, but there's only ONE `pmd.store`
4. **Testing Complexity**: Maintaining fake repositories that mirror real ones

## Proposed Architecture

### Layering

```
┌─────────────────────────────────────────┐
│              CLI / API                  │
├─────────────────────────────────────────┤
│         Application Container           │  ← Only holds services
├─────────────────────────────────────────┤
│              Services                   │  ← NO repository knowledge
│  (IndexingService, SearchService, etc.) │     Uses protocols for data access
├─────────────────────────────────────────┤
│           Data Access Layer             │  ← NEW: Only layer with repo knowledge
│    (wraps repositories in protocols)    │
├─────────────────────────────────────────┤
│             pmd.store                   │  ← Unchanged: repositories + Database
│  (SourceCollectionRepository, etc.)    │
└─────────────────────────────────────────┘
```

### Key Changes

1. **Services use protocols**, not concrete repositories
2. **New data access layer** implements protocols using repositories
3. **Factory creates data access layer**, injects into services
4. **Application holds services only**, not repositories

## Design Decisions

### Option A: Protocol-per-Operation (Granular)

Services define protocols for exactly what they need:

```python
# In pmd.services.indexing
class DocumentPersistence(Protocol):
    def save_document(self, collection_id: int, path: str, title: str, content: str) -> DocumentResult: ...
    def get_document(self, collection_id: int, path: str) -> DocumentResult | None: ...

class IndexingService:
    def __init__(self, persistence: DocumentPersistence, ...): ...
```

**Pros**: Maximum decoupling, services only see what they need
**Cons**: Many small protocols, more adapter classes

### Option B: Aggregate Data Access (Recommended)

Single data access class per domain combines related operations:

```python
# In pmd.data (new module)
class IndexingDataAccess:
    """Data access for indexing operations."""
    def __init__(self, db: Database):  # Only thing that touches store
        self._collection_repo = SourceCollectionRepository(db)
        self._document_repo = DocumentRepository(db)
        self._fts_repo = FTS5SearchRepository(db)
        # ... etc

    # Document operations
    def save_document(self, collection_id: int, path: str, title: str, content: str) -> DocumentResult: ...
    def get_document(self, collection_id: int, path: str) -> DocumentResult | None: ...

    # Collection operations
    def get_collection(self, name: str) -> SourceCollection | None: ...

    # FTS operations
    def index_for_search(self, doc_id: int, path: str, content: str) -> None: ...

class IndexingService:
    def __init__(self, data: IndexingDataAccess, loader: LoadingServiceProtocol): ...
```

**Pros**: Fewer classes, cleaner factory, easier testing
**Cons**: Services still see storage-shaped operations

### Option C: Keep Repositories, Remove Parameterization

Services create their own repositories from a shared Database:

```python
class IndexingService:
    def __init__(self, db: Database, loader: LoadingServiceProtocol):
        self._collection_repo = SourceCollectionRepository(db)
        self._document_repo = DocumentRepository(db)
        # ... internally created, not injected
```

**Pros**: Simplest change, no new abstractions
**Cons**: Services still know about repositories (just not injected)

## Recommended Approach: Option B

Create a `pmd.data` module with aggregate data access classes:

```
src/pmd/data/
├── __init__.py          # Exports: IndexingData, SearchData, StatusData
├── indexing.py          # IndexingData - combines doc, fts, content, embedding repos
├── search.py            # SearchData - combines fts, embedding, collection repos
├── loading.py           # LoadingData - combines doc, collection, metadata repos
└── status.py            # StatusData - read-only aggregations
```

### Migration Path

1. **Create `pmd.data` module** with data access classes
2. **Update services** to accept data access instead of repositories
3. **Update factory** to create data access layer
4. **Remove repository parameters** from Application container
5. **Delete workflow pipelines** (inline their logic into services)
6. **Update or delete fake repositories** (may not be needed)

## Testing Strategy

With Option B, testing options:

1. **Integration tests**: Use real SQLite `:memory:` database (fast, realistic)
2. **Data access fakes**: Mock the data access layer, not individual repos
3. **Hybrid**: Real DB for data layer tests, fake data access for service tests

## Implementation Tasks

### Phase 1: Create Data Access Layer
- [ ] Create `pmd.data.indexing.IndexingData`
- [ ] Create `pmd.data.search.SearchData`
- [ ] Create `pmd.data.loading.LoadingData`
- [ ] Create `pmd.data.status.StatusData`

### Phase 2: Migrate Services
- [ ] Update `IndexingService` to use `IndexingData`
- [ ] Update `SearchService` to use `SearchData`
- [ ] Update `LoadingService` to use `LoadingData`
- [ ] Update `StatusService` to use `StatusData`

### Phase 3: Simplify Application Layer
- [ ] Remove repository properties from `Application`
- [ ] Update `create_application()` factory
- [ ] Inline `IngestionPipeline` logic into `IndexingService`
- [ ] Inline `EmbeddingPipeline` logic into `IndexingService`

### Phase 4: Clean Up
- [ ] Delete or simplify `tests/pmd/fakes/repos.py`
- [ ] Update tests to use new patterns
- [ ] Remove unused repository imports from services

## Open Questions

1. Should `pmd.data` classes be protocols or concrete classes?
2. Should pipelines be kept as internal implementation details?
3. How should the search adapters (`pmd.search.adapters`) fit in?

## Appendix: Current Repository Usage Analysis

### IndexingService
- `source_collection_repo`: get_by_name, list_all
- `document_repo`: add_or_update, get, list_by_collection, delete, get_id
- `fts_repo`: index_document, remove_from_index
- `content_repo`: delete_orphaned
- `embedding_repo`: delete_orphaned
- `db`: direct execute for backfill_metadata

### LoadingService
- `source_collection_repo`: get_by_name
- `document_repo`: get, get_id
- `source_metadata_repo`: get_by_document

### SearchService
- `source_collection_repo`: get_by_name (for ID resolution)
- `fts_repo`: search
- `db`: vec_available property

### StatusService
- `source_collection_repo`: list_all, get_by_id
- `document_repo`: count_active, count_with_embeddings
- `fts_repo`: count_documents_missing_fts, list_paths_missing_fts, count_orphaned
- `embedding_repo`: count_embeddings, count_distinct_hashes, count_documents_missing_embeddings, list_paths_missing_embeddings, count_orphaned
