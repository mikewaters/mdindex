# Replace God Service Container with Explicit Composition

## Overview

This plan refactors the monolithic `ServiceContainer` into a clean composition root pattern where:
- Dependencies are explicit and injectable
- Lifecycle management is separate from service wiring
- Services accept interfaces, not container internals
- Sync/async concerns are cleanly separated

## Current State Analysis

### ServiceContainer Responsibilities (Too Many)
1. Database lifecycle (connect/close)
2. LLM provider lifecycle (async close)
3. Repository instantiation (4 repos)
4. LLM component instantiation (4 components)
5. Metadata component instantiation (4 components)
6. Service instantiation (3 services)
7. Configuration holding
8. Direct repository exposure to CLI/MCP

### Problematic Patterns
- Services reach into `_container.*` for dependencies
- CLI accesses repositories directly via `services.collection_repo`
- Mixed sync/async accessors (`get_llm_provider()` async, `get_tag_matcher()` sync)
- MCP server uses manual lifecycle instead of context manager
- `collection.py` bypasses container entirely

## Target Architecture

```
pmd/
├── app/                          # NEW: Composition root
│   ├── __init__.py              # Factory functions
│   ├── factories.py             # Preset factory helpers (MLX, OpenRouter)
│   └── types.py                 # Protocol definitions for dependencies
├── services/
│   ├── container.py             # SIMPLIFIED: Only lifecycle management
│   ├── indexing.py              # Constructor accepts dependencies
│   ├── search.py                # Constructor accepts dependencies
│   └── status.py                # Constructor accepts dependencies
```

## Implementation Steps

### Phase 1: Define Dependency Protocols

Create `pmd/app/types.py` with Protocol definitions for all injectable dependencies:
- `CollectionRepository` protocol
- `DocumentRepository` protocol
- `LLMProvider` protocol
- etc.

This allows services to depend on interfaces, not implementations.

### Phase 2: Refactor Service Constructors

Update service constructors to accept explicit dependencies:

**Before (IndexingService):**
```python
def __init__(self, container: "ServiceContainer"):
    self._container = container

# Usage
doc = self._container.document_repo.get(...)
```

**After (IndexingService):**
```python
def __init__(
    self,
    collection_repo: CollectionRepository,
    document_repo: DocumentRepository,
    fts_repo: FTS5SearchRepository,
    embedding_generator: EmbeddingGenerator | None = None,
):
    self._collection_repo = collection_repo
    self._document_repo = document_repo
    ...

# Usage
doc = self._document_repo.get(...)
```

### Phase 3: Create Composition Root

Create `pmd/app/__init__.py` with the main wiring function:

```python
async def create_application(config: Config) -> Application:
    """Wire all dependencies and return configured application."""
    db = Database(config.db_path)
    db.connect()

    # Create repositories
    collection_repo = CollectionRepository(db)
    document_repo = DocumentRepository(db)
    fts_repo = FTS5SearchRepository(db)
    embedding_repo = EmbeddingRepository(db)

    # Create LLM provider (async)
    llm_provider = await create_llm_provider(config)

    # Create services with explicit dependencies
    indexing = IndexingService(
        collection_repo=collection_repo,
        document_repo=document_repo,
        fts_repo=fts_repo,
        embedding_generator=EmbeddingGenerator(llm_provider, embedding_repo, config),
    )

    search = SearchService(
        fts_repo=fts_repo,
        embedding_repo=embedding_repo,
        embedding_generator=...,
    )

    return Application(
        db=db,
        llm_provider=llm_provider,
        indexing=indexing,
        search=search,
        status=StatusService(db, collection_repo, document_repo),
    )
```

### Phase 4: Simplify ServiceContainer to Lifecycle Manager

Reduce ServiceContainer to only manage lifecycle:

```python
class Application:
    """Holds wired services and manages their lifecycle."""

    def __init__(
        self,
        db: Database,
        llm_provider: LLMProvider | None,
        indexing: IndexingService,
        search: SearchService,
        status: StatusService,
    ):
        self._db = db
        self._llm_provider = llm_provider
        self.indexing = indexing
        self.search = search
        self.status = status

    async def close(self) -> None:
        """Clean shutdown of all resources."""
        if self._llm_provider:
            await self._llm_provider.close()
        self._db.close()

    async def __aenter__(self) -> "Application":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
```

### Phase 5: Create Factory Helpers

Add preset factories in `pmd/app/factories.py`:

```python
async def create_mlx_application(config: Config) -> Application:
    """Create application with local MLX provider."""
    return await create_application(config)

async def create_openrouter_application(config: Config) -> Application:
    """Create application with OpenRouter provider."""
    config.llm_provider = "openrouter"
    return await create_application(config)

def create_test_application(
    collection_repo: CollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
    ...
) -> Application:
    """Create application with test doubles."""
    return Application(
        db=InMemoryDatabase(),
        llm_provider=FakeLLMProvider(),
        indexing=IndexingService(
            collection_repo=collection_repo or InMemoryCollectionRepo(),
            ...
        ),
        ...
    )
```

### Phase 6: Update CLI Commands

Update CLI to use composition root:

**Before:**
```python
async with ServiceContainer(config) as services:
    collection = services.collection_repo.get_by_name(name)
```

**After:**
```python
async with create_application(config) as app:
    # Access through service methods, not repos
    result = await app.indexing.index_collection(name, source)
```

For collection management (sync only), provide a simpler helper:

```python
with create_collection_manager(config) as manager:
    manager.create(name, path, glob)
    manager.list_all()
```

### Phase 7: Update MCP Server

Update MCP server to use composition root:

**Before:**
```python
self._services = ServiceContainer(config)
self._services.connect()
...
doc = self.services.document_repo.get(...)
```

**After:**
```python
self._app = await create_application(config)
...
# Use service methods or expose specific read-only accessors
result = await self._app.search.hybrid_search(query)
```

### Phase 8: Add Testability Tests

Add tests demonstrating service construction with test doubles:

```python
def test_indexing_with_fake_repos():
    """IndexingService works with in-memory test doubles."""
    fake_collection_repo = InMemoryCollectionRepo()
    fake_document_repo = InMemoryDocumentRepo()

    service = IndexingService(
        collection_repo=fake_collection_repo,
        document_repo=fake_document_repo,
        fts_repo=InMemoryFTSRepo(),
    )

    # Test without database or LLM
    ...

def test_search_with_fake_llm():
    """SearchService works with fake LLM provider."""
    fake_llm = FakeLLMProvider()
    ...
```

## Migration Strategy

1. **Backward Compatible**: Keep ServiceContainer working during migration
2. **Gradual Migration**: Move one service at a time
3. **Feature Flags**: New app module optional at first
4. **Deprecation Warnings**: ServiceContainer.collection_repo etc. emit warnings

## Files Changed

| File | Change |
|------|--------|
| `src/pmd/app/__init__.py` | NEW: Composition root |
| `src/pmd/app/factories.py` | NEW: Preset factories |
| `src/pmd/app/types.py` | NEW: Dependency protocols |
| `src/pmd/services/container.py` | SIMPLIFIED: Lifecycle only |
| `src/pmd/services/indexing.py` | Constructor accepts deps |
| `src/pmd/services/search.py` | Constructor accepts deps |
| `src/pmd/services/status.py` | Constructor accepts deps |
| `src/pmd/cli/commands/*.py` | Use composition root |
| `src/pmd/mcp/server.py` | Use composition root |
| `tests/fakes/repos.py` | NEW: In-memory repo fakes |
| `tests/unit/services/test_*.py` | Test with doubles |

## Success Criteria

1. Services have no knowledge of ServiceContainer
2. All dependencies are explicit constructor parameters
3. CLI/MCP only access services, not repos directly
4. Tests can construct services with test doubles
5. Sync and async concerns are cleanly separated
6. Factory helpers simplify common setups
