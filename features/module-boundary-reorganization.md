# Feature: Module Boundary Reorganization

## Summary

Reorganize the `pmd/` module structure to clarify responsibilities, reduce cognitive overhead, and enable independent iteration on content collection, storage backends, and RAG integrations.

## Motivation

The current module structure has grown organically and presents several pain points:

1. **Ambiguous module purposes**: `sources/metadata/` vs `metadata/` distinction is unclear
2. **Scattered responsibilities**: Storage concerns spread across `store/`, `search/adapters/`, and `metadata/store/`
3. **Protocol duplication**: Similar protocols defined in `app/types.py` and `search/ports.py`
4. **Factory explosion**: Services require 10+ factory parameters due to assembly happening in wrong layer
5. **No clear extension points**: Adding a new RAG backend requires touching many unrelated modules

These issues create cognitive overhead when onboarding and make independent iteration difficult.

## Current Structure Analysis

```
pmd/
├── app/                    # Application container + ALL protocols (mixed concerns)
│   ├── __init__.py         # 374 lines - composition root + Application class
│   └── types.py            # Repository, LLM, metadata, config protocols
├── cli/                    # CLI (clear purpose ✓)
├── core/                   # Config, exceptions, types (clear ✓)
├── llm/                    # LLM providers (clear ✓)
├── metadata/               # Document metadata extraction + ontology + storage
│   ├── extraction/         # Profile implementations
│   ├── model/              # ExtractedMetadata types
│   ├── query/              # Tag matching
│   └── store/              # DocumentMetadataRepository (why here, not in store/?)
├── search/                 # Search pipeline + ports + adapters
│   ├── adapters/           # Port implementations (some use repositories directly)
│   ├── metadata/           # Metadata boosting (different from metadata/?)
│   └── ports.py            # Search-specific protocols (duplicates app/types.py)
├── services/               # High-level orchestration
├── sources/                # Document sources
│   ├── content/            # DocumentSource protocol + filesystem
│   └── metadata/           # Transport metadata (ETags, etc.) - confusing name
├── store/                  # SQLite persistence
│   ├── alembic/            # Migrations (old)
│   └── migrations/         # Migrations (new?) - which is canonical?
├── utils/                  # Miscellaneous
└── workflows/              # Pipelines
    └── pipelines/          # Ingestion, embedding pipelines
```

### Specific Confusions

| Question | Current Answer | Problem |
|----------|----------------|---------|
| Where do I add a metadata extractor? | `metadata/extraction/` | But `sources/metadata/` also exists |
| Where is storage logic? | `store/` | But `metadata/store/` has DocumentMetadataRepository |
| Which protocols do I implement? | `app/types.py` or `search/ports.py` | Duplication, unclear which is canonical |
| How do I add a vector store backend? | `store/embeddings.py`? `search/adapters/`? | No clear extension point |
| What's `sources/metadata/`? | Transport metadata (ETags) | Name suggests document metadata |

## Proposed Structure

### Option A: Clarify Existing Structure (Minimal Change)

Rename and consolidate without major restructuring:

```
pmd/
├── app/
│   ├── __init__.py         # Application class only
│   ├── factory.py          # create_application() - composition root
│   └── protocols.py        # ALL protocol definitions (consolidated)
├── cli/                    # Unchanged
├── core/                   # Unchanged
├── llm/                    # Unchanged
├── extraction/             # RENAMED from metadata/extraction/
│   ├── profiles/           # GenericProfile, ObsidianProfile, etc.
│   ├── registry.py
│   └── types.py            # ExtractedMetadata
├── ontology/               # EXTRACTED from metadata/
│   ├── model.py            # Tag hierarchy
│   ├── matcher.py          # Tag matching
│   └── booster.py          # MOVED from search/metadata/
├── search/
│   ├── pipeline.py         # HybridSearchPipeline
│   ├── fusion.py           # RRF
│   ├── adapters/           # All adapters (text, vector, tag, rerank)
│   └── chunking.py
├── services/               # Unchanged
├── sources/
│   ├── base.py             # DocumentSource protocol
│   ├── filesystem.py       # FileSystemSource
│   ├── registry.py
│   └── transport.py        # RENAMED from sources/metadata/ - ETags, etc.
├── store/                  # ALL persistence consolidated
│   ├── database.py
│   ├── models.py
│   ├── repositories/       # All repos in one place
│   │   ├── collections.py
│   │   ├── documents.py
│   │   ├── content.py
│   │   ├── embeddings.py
│   │   ├── fts.py
│   │   └── metadata.py     # MOVED from metadata/store/
│   └── migrations/
├── utils/
└── workflows/
```

**Key Changes:**
- `metadata/` split into `extraction/` (profiles) and `ontology/` (tags/hierarchy)
- `sources/metadata/` renamed to `sources/transport.py` (clarity)
- `metadata/store/` moved to `store/repositories/metadata.py`
- `search/metadata/` moved to `ontology/booster.py`
- Protocols consolidated in `app/protocols.py`

### Option B: Backend-Oriented Structure (For RAG Library Flexibility)

If the goal is to support multiple indexing/retrieval backends:

```
pmd/
├── core/                   # Domain types, config, exceptions
│   ├── types.py            # Document, Collection, SearchResult
│   ├── config.py
│   └── exceptions.py
├── backends/               # Pluggable index/retrieval backends
│   ├── base.py             # IndexBackend protocol
│   ├── sqlite/             # Current implementation
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── repositories.py # All repos
│   │   ├── search.py       # FTS5 + vector search
│   │   └── models.py
│   ├── llamaindex/         # LlamaIndex-based backend
│   │   └── ...
│   └── chroma/             # ChromaDB backend (future)
│       └── ...
├── sources/                # Document sources (unchanged)
│   ├── base.py
│   ├── filesystem.py
│   └── registry.py
├── extraction/             # Metadata extraction
│   ├── profiles/
│   └── ontology.py
├── llm/                    # LLM providers
├── services/               # Thin orchestration over backends
├── cli/
└── app.py                  # Minimal wiring
```

**Key Insight:** Each backend owns its complete storage + retrieval implementation. A `ChromaBackend` doesn't share repositories with `SQLiteBackend`.

## Recommended Approach: Option A

Option A provides immediate clarity without the risk of a major rewrite. Option B is better if you're committed to multi-backend support, but requires more upfront investment.

## Implementation Plan

### Phase 1: Consolidate Protocols

**Goal:** Single source of truth for all protocols.

**Files to modify:**
- `src/pmd/app/types.py` → `src/pmd/app/protocols.py`
- `src/pmd/search/ports.py` → merge into `app/protocols.py`

**Changes:**
1. Create `src/pmd/app/protocols.py` with all protocol definitions
2. Organize protocols by domain:
   - Storage protocols (repositories)
   - Search protocols (pipeline ports)
   - LLM protocols
   - Metadata protocols
3. Update all imports across codebase
4. Delete `src/pmd/search/ports.py`
5. Update `src/pmd/app/types.py` to re-export from `protocols.py` for backward compat

**Protocol Consolidation Map:**

| Current Location | Protocol | Consolidated Name |
|------------------|----------|-------------------|
| `app/types.py` | `EmbeddingGeneratorProtocol` | `EmbeddingGenerator` |
| `app/types.py` | `QueryExpanderProtocol` | `QueryExpander` |
| `app/types.py` | `DocumentRerankerProtocol` | `Reranker` |
| `search/ports.py` | `TextSearcher` | `TextSearcher` |
| `search/ports.py` | `VectorSearcher` | `VectorSearcher` |
| `search/ports.py` | `QueryExpander` | (merge with above) |
| `search/ports.py` | `Reranker` | (merge with above) |

### Phase 2: Rename `sources/metadata/` → `sources/transport.py`

**Goal:** Eliminate confusion between source transport metadata and document content metadata.

**Files to modify:**
- `src/pmd/sources/metadata/` → delete directory
- `src/pmd/sources/transport.py` → create single module

**Changes:**
1. Create `src/pmd/sources/transport.py` with:
   - `TransportMetadata` dataclass (ETags, last-modified, etc.)
   - Any transport-related utilities
2. Update imports in:
   - `src/pmd/sources/content/filesystem.py`
   - `src/pmd/services/loading.py`
   - `src/pmd/store/source_metadata.py`
3. Delete `src/pmd/sources/metadata/` directory

### Phase 3: Split `metadata/` Module

**Goal:** Clear separation between extraction (profiles) and ontology (tag hierarchy).

**Current structure:**
```
metadata/
├── extraction/     # Profile implementations
├── model/          # ExtractedMetadata
├── query/          # Tag matching
└── store/          # DocumentMetadataRepository
```

**New structure:**
```
extraction/         # Document metadata extraction
├── __init__.py
├── types.py        # ExtractedMetadata (from metadata/model/)
├── profiles/       # Profile implementations (from metadata/extraction/)
│   ├── generic.py
│   ├── obsidian.py
│   └── drafts.py
└── registry.py

ontology/           # Tag hierarchy and matching
├── __init__.py
├── model.py        # Ontology class (from metadata/ontology.py)
├── matcher.py      # LexicalTagMatcher (from metadata/query/)
└── booster.py      # OntologyMetadataBooster (from search/metadata/)
```

**Files to create:**
- `src/pmd/extraction/__init__.py`
- `src/pmd/extraction/types.py`
- `src/pmd/extraction/registry.py`
- `src/pmd/extraction/profiles/` (move existing)
- `src/pmd/ontology/__init__.py`
- `src/pmd/ontology/model.py`
- `src/pmd/ontology/matcher.py`
- `src/pmd/ontology/booster.py`

**Files to delete:**
- `src/pmd/metadata/` (entire directory after migration)
- `src/pmd/search/metadata/` (after moving booster)

### Phase 4: Consolidate Repositories in `store/`

**Goal:** All persistence in one place.

**Changes:**
1. Move `src/pmd/metadata/store/repository.py` → `src/pmd/store/repositories/metadata.py`
2. Reorganize `store/`:
   ```
   store/
   ├── __init__.py
   ├── database.py
   ├── models.py
   ├── schema.py
   ├── repositories/
   │   ├── __init__.py
   │   ├── collections.py    # from store/collections.py
   │   ├── documents.py      # from store/documents.py
   │   ├── content.py        # from store/content.py
   │   ├── embeddings.py     # from store/embeddings.py
   │   ├── fts.py            # from store/search.py
   │   ├── metadata.py       # from metadata/store/
   │   └── source_metadata.py # from store/source_metadata.py
   └── migrations/
   ```
3. Update all imports

### Phase 5: Extract `create_application()` to Separate File

**Goal:** Cleaner `app/__init__.py`, easier to understand wiring.

**Changes:**
1. Create `src/pmd/app/factory.py` with `create_application()`
2. Slim down `src/pmd/app/__init__.py` to just `Application` class
3. Update `__all__` exports

### Phase 6: Simplify Service Constructors

**Goal:** Reduce factory explosion by pre-assembling adapters.

**Current** (`SearchService.__init__`):
```python
def __init__(
    self,
    db,
    fts_repo,
    source_collection_repo,
    embedding_repo,
    embedding_generator_factory,    # Factory
    query_expander_factory,         # Factory
    reranker_factory,               # Factory
    tag_matcher_factory,            # Factory
    ontology_factory,               # Factory
    tag_retriever_factory,          # Factory
    metadata_repo_factory,          # Factory
    ...
):
```

**Proposed:**
```python
def __init__(
    self,
    pipeline: HybridSearchPipeline,  # Pre-assembled
    collection_repo: SourceCollectionRepositoryProtocol,
):
```

**Changes:**
1. Assemble `HybridSearchPipeline` with all adapters in `create_application()`
2. Simplify `SearchService` to receive assembled pipeline
3. Apply similar pattern to `IndexingService`

## Migration Strategy

Each phase can be completed independently and merged separately:

1. **Phase 1** (Protocols): Pure refactor, no behavior change
2. **Phase 2** (Transport): Rename + move, no behavior change
3. **Phase 3** (Split metadata): Structural change, no behavior change
4. **Phase 4** (Consolidate store): Move files, no behavior change
5. **Phase 5** (Factory extraction): Structural change, no behavior change
6. **Phase 6** (Simplify services): Interface change, requires careful testing

**Recommended order:** 1 → 2 → 4 → 3 → 5 → 6

Phases 1, 2, 4 are low-risk renames/moves. Phase 3 is a larger structural change. Phase 6 changes public interfaces.

## Backward Compatibility

### Import Compatibility Shims

For external consumers (if any), provide re-exports:

```python
# src/pmd/metadata/__init__.py (temporary shim)
from pmd.extraction import ExtractedMetadata, MetadataProfile
from pmd.ontology import Ontology, LexicalTagMatcher

__all__ = ["ExtractedMetadata", "MetadataProfile", "Ontology", "LexicalTagMatcher"]

import warnings
warnings.warn(
    "pmd.metadata is deprecated, use pmd.extraction and pmd.ontology",
    DeprecationWarning,
    stacklevel=2,
)
```

**Note:** Per CLAUDE.md instructions, backward compatibility shims should be avoided. Delete old modules entirely if this is internal-only code.

## Validation Criteria

After each phase:

1. All tests pass
2. `pmd` CLI commands work correctly
3. No circular import errors
4. Type checking passes (`mypy` or `pyright`)

## File Change Summary

### Files to Create
- `src/pmd/app/protocols.py`
- `src/pmd/app/factory.py`
- `src/pmd/sources/transport.py`
- `src/pmd/extraction/__init__.py`
- `src/pmd/extraction/types.py`
- `src/pmd/extraction/registry.py`
- `src/pmd/extraction/profiles/__init__.py`
- `src/pmd/ontology/__init__.py`
- `src/pmd/ontology/model.py`
- `src/pmd/ontology/matcher.py`
- `src/pmd/ontology/booster.py`
- `src/pmd/store/repositories/__init__.py`

### Files to Move
| From | To |
|------|-----|
| `metadata/extraction/*.py` | `extraction/profiles/*.py` |
| `metadata/model/types.py` | `extraction/types.py` |
| `metadata/ontology.py` | `ontology/model.py` |
| `metadata/query/*.py` | `ontology/matcher.py` |
| `metadata/store/repository.py` | `store/repositories/metadata.py` |
| `search/metadata/*.py` | `ontology/booster.py` |
| `sources/metadata/*.py` | `sources/transport.py` |
| `store/collections.py` | `store/repositories/collections.py` |
| `store/documents.py` | `store/repositories/documents.py` |
| `store/content.py` | `store/repositories/content.py` |
| `store/embeddings.py` | `store/repositories/embeddings.py` |
| `store/search.py` | `store/repositories/fts.py` |
| `store/source_metadata.py` | `store/repositories/source_metadata.py` |

### Files to Delete
- `src/pmd/search/ports.py` (merged into `app/protocols.py`)
- `src/pmd/metadata/` (entire directory after migration)
- `src/pmd/sources/metadata/` (entire directory after migration)
- `src/pmd/search/metadata/` (entire directory after migration)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Circular imports during migration | Complete each phase fully before starting next |
| Missed import updates | Use IDE "Find Usages" before deleting old paths |
| Test breakage | Run full test suite after each file move |
| External consumers break | N/A per CLAUDE.md (pre-alpha, breaking changes OK) |

## Future Considerations

### Backend Abstraction (Option B)

If multi-backend support becomes a priority:

1. Define `IndexBackend` protocol in `app/protocols.py`:
   ```python
   class IndexBackend(Protocol):
       async def add_documents(self, docs: Iterable[Document]) -> None: ...
       async def delete_documents(self, doc_ids: Iterable[str]) -> None: ...
       async def search(self, query: str, **kwargs) -> list[SearchResult]: ...
   ```

2. Move current SQLite implementation to `backends/sqlite/`
3. Create `backends/llamaindex/` as alternative

### Module Documentation

After reorganization, each top-level module should have a README:

```
extraction/README.md   # How to add new profiles
ontology/README.md     # Tag hierarchy and matching explained
store/README.md        # Repository patterns and schema
sources/README.md      # How to add new source types
```

## References

- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Ports and Adapters Pattern](https://herbertograca.com/2017/09/14/ports-adapters-architecture/)
- [Python Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
