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
│   └── types.py            # Repository, LLM, metadata, config protocols (21 protocols)
├── cli/                    # CLI (clear purpose ✓)
│   └── commands/           # CLI command implementations
├── core/                   # Config, exceptions, types (clear ✓)
│   ├── config.py
│   ├── exceptions.py
│   ├── types.py
│   └── instrumentation.py
├── llm/                    # LLM providers (clear ✓)
├── metadata/               # Document metadata extraction + ontology + storage
│   ├── extraction/         # Profile implementations (generic, obsidian, drafts)
│   ├── model/              # ExtractedMetadata types + ontology
│   │   ├── data/           # JSON data files (tag_aliases.json, tag_ontology.json)
│   │   ├── ontology.py     # ⚠️ DUPLICATE: also at metadata/ontology.py
│   │   └── aliases.py      # ⚠️ DUPLICATE: also at metadata/aliases.py
│   ├── data/               # ⚠️ DUPLICATE: same JSON files as model/data/
│   ├── ontology.py         # ⚠️ DUPLICATE: also in model/
│   ├── aliases.py          # ⚠️ DUPLICATE: also in model/
│   ├── query/              # Tag matching (inference, retrieval, scoring)
│   └── store/              # DocumentMetadataRepository (why here, not in store/?)
├── search/                 # Search pipeline + ports + adapters
│   ├── adapters/           # Port implementations (text, vector, tag, rerank, expansion, boost)
│   ├── metadata/           # Metadata boosting (inference, retrieval, scoring)
│   ├── ports.py            # Search-specific protocols (8 protocols)
│   ├── pipeline.py         # HybridSearchPipeline orchestration
│   ├── fusion.py           # RRF result fusion
│   ├── chunking.py         # Document chunking
│   ├── scoring.py          # Scoring utilities
│   ├── text.py             # Text search utilities
│   ├── dspy_rag.py         # DSPy RAG integration
│   ├── dspy_modules.py     # DSPy modules
│   └── dspy_retriever.py   # DSPy retriever
├── services/               # High-level orchestration
│   ├── loading.py          # Document loading
│   ├── loading_llamaindex.py # LlamaIndex loader
│   ├── indexing.py         # Indexing service
│   ├── search.py           # Search service
│   ├── status.py           # Status service
│   └── caching.py          # Caching utilities
├── sources/                # Document sources
│   ├── content/            # DocumentSource protocol + filesystem
│   │   ├── base.py
│   │   ├── filesystem.py
│   │   ├── glob_matcher.py
│   │   ├── llamaindex.py
│   │   └── registry.py
│   └── metadata/           # ⚠️ CONFUSING: Contains extraction profiles, NOT transport metadata
│       ├── types.py        # ⚠️ DUPLICATE: ExtractedMetadata, MetadataProfile (same as metadata/model/)
│       ├── base.py         # ⚠️ DUPLICATE: base profile
│       ├── obsidian.py     # ⚠️ DUPLICATE: identical to metadata/extraction/obsidian.py
│       ├── drafts.py       # ⚠️ DUPLICATE: identical to metadata/extraction/drafts.py
│       ├── parsing.py      # ⚠️ DUPLICATE: identical to metadata/extraction/parsing.py
│       └── registry.py     # ⚠️ DUPLICATE: profile registry
├── store/                  # SQLite persistence
│   ├── database.py
│   ├── models.py
│   ├── schema.py
│   ├── collections.py      # SourceCollectionRepository
│   ├── documents.py        # DocumentRepository
│   ├── content.py          # ContentRepository
│   ├── embeddings.py       # EmbeddingRepository
│   ├── search.py           # FTS5SearchRepository
│   ├── source_metadata.py  # SourceMetadataRepository
│   ├── vector_search.py    # Vector search operations
│   ├── migrate.py          # Custom migration orchestrator
│   ├── alembic/            # ⚠️ OLD: Alembic-based migrations (DELETE)
│   └── migrations/         # NEW: Custom migration system (KEEP)
├── utils/                  # Miscellaneous
│   └── hashing.py
└── workflows/              # Pipelines
    ├── contracts.py        # ChunkerProtocol
    └── pipelines/          # Ingestion, embedding pipelines
```

### Critical Duplication Found

The codebase contains **significant code duplication** that must be resolved BEFORE the reorganization:

| Duplicate | Location 1 | Location 2 | Resolution |
|-----------|-----------|-----------|------------|
| `ExtractedMetadata` | `metadata/model/types.py` | `sources/metadata/types.py` | Keep `metadata/model/`, delete `sources/metadata/` |
| `MetadataProfile` | `metadata/model/types.py` | `sources/metadata/types.py` | Keep `metadata/model/`, delete `sources/metadata/` |
| `ObsidianProfile` | `metadata/extraction/obsidian.py` | `sources/metadata/obsidian.py` | Keep `metadata/extraction/`, delete `sources/metadata/` |
| `DraftsProfile` | `metadata/extraction/drafts.py` | `sources/metadata/drafts.py` | Keep `metadata/extraction/`, delete `sources/metadata/` |
| `parsing.*` | `metadata/extraction/parsing.py` | `sources/metadata/parsing.py` | Keep `metadata/extraction/`, delete `sources/metadata/` |
| `ScoredResult` | `search/metadata/scoring.py` | `metadata/query/scoring.py` | Consolidate to single location |
| `SupportsLoadData` | `sources/content/llamaindex.py` | `services/loading_llamaindex.py` | Consolidate to `sources/content/` |
| `ontology.py` | `metadata/ontology.py` | `metadata/model/ontology.py` | Keep `metadata/model/` |
| `aliases.py` | `metadata/aliases.py` | `metadata/model/aliases.py` | Keep `metadata/model/` |
| `tag_*.json` | `metadata/data/` | `metadata/model/data/` | Keep `metadata/model/data/` |

### Specific Confusions

| Question | Current Answer | Problem |
|----------|----------------|---------|
| Where do I add a metadata extractor? | `metadata/extraction/` | But `sources/metadata/` has duplicate implementations |
| Where is storage logic? | `store/` | But `metadata/store/` has DocumentMetadataRepository |
| Which protocols do I implement? | `app/types.py` | `search/ports.py` has **complementary** (not duplicate) protocols for search pipeline |
| How do I add a vector store backend? | `store/embeddings.py`? `search/adapters/`? | No clear extension point |
| What's `sources/metadata/`? | **Duplicated extraction code** | Despite the name, it's NOT transport metadata—it's duplicate profile code |
| Where's the DSPy integration? | `search/dspy_*.py` | Undocumented, no clear placement strategy |
| Which migration system? | `store/alembic/` or `store/migrations/`? | Two parallel systems exist |

### Protocol Clarification

The original analysis incorrectly claimed `app/types.py` and `search/ports.py` contain duplicates. In fact:

- **`app/types.py`** (21 protocols): Infrastructure layer—repositories, LLM providers, configuration
- **`search/ports.py`** (8 protocols): Search layer—pipeline ports for text/vector/tag search, reranking, boosting

These are **complementary**, not duplicated. Both should be consolidated into `app/protocols.py` for organizational clarity, but they serve different architectural layers.

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

### Phase 0: Eliminate Code Duplication (PREREQUISITE)

**Goal:** Remove duplicate code before reorganization to avoid moving dead code.

**Rationale:** The codebase has significant duplication that must be eliminated first. Moving duplicated code will create confusion about which is canonical.

**Files to delete:**
- `src/pmd/sources/metadata/` (entire directory) - duplicates `metadata/extraction/`
- `src/pmd/metadata/data/` - duplicates `metadata/model/data/`
- `src/pmd/metadata/ontology.py` (top-level) - duplicates `metadata/model/ontology.py`
- `src/pmd/metadata/aliases.py` (top-level) - duplicates `metadata/model/aliases.py`

**Files to consolidate:**
- `ScoredResult` protocol: Keep in `search/metadata/scoring.py`, delete from `metadata/query/scoring.py`
- `SupportsLoadData` protocol: Keep in `sources/content/llamaindex.py`, update `services/loading_llamaindex.py` to import from there

**Migration steps:**
1. Update all imports that reference `sources/metadata/` to use `metadata/extraction/` and `metadata/model/`
2. Verify no code references the deleted files
3. Delete the duplicate files/directories
4. Run tests to confirm nothing broke

**Affected imports:**
```python
# BEFORE (delete these patterns)
from pmd.sources.metadata.types import ExtractedMetadata, MetadataProfile
from pmd.sources.metadata.obsidian import ObsidianProfile
from pmd.sources.metadata.drafts import DraftsProfile
from pmd.sources.metadata.parsing import parse_frontmatter

# AFTER (canonical locations)
from pmd.metadata.model import ExtractedMetadata
from pmd.metadata.model.types import MetadataProfile
from pmd.metadata.extraction.obsidian import ObsidianProfile
from pmd.metadata.extraction.drafts import DraftsProfile
from pmd.metadata.extraction.parsing import parse_frontmatter
```

### Phase 1: Consolidate Protocols

**Goal:** Single source of truth for all protocols, organized by domain.

**Files to modify:**
- `src/pmd/app/types.py` → `src/pmd/app/protocols.py`
- `src/pmd/search/ports.py` → merge into `app/protocols.py`

**Changes:**
1. Create `src/pmd/app/protocols.py` with all protocol definitions
2. Organize protocols by domain:
   - Storage protocols (repositories) - from `app/types.py`
   - Search protocols (pipeline ports) - from `search/ports.py`
   - LLM protocols - from `app/types.py`
   - Metadata protocols - from `app/types.py`
3. Update all imports across codebase
4. Delete `src/pmd/search/ports.py`
5. Delete `src/pmd/app/types.py` entirely (no backward compat per CLAUDE.md)

**Protocol Consolidation Map:**

| Current Location | Protocol | Section in `protocols.py` |
|------------------|----------|---------------------------|
| `app/types.py` | `DatabaseProtocol` | Storage |
| `app/types.py` | `SourceCollectionRepositoryProtocol` | Storage |
| `app/types.py` | `DocumentRepositoryProtocol` | Storage |
| `app/types.py` | `FTSRepositoryProtocol` | Storage |
| `app/types.py` | `EmbeddingRepositoryProtocol` | Storage |
| `app/types.py` | `DocumentMetadataRepositoryProtocol` | Storage |
| `app/types.py` | `LLMProviderProtocol` | LLM |
| `app/types.py` | `EmbeddingGeneratorProtocol` | LLM |
| `app/types.py` | `QueryExpanderProtocol` | LLM |
| `app/types.py` | `DocumentRerankerProtocol` | LLM |
| `app/types.py` | `TagMatcherProtocol` | Metadata |
| `app/types.py` | `OntologyProtocol` | Metadata |
| `app/types.py` | `TagRetrieverProtocol` | Metadata |
| `app/types.py` | `SearchConfigProtocol` | Config |
| `app/types.py` | `ConfigProtocol` | Config |
| `app/types.py` | `LoadingServiceProtocol` | Services |
| `search/ports.py` | `TextSearcher` | Search Pipeline |
| `search/ports.py` | `VectorSearcher` | Search Pipeline |
| `search/ports.py` | `TagSearcher` | Search Pipeline |
| `search/ports.py` | `QueryExpander` | Search Pipeline |
| `search/ports.py` | `Reranker` | Search Pipeline |
| `search/ports.py` | `MetadataBooster` | Search Pipeline |
| `search/ports.py` | `TagInferencer` | Search Pipeline |

**Note:** `QueryExpander` and `Reranker` appear in both files with slightly different signatures. Unify to the `search/ports.py` versions (async, pipeline-focused).

### Phase 2: (Absorbed into Phase 0)

The original Phase 2 was to rename `sources/metadata/` to `sources/transport.py`. However, analysis revealed that `sources/metadata/` contains **duplicate extraction profile code**, not transport metadata.

The directory is deleted entirely in Phase 0. No `transport.py` is needed—there is no actual transport metadata code to preserve.

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

**Goal:** All persistence in one place, single migration system.

**Changes:**

1. **Delete Alembic migration system:**
   - Delete `src/pmd/store/alembic/` directory entirely
   - Keep `src/pmd/store/migrations/` as the canonical migration system
   - Keep `src/pmd/store/migrate.py` as the migration runner

2. Move `src/pmd/metadata/store/repository.py` → `src/pmd/store/repositories/metadata.py`

3. Reorganize `store/`:
   ```
   store/
   ├── __init__.py
   ├── database.py
   ├── models.py
   ├── schema.py
   ├── migrate.py            # Migration runner
   ├── vector_search.py      # Vector search operations
   ├── repositories/
   │   ├── __init__.py
   │   ├── collections.py    # from store/collections.py
   │   ├── documents.py      # from store/documents.py
   │   ├── content.py        # from store/content.py
   │   ├── embeddings.py     # from store/embeddings.py
   │   ├── fts.py            # from store/search.py
   │   ├── metadata.py       # from metadata/store/
   │   └── source_metadata.py # from store/source_metadata.py
   └── migrations/           # CANONICAL: Custom migration system
       ├── runner.py
       └── versions/
           ├── v0001_initial_schema.py
           └── v0002_rename_source_collections.py
   ```

4. Update all imports

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

### Phase 7: Update Tests

**Goal:** Update test imports and organization to match new module structure.

**Test files requiring import updates (~25 files):**

| Category | Files | Import Changes |
|----------|-------|----------------|
| Metadata extraction | `tests/pmd/unit/metadata/extraction/test_*.py` | `pmd.metadata.*` → `pmd.extraction.*` |
| Ontology | `tests/pmd/unit/search/test_ontology.py` | `pmd.metadata.*` → `pmd.ontology.*` |
| Search metadata | `tests/pmd/unit/search/metadata/test_*.py` | `pmd.search.metadata.*` → `pmd.ontology.*` |
| Protocols | `tests/pmd/fakes/search.py` | `pmd.search.ports` → `pmd.app.protocols` |
| Repositories | `tests/pmd/conftest.py`, `tests/pmd/integration/*.py` | `pmd.store.*` → `pmd.store.repositories.*` |

**Critical conftest.py updates:**

1. `tests/pmd/conftest.py`:
   ```python
   # BEFORE
   from pmd.store.collections import CollectionRepository
   from pmd.store.documents import DocumentRepository
   from pmd.store.embeddings import EmbeddingRepository
   from pmd.store.search import FTS5SearchRepository

   # AFTER
   from pmd.store.repositories.collections import CollectionRepository
   from pmd.store.repositories.documents import DocumentRepository
   from pmd.store.repositories.embeddings import EmbeddingRepository
   from pmd.store.repositories.fts import FTS5SearchRepository
   ```

2. `tests/pmd/integration/conftest.py`: Same pattern

3. `tests/pmd/fakes/search.py`:
   ```python
   # BEFORE
   from pmd.search.ports import BoostInfo, RerankScore

   # AFTER
   from pmd.app.protocols import BoostInfo, RerankScore
   ```

**Test directory structure:**
Keep current test directory structure (`tests/pmd/unit/metadata/`). Only update imports—do not reorganize test directories.

**Execution order:**
1. Update `conftest.py` files first (fixtures used across tests)
2. Update `fakes/` modules
3. Update unit tests
4. Update integration tests
5. Run full test suite

## Migration Strategy

Each phase can be completed independently and merged separately:

0. **Phase 0** (Duplication): Delete duplicate code—**MUST be first**
1. **Phase 1** (Protocols): Pure refactor, no behavior change
2. **Phase 2** (Absorbed): No work needed
3. **Phase 3** (Split metadata): Structural change, no behavior change
4. **Phase 4** (Consolidate store): Move files + delete Alembic, no behavior change
5. **Phase 5** (Factory extraction): Structural change, no behavior change
6. **Phase 6** (Simplify services): Interface change, requires careful testing
7. **Phase 7** (Tests): Import updates—**run after each phase that moves modules**

**Recommended order:** 0 → 1 → 4 → 3 → 5 → 6

- Phase 0 eliminates dead code before any moves
- Phases 1, 4 are low-risk renames/moves
- Phase 3 is a larger structural change
- Phase 6 changes public interfaces
- Phase 7 should be executed incrementally after each phase, not as a batch at the end

## Backward Compatibility

Per CLAUDE.md: This is a pre-alpha product. **No backward compatibility shims.** Delete old modules entirely after migration.

Do NOT create re-export shims like `pmd.metadata.__init__.py` pointing to new locations. Update all imports directly.

## Validation Criteria

After each phase:

1. All tests pass
2. `pmd` CLI commands work correctly
3. No circular import errors
4. Type checking passes (`mypy` or `pyright`)

## File Change Summary

### Files to Create
- `src/pmd/app/protocols.py` - consolidated protocols from `app/types.py` + `search/ports.py`
- `src/pmd/app/factory.py` - `create_application()` extracted from `app/__init__.py`
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
| `metadata/model/ontology.py` | `ontology/model.py` |
| `metadata/model/data/*.json` | `ontology/data/*.json` |
| `metadata/query/*.py` | `ontology/matcher.py` |
| `metadata/store/repository.py` | `store/repositories/metadata.py` |
| `search/metadata/*.py` | `ontology/booster.py` |
| `store/collections.py` | `store/repositories/collections.py` |
| `store/documents.py` | `store/repositories/documents.py` |
| `store/content.py` | `store/repositories/content.py` |
| `store/embeddings.py` | `store/repositories/embeddings.py` |
| `store/search.py` | `store/repositories/fts.py` |
| `store/source_metadata.py` | `store/repositories/source_metadata.py` |

### Files to Delete (Phase 0 - Duplicates)
- `src/pmd/sources/metadata/` (entire directory - duplicates `metadata/extraction/`)
- `src/pmd/metadata/data/` (duplicates `metadata/model/data/`)
- `src/pmd/metadata/ontology.py` (top-level - duplicates `metadata/model/ontology.py`)
- `src/pmd/metadata/aliases.py` (top-level - duplicates `metadata/model/aliases.py`)

### Files to Delete (After Migration)
- `src/pmd/app/types.py` (merged into `app/protocols.py`)
- `src/pmd/search/ports.py` (merged into `app/protocols.py`)
- `src/pmd/metadata/` (entire directory after migration to `extraction/` + `ontology/`)
- `src/pmd/search/metadata/` (after moving to `ontology/`)
- `src/pmd/store/alembic/` (entire directory - superseded by `migrations/`)

## DSPy Integration Placement

The `search/` module contains DSPy integration files not addressed in the original structure:

| File | Purpose | Placement Decision |
|------|---------|-------------------|
| `search/dspy_rag.py` | DSPy RAG integration | Keep in `search/` - search-specific |
| `search/dspy_modules.py` | DSPy modules | Keep in `search/` - search-specific |
| `search/dspy_retriever.py` | DSPy retriever | Keep in `search/` - search-specific |

**Rationale:** These files are tightly coupled to search functionality. They should remain in `search/` but could be grouped into a `search/dspy/` subdirectory if they grow:

```
search/
├── dspy/              # Optional: group if files grow
│   ├── __init__.py
│   ├── rag.py
│   ├── modules.py
│   └── retriever.py
├── pipeline.py
├── fusion.py
└── ...
```

**Action:** No immediate change required. Monitor for growth.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Circular imports during migration | Complete each phase fully before starting next |
| Missed import updates | Use IDE "Find Usages" before deleting old paths |
| Test breakage | Run full test suite after each file move |
| External consumers break | N/A per CLAUDE.md (pre-alpha, breaking changes OK) |
| Duplicate code not fully identified | Phase 0 includes verification step before deletion |

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
