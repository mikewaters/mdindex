# Implementation Plan: Isolate Retrieval Pipeline from Infrastructure

## Overview

Refactor `HybridSearchPipeline` to depend only on abstract ports rather than concrete storage/LLM implementations. This creates a pure retrieval algorithm that is testable, swappable, and infrastructure-agnostic.

## Current State Analysis

### Direct Infrastructure Dependencies in Pipeline

1. **FTS Search** (Line 353): `self.fts_repo.search()` - direct repository call
2. **Vector Search** (Line 368): `self.embedding_generator.embedding_repo.search_vectors()` - leaky abstraction through embedding generator
3. **Tag Search** (Line 393): `self.tag_retriever.search()` - already port-like
4. **Query Embedding** (Line 366): `self.embedding_generator.embed_query()` - LLM call
5. **Query Expansion** (Line 309): `self.query_expander.expand()` - LLM call
6. **Reranking** (Line 448): `self.reranker.get_rerank_scores()` - LLM call
7. **Metadata Boost** (Lines 504-508): Direct DB queries via `build_path_to_id_map()` and `get_document_tags_batch()`

### Key Insight

Some components already behave like ports (`tag_retriever`, `query_expander`, `reranker`), but:
- `embedding_generator` exposes its internal `embedding_repo` (leaky)
- `_apply_metadata_boost` directly queries the database
- No formal Protocol definitions for testability

## Target Architecture

```
                    ┌──────────────────────────────────────┐
                    │       HybridSearchPipeline           │
                    │  (Pure algorithm, no infrastructure) │
                    └──────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
    ┌───────────┐             ┌───────────────┐           ┌───────────────┐
    │   Ports   │             │    Fusion     │           │    Config     │
    │(Protocols)│             │  (Pure funcs) │           │  (Dataclass)  │
    └───────────┘             └───────────────┘           └───────────────┘
          │
    ┌─────┴─────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │           │             │             │             │             │
    ▼           ▼             ▼             ▼             ▼             ▼
TextSearcher VectorSearcher TagSearcher QueryExpander Reranker MetadataBooster
```

## Port Definitions

### 1. TextSearcher (FTS)
```python
class TextSearcher(Protocol):
    """Full-text search capability."""
    def search(
        self,
        query: str,
        limit: int,
        collection_id: int | None = None,
    ) -> list[SearchResult]: ...
```

### 2. VectorSearcher
```python
class VectorSearcher(Protocol):
    """Vector similarity search capability."""
    async def search(
        self,
        query: str,
        limit: int,
        collection_id: int | None = None,
    ) -> list[SearchResult]: ...
```
Note: Takes raw query string, handles embedding internally.

### 3. TagSearcher
```python
class TagSearcher(Protocol):
    """Tag-based document retrieval."""
    def search(
        self,
        tags: dict[str, float] | set[str],
        limit: int,
        collection_id: int | None = None,
    ) -> list[SearchResult]: ...
```

### 4. QueryExpander
```python
class QueryExpander(Protocol):
    """Query expansion via LLM or other means."""
    async def expand(
        self,
        query: str,
        num_variations: int = 2,
    ) -> list[str]: ...
```

### 5. Reranker
```python
class Reranker(Protocol):
    """Document reranking capability."""
    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]: ...
```

### 6. MetadataBooster
```python
class MetadataBooster(Protocol):
    """Boost scores based on metadata/tag matches."""
    def boost(
        self,
        results: list[SearchResult],
        query_tags: dict[str, float],
    ) -> list[tuple[SearchResult, BoostInfo]]: ...
```
Note: Encapsulates all DB lookups for tag matching internally.

### 7. TagInferencer (Supporting)
```python
class TagInferencer(Protocol):
    """Infer tags from query text."""
    def infer_tags(self, query: str) -> set[str]: ...
    def expand_tags(self, tags: set[str]) -> dict[str, float]: ...
```

## Implementation Steps

### Phase 1: Define Ports (src/pmd/search/ports.py)

Create Protocol definitions for all six ports plus supporting types:
- `TextSearcher`, `VectorSearcher`, `TagSearcher`
- `QueryExpander`, `Reranker`, `MetadataBooster`
- `TagInferencer` (combines matcher + ontology)
- Supporting result types: `RerankResult`, `BoostInfo`

### Phase 2: Create Adapters (src/pmd/search/adapters/)

Implement adapters that wrap existing infrastructure:

1. **FTS5TextSearcher** - Wraps `FTS5SearchRepository`
2. **EmbeddingVectorSearcher** - Wraps `EmbeddingGenerator` + `EmbeddingRepository`
3. **TagRetrieverSearcher** - Wraps existing `TagRetriever`
4. **LLMQueryExpander** - Wraps existing `QueryExpander` (may just re-export)
5. **LLMReranker** - Wraps existing `DocumentReranker`
6. **OntologyMetadataBooster** - New: encapsulates `metadata_repo`, `ontology`, DB lookups
7. **LexicalTagInferencer** - Wraps `LexicalTagMatcher` + `Ontology`

### Phase 3: Refactor HybridSearchPipeline

Update constructor to accept ports:
```python
def __init__(
    self,
    text_searcher: TextSearcher,
    vector_searcher: VectorSearcher | None = None,
    tag_searcher: TagSearcher | None = None,
    query_expander: QueryExpander | None = None,
    reranker: Reranker | None = None,
    metadata_booster: MetadataBooster | None = None,
    tag_inferencer: TagInferencer | None = None,
    config: SearchPipelineConfig | None = None,
):
```

Remove all direct DB/repo calls from pipeline methods:
- `_parallel_search`: Use ports instead of repos
- `_apply_metadata_boost`: Delegate to `MetadataBooster` port
- `_expand_query`: Use `QueryExpander` port (already clean)
- `_rerank_with_blending`: Use `Reranker` port (already clean)

### Phase 4: Update SearchService Wiring

Move adapter creation to `SearchService.hybrid_search()`:
```python
async def hybrid_search(self, query: str, ...):
    # Create adapters from container resources
    text_searcher = FTS5TextSearcher(self._container.fts_repo)
    vector_searcher = EmbeddingVectorSearcher(
        await self._container.get_embedding_generator()
    )
    tag_searcher = TagRetrieverSearcher(
        self._container.db,
        self._container.metadata_repo,
    )
    # ... etc

    pipeline = HybridSearchPipeline(
        text_searcher=text_searcher,
        vector_searcher=vector_searcher,
        # ...
    )
```

### Phase 5: Create In-Memory Fakes (tests/fakes/)

Implement test doubles for contract testing:

1. **InMemoryTextSearcher** - Returns pre-configured results
2. **InMemoryVectorSearcher** - Simulates embedding + similarity
3. **InMemoryTagSearcher** - Tag-based lookup from dict
4. **StubQueryExpander** - Returns configured variations
5. **StubReranker** - Returns configured scores
6. **InMemoryMetadataBooster** - Boost from in-memory tag map

### Phase 6: Add Contract Tests

Create `tests/unit/search/test_pipeline_contracts.py`:
- Test pipeline behavior with various combinations of fakes
- Test fusion algorithm correctness
- Test metadata boost integration
- Test score normalization
- Test error handling (missing searchers, failed LLM calls)

## Configuration Documentation

Document default weights in `SearchPipelineConfig`:
```python
@dataclass
class SearchPipelineConfig:
    """Configuration for hybrid search pipeline.

    Weights:
        fts_weight: Weight for full-text search in RRF (default: 1.0)
        vec_weight: Weight for vector search in RRF (default: 1.0)
        tag_weight: Weight for tag search in RRF (default: 0.8)
        rrf_k: RRF smoothing constant (default: 60, higher = more uniform)

    Expansion:
        expansion_weight: Weight for expanded query variations (default: 0.5)

    Reranking:
        rerank_candidates: Number of candidates for LLM reranking (default: 30)

    Metadata:
        metadata_boost_factor: Multiplicative boost for tag matches (default: 1.15)
        metadata_max_boost: Maximum total boost cap (default: 2.0)
    """
```

## File Structure After Refactor

```
src/pmd/search/
├── __init__.py
├── ports.py              # NEW: Protocol definitions
├── adapters/             # NEW: Infrastructure adapters
│   ├── __init__.py
│   ├── text.py           # FTS5TextSearcher
│   ├── vector.py         # EmbeddingVectorSearcher
│   ├── tag.py            # TagRetrieverSearcher
│   ├── expansion.py      # LLMQueryExpander
│   ├── rerank.py         # LLMReranker
│   └── boost.py          # OntologyMetadataBooster
├── pipeline.py           # MODIFIED: Uses ports only
├── fusion.py             # UNCHANGED: Pure functions
├── scoring.py            # UNCHANGED: Pure functions
├── chunking.py           # UNCHANGED
└── text.py               # UNCHANGED

tests/
├── fakes/                # NEW: Test doubles
│   ├── __init__.py
│   └── search.py         # In-memory search fakes
└── unit/search/
    └── test_pipeline_contracts.py  # NEW: Contract tests
```

## Backwards Compatibility

- Existing `SearchService` consumers unchanged
- `HybridSearchPipeline` constructor signature changes (breaking for direct users)
- Add factory function for legacy compatibility if needed:
  ```python
  def create_pipeline_from_repos(fts_repo, embedding_gen, ...) -> HybridSearchPipeline:
      """Legacy factory - creates pipeline with adapter wiring."""
  ```

## Success Criteria

1. `HybridSearchPipeline` has zero imports from `pmd.store.*` or `pmd.llm.*`
2. All infrastructure access happens through Protocol-typed parameters
3. Contract tests pass with in-memory fakes (no DB/LLM needed)
4. Existing integration tests still pass
5. Configuration defaults documented in docstrings
