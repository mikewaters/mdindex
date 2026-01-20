# Proposal: Refactor IngestPipeline to use LlamaIndex IngestionPipeline

## Summary

Refactor `src/idx/pipelines/ingest.py` to leverage `llama_index.core.ingestion.IngestionPipeline` instead of the current custom implementation, gaining built-in document deduplication, caching, and better integration with LlamaIndex's RAG ecosystem.

## Current State

The existing `IngestPipeline` class handles:
1. Dataset management (create/retrieve datasets in SQLite)
2. Document enumeration from `DirectorySource` / `ObsidianVaultSource`
3. Text normalization via `TextNormalizer`
4. Content hashing (`SHA256`) for change detection
5. CRUD operations on `Document` model via `DocumentRepository`
6. FTS5 indexing via `FTSManager`
7. Stale document detection and soft-deletion

**Key integration points:**
- `idx.store.models.Document` - SQLAlchemy ORM model
- `idx.store.fts.FTSManager` - FTS5 virtual table
- `idx.source.directory.DirectorySource` - File enumeration
- `idx.transform.normalize.TextNormalizer` - Content normalization

## LlamaIndex IngestionPipeline Capabilities

LlamaIndex's `IngestionPipeline` provides:

| Feature | LlamaIndex | Current idx |
|---------|------------|-------------|
| Document loading | `SimpleDirectoryReader` | `DirectorySource` |
| Change detection | `doc_id` + `document_hash` in docstore | `content_hash` in `Document` |
| Deduplication | `DocstoreStrategy.UPSERTS` | Manual path lookup |
| Transformations | `TransformComponent` chain | Single `TextNormalizer` |
| Caching | `IngestionCache` | None |
| Vector indexing | Native vector store integration | Not implemented yet |
| FTS indexing | Not built-in | Custom FTS5 via `FTSManager` |

## Proposed Architecture

### Option A: Full LlamaIndex Adoption (Recommended)

Replace the custom pipeline with LlamaIndex primitives, adapting our storage layer:

```
┌─────────────────────────────────────────────────────────────────┐
│                     IngestionPipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  SimpleDirectoryReader  ──►  TextNormalizer  ──►  FTSIndexer   │
│         │                    (TransformComponent)               │
│         ▼                                                       │
│  ObsidianReader (custom)                                        │
└─────────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  SQLDocStore    │         │  SQLiteVectorStore │
│  (custom)       │         │  (future)          │
└─────────────────┘         └─────────────────┘
         │
         ▼
┌─────────────────┐
│  FTSManager     │
│  (existing)     │
└─────────────────┘
```

**Components to implement:**

1. **`SQLDocStore`** - Custom `BaseDocumentStore` backed by our SQLite database
   - Maps LlamaIndex `Document` to `idx.store.models.Document`
   - Implements `add_documents`, `delete_document`, `get_document`, `document_exists`
   - Uses `content_hash` for deduplication

2. **`TextNormalizerTransform`** - Wrap `TextNormalizer` as a `TransformComponent`
   - Applies normalization to `Document.text`

3. **`FTSIndexerTransform`** - Custom `TransformComponent` for FTS indexing
   - Calls `FTSManager.upsert()` after document persistence

4. **`ObsidianReader`** - Custom reader extending `SimpleDirectoryReader`
   - Parses frontmatter for tags/aliases
   - Sets appropriate metadata

### Option B: Hybrid Approach

Keep our storage layer, use LlamaIndex only for transformation pipeline:

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent

class TextNormalizerTransform(TransformComponent):
    def __call__(self, nodes, **kwargs):
        normalizer = TextNormalizer()
        for node in nodes:
            node.text = normalizer.normalize(node.text)
        return nodes

# Use LlamaIndex pipeline for transformations only
pipeline = IngestionPipeline(
    transformations=[
        TextNormalizerTransform(),
        # Future: SentenceSplitter(), OpenAIEmbedding(), etc.
    ]
)

# Orchestrate manually
for source_doc in DirectorySource(path).enumerate():
    # Convert to LlamaIndex Document
    llama_doc = LlamaIndexDocument(text=source_doc.content, ...)

    # Run through LlamaIndex transformations
    nodes = pipeline.run(documents=[llama_doc])

    # Persist using existing infrastructure
    doc_repo.create(...)
    fts.upsert(...)
```

**Pros**: Minimal changes, incremental adoption
**Cons**: Doesn't leverage LlamaIndex's deduplication/caching

## Recommendation: Option A (Full Adoption)

Full adoption provides the most value:
1. Built-in deduplication eliminates manual hash checking
2. `IngestionCache` avoids reprocessing unchanged files
3. Clear path to vector search integration
4. Standard interface for future transformations (chunking, embeddings)

## Implementation Plan

### Phase 1: Core Integration

1. **Create `idx.store.llama.SQLDocStore`**
   - Implement `BaseDocumentStore` interface
   - Map between LlamaIndex `Document` and `idx.store.models.Document`
   - Support `DocstoreStrategy.UPSERTS`

2. **Create `idx.transform.llama.TextNormalizerTransform`**
   - Wrap existing `TextNormalizer` as `TransformComponent`

3. **Create `idx.transform.llama.FTSIndexerTransform`**
   - Post-process callback to update FTS5 index
   - Must run after document IDs are assigned

### Phase 2: Reader Adaptation

4. **Create `idx.source.llama.ObsidianReader`**
   - Extend `SimpleDirectoryReader` or implement custom `BaseReader`
   - Parse YAML frontmatter
   - Extract tags/aliases to metadata

5. **Adapt `DirectorySource` or use `SimpleDirectoryReader`**
   - `SimpleDirectoryReader` with `filename_as_id=True` may suffice
   - Evaluate glob pattern support

### Phase 3: Pipeline Refactor

6. **Refactor `IngestPipeline` to use `IngestionPipeline`**
   ```python
   class IngestPipeline:
       def __init__(self):
           self._docstore = SQLDocStore(get_session)
           self._cache = IngestionCache()

       def ingest_directory(self, config: IngestDirectoryConfig) -> IngestResult:
           reader = SimpleDirectoryReader(
               str(config.directory),
               filename_as_id=True,
               required_exts=self._patterns_to_exts(config.patterns),
           )

           pipeline = IngestionPipeline(
               transformations=[
                   TextNormalizerTransform(),
                   FTSIndexerTransform(self._fts),
               ],
               docstore=self._docstore,
               cache=self._cache,
               docstore_strategy=DocstoreStrategy.UPSERTS,
           )

           documents = reader.load_data()
           nodes = pipeline.run(documents=documents)

           return self._build_result(nodes)
   ```

### Phase 4: Stale Document Management (Separate from Ingestion)

7. **Add `check_stale_documents()` to `idx.core.status`**
   - Accepts dataset_id and source_path
   - Returns `ComponentStatus` with stale document count and paths
   - Does NOT modify data

8. **Add `delete_stale_documents()` to `idx.store.cleanup`**
   - Hard-deletes documents and their FTS entries
   - Convenience function with `dry_run` option
   - Intended for maintenance scripts/cron jobs

9. **Add `hard_delete_by_paths()` to `DocumentRepository`**
   - Complements existing `soft_delete_by_paths()`
   - Required for cleanup operations

### Phase 5: Testing & Migration

10. **Update tests**
    - Ensure existing behavior preserved
    - Add tests for LlamaIndex integration

11. **Migration path**
    - Existing `Document` records should work unchanged
    - May need one-time reindex for cache population

## API Changes

### Before
```python
pipeline = IngestPipeline()
result = pipeline.ingest_directory(IngestDirectoryConfig(
    directory=Path("/docs"),
    dataset_name="my-docs",
))
```

### After
```python
pipeline = IngestPipeline()
result = pipeline.ingest_directory(IngestDirectoryConfig(
    directory=Path("/docs"),
    dataset_name="my-docs",
    # New options available:
    enable_cache=True,
    transformations=["normalize", "chunk"],  # Optional
))
```

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    "llama-index-core>=0.11.0",
]
```

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LlamaIndex Document model mismatch | Implement adapter in `SQLDocStore` |
| FTS5 integration complexity | Use transform callback, not vector store |
| Performance regression | Benchmark before/after, leverage caching |
| Glob pattern incompatibility | Fall back to custom reader if needed |

## Design Decisions

1. **Dataset concept**: The `Document.dataset_id` foreign key is sufficient. The `SQLDocStore` receives `dataset_id` at construction and uses it when persisting documents. No special encoding needed.

2. **Hard-delete semantics**: Use LlamaIndex's default hard-delete behavior. Simplifies the implementation and aligns with standard LlamaIndex patterns.

3. **Stale document handling**: **Remove from ingestion path entirely.** Instead:
   - **Detection**: Add `check_stale_documents()` to `idx.core.status` to identify documents in DB that no longer exist in source
   - **Cleanup**: Add `delete_stale_documents()` to `idx.store.cleanup` for use in maintenance scripts/cron jobs

   This separation allows:
   - Ingestion to focus solely on create/update/skip
   - Operations teams to run cleanup on their own schedule
   - Status checks to surface staleness as a health metric

4. **Obsidian frontmatter**: Custom `ObsidianReader` extending `SimpleDirectoryReader` that parses YAML frontmatter into document metadata.

## Stale Document Architecture

### Status Check (`idx.core.status`)

```python
@dataclass
class StaleDocumentInfo:
    """Information about stale documents in a dataset."""
    dataset_id: int
    dataset_name: str
    stale_count: int
    stale_paths: list[str]

def check_stale_documents(
    dataset_id: int,
    source_path: Path,
    patterns: list[str] | None = None,
) -> ComponentStatus:
    """Check for documents in DB that no longer exist in source.

    Compares indexed document paths against current source enumeration.

    Returns:
        ComponentStatus with stale document details.
    """
    # Enumerate current source files
    source = DirectorySource(source_path, patterns=patterns)
    source_paths = {doc.relative_path for doc in source.enumerate()}

    # Get indexed paths from DB
    with get_session() as session:
        repo = DocumentRepository(session)
        indexed_paths = repo.list_paths_by_dataset(dataset_id, active_only=True)

    # Stale = in DB but not in source
    stale_paths = indexed_paths - source_paths

    if stale_paths:
        return ComponentStatus(
            name="stale_documents",
            healthy=False,  # or True with warning?
            message=f"Found {len(stale_paths)} stale documents",
            details={"stale_paths": list(stale_paths)[:100]},  # limit for display
        )
    return ComponentStatus(
        name="stale_documents",
        healthy=True,
        message="No stale documents found",
    )
```

### Cleanup (`idx.store.cleanup`)

```python
class IndexCleanup:
    # ... existing methods ...

    def delete_stale_documents(
        self,
        dataset_id: int,
        stale_paths: set[str],
    ) -> int:
        """Hard-delete stale documents and their FTS entries.

        Args:
            dataset_id: Dataset to clean up.
            stale_paths: Paths of documents to delete.

        Returns:
            Number of documents deleted.
        """
        if not stale_paths:
            return 0

        doc_repo = DocumentRepository(self._session)

        # Get doc IDs for FTS cleanup
        doc_ids = []
        for path in stale_paths:
            doc = doc_repo.get_by_path(dataset_id, path)
            if doc:
                doc_ids.append(doc.id)

        # Delete FTS entries
        self.cleanup_fts_for_documents(doc_ids)

        # Hard-delete documents
        deleted = doc_repo.hard_delete_by_paths(dataset_id, stale_paths)

        logger.info(f"Deleted {deleted} stale documents from dataset {dataset_id}")
        return deleted


def cleanup_stale_documents(
    dataset_id: int,
    source_path: Path,
    patterns: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Convenience function: detect and delete stale documents.

    Args:
        dataset_id: Dataset to clean up.
        source_path: Path to source directory.
        patterns: Glob patterns for source enumeration.
        dry_run: If True, only report what would be deleted.

    Returns:
        Number of documents deleted (or would be deleted if dry_run).
    """
    # Detect stale
    source = DirectorySource(source_path, patterns=patterns)
    source_paths = {doc.relative_path for doc in source.enumerate()}

    with get_session() as session:
        repo = DocumentRepository(session)
        indexed_paths = repo.list_paths_by_dataset(dataset_id, active_only=True)
        stale_paths = indexed_paths - source_paths

        if dry_run:
            logger.info(f"Would delete {len(stale_paths)} stale documents")
            return len(stale_paths)

        cleanup = IndexCleanup(session)
        return cleanup.delete_stale_documents(dataset_id, stale_paths)
```

## Conclusion

Adopting LlamaIndex's `IngestionPipeline` provides a solid foundation for:
- Deduplication and change detection
- Future vector search integration
- Standardized transformation pipeline
- Caching for incremental ingestion

The main work is implementing `SQLDocStore` to bridge LlamaIndex's document model with our existing SQLite storage layer. The transformation components are straightforward wrappers around existing code.
