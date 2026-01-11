# Caching Architecture Proposal

## Context
Currently, the `IngestionPipeline` reads documents directly from their source (e.g., local filesystem) and stores their content in the database. The `source_metadata` table stores the URI of the original source.

We want to introduce a "caching" stage to the ingestion pipeline. This stage will copy the ingested files to a local cache directory. The system will then use these cached files as the primary reference for the document, updating the `source_uri` in the metadata to point to the cached location (using a `file:///` URI scheme).

## Goals
1.  **Local Cache:** Maintain a local copy of all ingested files.
2.  **URI Update:** Update document metadata to point to the cached file instead of the original source.
3.  **fsspec Integration:** Use `fsspec` for filesystem operations to abstract the storage details.
4.  **Configuration:** Allow configuration of the cache base path.

## Architecture

### 1. Configuration
A new configuration section `[cache]` will be added to `pmd.toml` and the `Config` object.

```toml
[cache]
base_path = "~/.cache/pmd/files"
```

The `Config` dataclass will be updated to include a `CacheConfig` object:

```python
@dataclass
class CacheConfig:
    base_path: str
```

### 2. Cache Structure
The cache will be organized by collection name to avoid collisions and maintain organization.

```
<base_path>/
  ├── <collection_name_1>/
  │   ├── document1.md
  │   └── subfolder/
  │       └── document2.md
  └── <collection_name_2>/
      └── ...
```

### 3. Ingestion Pipeline Modification

The caching logic should be integrated into the `IngestionPipeline` or the `LoadingService`. Given that the `LoadingService` yields `LoadedDocument` objects which are then persisted by the `IngestionPipeline`, intercepting the flow in the `IngestionPipeline` (before persistence) seems appropriate, or decorating the `LoadingService`.

However, modifying the `LoadingService` might be cleaner if we consider "loading" to include "fetching and caching".

**Proposed Flow:**

1.  **Load:** `LoadingService` fetches the document from the original source.
2.  **Cache:** A new component or step takes the `LoadedDocument`.
    *   Constructs the target path: `<cache_base>/<collection_name>/<doc.path>`.
    *   Writes `doc.content` to this target path using `fsspec`.
    *   Updates `doc.ref.uri` (or a new field) to `file://<target_path>`.
3.  **Persist:** The `IngestionPipeline` persists the document. The `SourceMetadata` saved to the database will now reflect the `file://` URI of the cached file.

### 4. Implementation Details

#### Dependencies
*   Add `fsspec` to `pyproject.toml`.

#### New Component: `DocumentCacher` (Optional but recommended for separation of concerns)
A service or utility class responsible for writing content to the cache.

```python
class DocumentCacher:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.fs = fsspec.filesystem("file")

    def cache_document(self, collection_name: str, doc: LoadedDocument) -> str:
        # Determine path
        # Write content
        # Return new URI
        
    def remove_document(self, collection_name: str, doc_path: str) -> None:
        # Remove file from cache
```

#### Integration Point
In `src/pmd/workflows/pipelines/ingestion.py`, inside `_persist_document` or just before calling it in `execute`.

```python
# In IngestionPipeline.execute loop:

async for doc in load_result.documents:
    # ...
    
    # NEW: Cache the document
    if self._cache_config:
        cached_uri = self._cacher.cache_document(collection_name, doc)
        # Update the doc's reference to point to the cache
        doc.ref = replace(doc.ref, uri=cached_uri)
    
    # Existing persistence logic...
    result = await self._persist_document(...)
```

#### Cleanup
The `_cleanup_stale_documents` method in `IngestionPipeline` currently marks documents as inactive and removes them from the FTS index. It should be updated to also remove the corresponding file from the cache to prevent indefinite growth of the cache directory.

```python
# In IngestionPipeline._cleanup_stale_documents:

# ...
if self._cacher:
    self._cacher.remove_document(collection_name, doc.filepath)
# ...
```

### 5. Data Migration (Consideration)
Existing documents in the database will have URIs pointing to their original locations.
*   **Option A (Lazy):** Only update when the document is re-ingested/updated.
*   **Option B (Migration):** A specific migration script to re-process all documents.
*   **Decision:** For now, we will rely on re-indexing (Option A) or a forced re-index if immediate consistency is required.

## Benefits
*   **Availability:** We have a local copy of the file even if the original source (e.g., a network drive or external volume) becomes unavailable.
*   **Uniformity:** All `source_uri`s in our system become `file://` URIs, potentially simplifying downstream consumption.

## Strategic Alignment
This feature aligns with **Proposal 4: Universal Data Source (LlamaIndex)** in the [LlamaIndex Architecture Plan](LLAMAINDEX_ARCHITECTURE_PLAN.md).

*   **Bridge to Local-First:** While LlamaIndex allows us to ingest from remote sources (Notion, Slack, Web), our core philosophy is "Local First". This caching layer acts as the bridge, materializing ephemeral or remote content from LlamaIndex loaders into concrete local files.
*   **Unified Processing:** By converting all ingested content into cached local files, downstream components (chunking, embedding, display) can treat all data uniformly as local files, regardless of whether they originated from a local folder or a remote API via LlamaIndex.

### Dependency & Ordering
**Recommendation: Implement Caching Architecture First.**

This feature is a **prerequisite** for a robust implementation of Proposal 4.
1.  **Foundation:** The caching layer provides the necessary *storage mechanism* to "ground" remote data fetched by LlamaIndex. Without it, the LlamaIndex integration would require complex handling of non-file URIs throughout the system or a separate ad-hoc storage solution.
2.  **Simplification:** By implementing caching first, the future LlamaIndex adapter needs only to fetch content and pass it to the ingestion pipeline. The pipeline handles the complexity of persisting it to a concrete, addressable local file.
