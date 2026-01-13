# LlamaIndex Loader Adapter Instructions

## Purpose
Define an explicit adapter for LlamaIndex loaders that integrates with the new `LoadingService` while enforcing strict identity, metadata, and no-chunking constraints.

## Requirements Summary
1) Unique document identity: every loaded document must have a stable unique URI. If the LlamaIndex `Document` does not provide a path, a unique URI must still be constructed deterministically.
2) Content type: must be supplied by the adapter caller (the code that configures the loader knows the content type).
3) Metadata: extract metadata from document content via `pmd.metadata` profiles, but also augment it with any LlamaIndex metadata present on the `Document`.
4) No multi-document per source item: by default, reject loaders that produce multiple documents (chunking). Allow opt-in for loaders that legitimately produce multiple documents.

## Adapter Location and Naming
Create a new adapter class that is dedicated to LlamaIndex loaders and separate from the generic `DocumentSource` abstraction:
- New module: `src/pmd/services/loading_llamaindex.py`
- Class name: `LlamaIndexLoaderAdapter`

Rationale: this avoids forcing LlamaIndex into the `list_documents` + `fetch_content` split and aligns with the Loader service's data model.

## Expected LlamaIndex Inputs
Assume LlamaIndex loaders return `Document` objects that include:
- `text` or `get_content()` for content
- `metadata: dict` for metadata
- `id_` optionally

The adapter should treat `metadata` as an augmentation layer and not as a replacement for `pmd.metadata` extraction.

## Sync-to-Async Wrapping

LlamaIndex loaders are typically synchronous (`loader.load_data()`), but the adapter exposes an async interface for consistency with `LoadingService`. The adapter must wrap synchronous calls:

```python
import asyncio

async def load_eager(self) -> list[LoadedDocument]:
    # Wrap sync loader in thread to avoid blocking
    docs = await asyncio.to_thread(self._loader.load_data)
    return self._convert_documents(docs)
```

This keeps the async contract while allowing synchronous loaders to work correctly.

## Adapter Interface

```python
class LlamaIndexLoaderAdapter:
    def __init__(
        self,
        loader: Any,
        *,
        content_type: str,
        encoding: str = "utf-8",
        uri_key: str = "path",
        title_key: str = "title",
        namespace: str = "llamaindex",
        allow_multiple: bool = False,
    ) -> None:
        """Initialize adapter for a LlamaIndex loader.

        Args:
            loader: LlamaIndex loader instance with load_data() method.
            content_type: MIME type for loaded content (required).
            encoding: Text encoding (default: utf-8).
            uri_key: Metadata key to use for document path/URI (default: "path").
            title_key: Metadata key to use for document title (default: "title").
            namespace: URI scheme namespace (default: "llamaindex").
            allow_multiple: If True, allow loaders that return multiple documents.
                If False (default), raise ValueError if loader returns > 1 document.
        """
        ...

    async def load_eager(self) -> list[LoadedDocument]:
        """Load all documents from the loader.

        Returns:
            List of LoadedDocument instances.

        Raises:
            ValueError: If allow_multiple=False and loader returns multiple documents.
            ValueError: If no unique identifier can be constructed for a document.
        """
        ...
```

Notes:
- `content_type` is required and must be supplied by the caller.
- `uri_key` defaults to `"path"`, but can be any metadata key that uniquely identifies documents.
- `namespace` is used to build canonical URIs if only an ID is available.
- `allow_multiple` defaults to False to prevent accidental chunking at loader stage.

## URI and Identity Rules

The adapter must construct a stable unique URI for each document. URIs are opaque identifiers used for deduplication and change detection - they are not expected to be resolvable.

Fallback chain:
1) If `metadata[uri_key]` exists, use it as the path portion.
2) Else if the LlamaIndex document has `id_`, use `id_`.
3) Else compute a stable hash of `(namespace + content + sorted(metadata))` as a fallback ID.

The final URI should be constructed as:
```
{namespace}://{identifier}
```

The `LoadedDocument.ref.path` should match the identifier (the part after `://`) to keep indexing consistent.

Note: The hash fallback includes the namespace to ensure the same content loaded via different adapters or into different contexts produces different URIs.

## Metadata Augmentation Policy
The adapter should combine:
- Extracted metadata from `pmd.metadata` profiles (tags, attributes)
- LlamaIndex `Document.metadata` dict

Rules:
- LlamaIndex metadata must not overwrite extracted tags unless explicitly mapped.
- Preserve LlamaIndex metadata inside `ExtractedMetadata.attributes` under a reserved key: `"_llamaindex"`.
- If metadata keys conflict, extracted metadata wins.

Example:
```python
extracted = ExtractedMetadata(
    tags=["python", "tutorial"],
    attributes={
        "author": "extracted author",
        "_llamaindex": {
            "source": "web",
            "author": "llamaindex author",  # Does NOT override extracted
            "page_number": 1,
        },
    },
)
```

## Multi-Document Handling

By default (`allow_multiple=False`), the adapter rejects loaders that return multiple documents:

```python
docs = await asyncio.to_thread(self._loader.load_data)
if len(docs) > 1 and not self._allow_multiple:
    raise ValueError(
        f"Loader returned {len(docs)} documents but allow_multiple=False. "
        "Set allow_multiple=True if this loader legitimately produces multiple documents, "
        "or use a different loader that returns single documents."
    )
```

When `allow_multiple=True`, all documents are processed and must each have unique URIs. Duplicate URIs raise `ValueError`.

This prevents chunking at the loader stage. Chunking should only happen during indexing/embedding.

## Integration with LoadingService

The `LoadingService` should support injecting this adapter for LlamaIndex loaders. The service provides the `collection_id` that the adapter cannot know:

```python
# In LoadingService

async def load_from_llamaindex(
    self,
    collection_name: str,
    adapter: LlamaIndexLoaderAdapter,
) -> EagerLoadResult:
    """Load documents from a LlamaIndex loader adapter.

    Args:
        collection_name: Name of the collection to load into.
        adapter: Configured LlamaIndex loader adapter.

    Returns:
        EagerLoadResult with loaded documents.

    Raises:
        CollectionNotFoundError: If collection does not exist.
    """
    collection = self._collection_repo.get_by_name(collection_name)
    if not collection:
        raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

    # Load via adapter (does not have collection context)
    loaded_docs = await adapter.load_eager()

    # Inject collection_id into each document
    for doc in loaded_docs:
        doc.collection_id = collection.id

    return EagerLoadResult(
        documents=loaded_docs,
        enumerated_paths={doc.path for doc in loaded_docs},
        errors=[],
    )
```

This keeps the adapter independent of collection details while allowing full integration.

## Error Handling
- Raise `ValueError` if no unique identifier can be constructed.
- Raise `ValueError` if `allow_multiple=False` and loader returns multiple documents.
- Raise `ValueError` if duplicate URIs are detected when `allow_multiple=True`.

## Tests to Add

| Test Case | Description |
|-----------|-------------|
| test_uri_from_metadata_path | Uses `metadata[uri_key]` when present |
| test_uri_from_id | Falls back to `id_` when no path |
| test_uri_from_hash | Falls back to content hash when no id |
| test_hash_includes_namespace | Hash fallback includes namespace for uniqueness |
| test_metadata_augmentation | Extracted metadata preserved, LlamaIndex in `_llamaindex` |
| test_metadata_conflict_extracted_wins | Extracted values take precedence |
| test_single_doc_default | Raises ValueError if multiple docs and allow_multiple=False |
| test_allow_multiple_true | Accepts multiple docs when allow_multiple=True |
| test_duplicate_uri_rejected | Raises ValueError on duplicate URIs |
| test_content_type_propagated | content_type appears in LoadedDocument |
| test_sync_loader_wrapped | Synchronous loader wrapped with asyncio.to_thread |
| test_collection_id_injected | LoadingService injects collection_id correctly |

## File Changes Summary

| File | Change |
|------|--------|
| `src/pmd/services/loading_llamaindex.py` | **Add**: New adapter module |
| `src/pmd/services/loading.py` | **Update**: Add `load_from_llamaindex` method |
| `src/pmd/services/__init__.py` | **Update**: Export `LlamaIndexLoaderAdapter` |
| `tests/unit/services/test_loading_llamaindex.py` | **Add**: Unit tests for adapter |
