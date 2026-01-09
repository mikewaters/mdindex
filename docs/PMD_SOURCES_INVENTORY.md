# Inventory of `pmd.sources` Usages

This document lists all symbols from the `pmd.sources` module that are used outside of `pmd.sources` within the `src/pmd` codebase.

## `pmd.sources`

### `DocumentReference`
Used by:
- `pmd.services.indexing`:
  - `IndexingService._index_document` (argument type annotation)
  - `IndexingService._update_source_metadata` (argument type annotation)

### `DocumentSource`
Used by:
- `pmd.services.indexing`:
  - `IndexingService.index_collection` (argument type annotation)
  - `IndexingService._index_document` (argument type annotation)

### `FetchResult`
Used by:
- `pmd.services.indexing`:
  - `IndexingService._update_source_metadata` (argument type annotation and internal import)

### `SourceFetchError`
Used by:
- `pmd.services.indexing`:
  - `IndexingService.index_collection` (exception handling)

### `SourceListError`
Used by:
- `pmd.services.indexing`:
  - `IndexingService.index_collection` (exception handling)

### `SourceRegistry`
Used by:
- `pmd.services.indexing`:
  - `IndexingService.__init__` (argument type annotation)

### `get_default_registry`
Used by:
- `pmd.cli.commands.index`:
  - `_handle_index_async` (to get registry instance)
- `pmd.mcp.server`:
  - `PMDMCPServer.index_collection` (to get registry instance)
- `pmd.services.container`:
  - `ServiceContainer` (docstring example)
- `pmd.services.indexing`:
  - `IndexingService.__init__` (default argument value)
- `pmd.services.__init__`:
  - Module docstring example

## `pmd.sources.metadata`

### `get_default_profile_registry`
Used by:
- `pmd.services.indexing`:
  - `IndexingService._extract_metadata_via_profiles` (to get profile registry)
