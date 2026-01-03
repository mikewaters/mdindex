# Sources Module Architecture

**Location:** `src/pmd/sources/`

Document source abstraction with a fixed, explicit set of sources.

## Files and Key Abstractions

### `base.py`

**`DocumentSource`** (Protocol) - Source contract

Methods:
- `list_documents()` - Enumerate documents
- `fetch_content()` - Fetch document content
- `capabilities()` - Describe source features
- `check_modified()` - Change detection

**`DocumentReference`** - Document metadata
- uri, path, title, metadata

**`FetchResult`** - Fetch operation result
- content, content_type, encoding, metadata

**`SourceCapabilities`** - Feature flags
- supports_incremental, supports_etag, etc.

### `filesystem.py`

**`FileSystemSource`** - Local filesystem source

Features:
- Glob pattern matching
- Nanosecond mtime comparison
- Content type detection

### `llamaindex.py`

**`LlamaIndexSource`** - Adapter for LlamaIndex readers/loaders

Features:
- Wraps any object exposing `load_data(**kwargs)`
- Maps returned documents to PMD `DocumentReference`/`FetchResult`
- Caches loaded documents for reuse during indexing
