# Store Module Architecture

**Location:** `src/pmd/store/`

Data access layer implementing the Repository pattern with SQLite.

## Files and Key Abstractions

### `database.py`

**`Database`** - Connection manager
- Lazy connection initialization
- FTS5 and sqlite-vec extension loading
- Transaction support via context manager
- Graceful degradation if sqlite-vec unavailable

**Invariants:**
- Foreign keys enabled (PRAGMA foreign_keys = ON)
- Row factory set to sqlite3.Row
- `vec_available` property indicates vector search capability

### `schema.py`

Database schema definitions:
- `SCHEMA_VERSION = 1`
- `EMBEDDING_DIMENSION = 768`
- `SCHEMA_SQL` - Relational tables
- `VECTOR_TABLE_SQL` - sqlite-vec virtual table

**Tables:**
- `content` - Content-addressable storage (hash-based)
- `collections` - Named document collections
- `documents` - File-to-content mappings
- `documents_fts` - FTS5 virtual table
- `content_vectors` - Embedding metadata
- `content_vectors_vec` - Vector storage (sqlite-vec)
- `path_contexts` - Directory-level context
- `source_metadata` - Remote document metadata

### `documents.py`

**`DocumentRepository`** - Document CRUD with CAS

Key methods:
- `add_or_update()` - Insert/update with content deduplication
- `get()` - Retrieve by collection_id + path
- `delete()` - Soft-delete (mark inactive)
- `check_if_modified()` - Hash comparison for change detection

**Invariants:**
- Documents reference content by SHA256 hash
- Soft deletes via `active` flag (no hard deletion)
- Transactions ensure atomicity of content + document

### `embeddings.py`

**`EmbeddingRepository`** - Vector storage and similarity search

Key methods:
- `store_embedding()` - Store vector + metadata
- `search_vectors()` - L2-based similarity search
- `has_embeddings()` - Check embedding existence

**Invariants:**
- Separates metadata (content_vectors) from vectors (content_vectors_vec)
- Score = 1 / (1 + distance) for L2 conversion
- Deduplicates by document path (best chunk per doc)

### `search.py`

**`SearchRepository[QueryT]`** (ABC, Generic) - Abstract search interface
- Type parameter allows different query types

**`FTS5SearchRepository`** - BM25 full-text search
- Porter stemming + Unicode61 tokenization
- Score normalization to [0, 1]

**Invariants:**
- FTS5 returns negative BM25 scores (converted to positive)
- Max-normalization applied to result set

### `vector_search.py`

**`VectorSearchRepository`** - Adapter wrapping EmbeddingRepository
- Implements `SearchRepository[list[float]]`
- Enables polymorphic treatment with FTS5

### `collections.py`

**`CollectionRepository`** - Collection management

Key methods:
- `create()` - Create with validation
- `remove()` - Delete with cascading cleanup
- `rename()` - Rename with uniqueness check

**Invariants:**
- Cascading deletion removes documents, orphaned content, embeddings
- Collection names are unique

### `source_metadata.py`

**`SourceMetadataRepository`** - Remote document tracking

Key methods:
- `upsert()` - Insert or update metadata
- `needs_refresh()` - Check if document exceeds max_age
- `get_stale_documents()` - Find documents needing refresh
