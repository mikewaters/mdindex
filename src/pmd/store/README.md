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

### `migrations/`

Versioned database migrations using SQLite `PRAGMA user_version`.

**Structure:**
```
migrations/
├── __init__.py      # Exports MigrationRunner
├── runner.py        # Migration runner implementation
└── versions/        # Versioned migration files
    ├── __init__.py
    └── v0001_initial_schema.py
```

**`MigrationRunner`** - Applies versioned migrations
- Uses `PRAGMA user_version` for tracking
- Discovers migrations in `versions/` package
- Idempotent execution (safe to re-run)

**Adding New Migrations:**
1. Create `versions/v{NNNN}_{description}.py`
2. Define `VERSION = N` (integer, must be unique)
3. Define `DESCRIPTION = "..."` (human-readable)
4. Define `up(conn)` function to apply changes

```python
# Example: v0002_add_feature.py
VERSION = 2
DESCRIPTION = "Add feature table"

def up(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS feature (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
    ''')
```

**Migration Policy:**
- Migrations are forward-only (no rollback support)
- Use `IF NOT EXISTS` for idempotency
- Test with fresh DB and existing DBs
- Never modify existing migrations after release

### `schema.py`

Constants and reference schema (migrations are authoritative):
- `EMBEDDING_DIMENSION = 768`
- `VECTOR_TABLE_SQL` - sqlite-vec virtual table (created separately)

**Tables (created by v0001_initial_schema):**
- `content` - Content-addressable storage (hash-based)
- `source_collections` - Named document source collections
- `documents` - File-to-content mappings
- `documents_fts` - FTS5 virtual table
- `content_vectors` - Embedding metadata
- `content_vectors_vec` - Vector storage (sqlite-vec, created after extension load)
- `source_metadata` - Remote document metadata
- `document_metadata` - Extracted tags/attributes
- `document_tags` - Tag inverted index

### `documents.py`

**`DocumentRepository`** - Document CRUD with CAS

Key methods:
- `add_or_update()` - Insert/update with content deduplication
- `get()` - Retrieve by source_collection_id + path
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

**`SourceCollectionRepository`** - Source collection management

Key methods:
- `create()` - Create with validation
- `remove()` - Delete with cascading cleanup
- `rename()` - Rename with uniqueness check

**Invariants:**
- Cascading deletion removes documents, orphaned content, embeddings
- Source collection names are unique

### `source_metadata.py`

**`SourceMetadataRepository`** - Remote document tracking

Key methods:
- `upsert()` - Insert or update metadata
- `needs_refresh()` - Check if document exceeds max_age
- `get_stale_documents()` - Find documents needing refresh
