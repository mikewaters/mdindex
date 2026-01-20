# Data Architecture

This document describes the data model, database schema, and data ingestion pipeline for PMD (Python Markdown Search).

---

## Table of Contents

1. [Database Overview](#1-database-overview)
2. [Database Entities](#2-database-entities)
3. [Virtual Tables](#3-virtual-tables)
4. [Indexes](#4-indexes)
5. [Entity Relationships](#5-entity-relationships)
6. [Data Ingestion Pipeline](#6-data-ingestion-pipeline)
7. [Content-Addressable Storage](#7-content-addressable-storage)
8. [Incremental Update Strategy](#8-incremental-update-strategy)

---

## 1. Database Overview

PMD uses SQLite as its primary data store, with two optional extensions:

| Component | Purpose | Required |
|-----------|---------|----------|
| **SQLite** | Relational data storage | Yes |
| **FTS5** | Full-text search with BM25 ranking | Yes (built-in) |
| **sqlite-vec** | Vector similarity search | Optional |

**Database Location**: `~/.cache/pmd/index.db` (configurable via `PMD_CONFIG`)

**Schema Version**: 1

**Default Embedding Dimension**: 768 (nomic/modernbert-embed-base)

---

## 2. Database Entities

### 2.1 collections

Represents indexed document collections (directories, HTTP endpoints, or custom sources).

```sql
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    pwd TEXT NOT NULL,
    glob_pattern TEXT NOT NULL DEFAULT '**/*.md',
    source_type TEXT NOT NULL DEFAULT 'filesystem',
    source_config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique collection identifier |
| `name` | TEXT | NOT NULL, UNIQUE | Human-readable collection name |
| `pwd` | TEXT | NOT NULL | Base path (filesystem) or URI (remote) |
| `glob_pattern` | TEXT | DEFAULT '**/*.md' | File matching pattern |
| `source_type` | TEXT | DEFAULT 'filesystem' | Source type: `filesystem`, `http`, `entity` |
| `source_config` | TEXT | - | JSON-serialized source configuration |
| `created_at` | TEXT | NOT NULL | ISO timestamp of creation |
| `updated_at` | TEXT | NOT NULL | ISO timestamp of last update |

### 2.2 content

Content-addressable storage for document bodies. Enables deduplication of identical content.

```sql
CREATE TABLE IF NOT EXISTS content (
    hash TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `hash` | TEXT | PRIMARY KEY | SHA256 hash of document content |
| `doc` | TEXT | NOT NULL | Complete document body |
| `created_at` | TEXT | NOT NULL | ISO timestamp when first stored |

### 2.3 documents

Maps document paths to content, tracking metadata and soft-delete status.

```sql
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    hash TEXT NOT NULL REFERENCES content(hash),
    active INTEGER NOT NULL DEFAULT 1,
    modified_at TEXT NOT NULL,
    UNIQUE(collection_id, path)
);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique document identifier |
| `collection_id` | INTEGER | NOT NULL, FK→collections | Parent collection |
| `path` | TEXT | NOT NULL | Relative path within collection |
| `title` | TEXT | NOT NULL | Document title (extracted from content) |
| `hash` | TEXT | NOT NULL, FK→content | Reference to content (enables dedup) |
| `active` | INTEGER | DEFAULT 1 | Soft-delete flag (1=active, 0=deleted) |
| `modified_at` | TEXT | NOT NULL | ISO timestamp of last modification |

**Constraint**: `UNIQUE(collection_id, path)` ensures one document per path per collection.

### 2.4 content_vectors

Metadata for vector embeddings, tracking chunks and their positions.

```sql
CREATE TABLE IF NOT EXISTS content_vectors (
    hash TEXT NOT NULL,
    seq INTEGER NOT NULL,
    pos INTEGER NOT NULL,
    model TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (hash, seq)
);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `hash` | TEXT | NOT NULL | Content hash (FK→content) |
| `seq` | INTEGER | NOT NULL | Chunk sequence number (0, 1, 2, ...) |
| `pos` | INTEGER | NOT NULL | Character position in original document |
| `model` | TEXT | NOT NULL | Embedding model name |
| `embedded_at` | TEXT | NOT NULL | ISO timestamp of embedding creation |

**Primary Key**: `(hash, seq)` enables tracking multiple chunks per document.

### 2.5 source_metadata

Tracks HTTP metadata for remote documents (ETags, Last-Modified) enabling efficient incremental updates.

```sql
CREATE TABLE IF NOT EXISTS source_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id),
    source_uri TEXT NOT NULL,
    etag TEXT,
    last_modified TEXT,
    last_fetched_at TEXT NOT NULL,
    fetch_duration_ms INTEGER,
    http_status INTEGER,
    content_type TEXT,
    extra_metadata TEXT
);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Metadata identifier |
| `document_id` | INTEGER | NOT NULL, UNIQUE, FK→documents | One-to-one with document |
| `source_uri` | TEXT | NOT NULL | Original fetch URI |
| `etag` | TEXT | - | HTTP ETag header |
| `last_modified` | TEXT | - | HTTP Last-Modified header |
| `last_fetched_at` | TEXT | NOT NULL | ISO timestamp of last fetch |
| `fetch_duration_ms` | INTEGER | - | Fetch duration in milliseconds |
| `http_status` | INTEGER | - | HTTP status code (200, 304, etc.) |
| `content_type` | TEXT | - | MIME type |
| `extra_metadata` | TEXT | - | JSON-serialized additional metadata |

### 2.7 ollama_cache

Caches Ollama API responses to avoid repeated requests.

```sql
CREATE TABLE IF NOT EXISTS ollama_cache (
    hash TEXT PRIMARY KEY,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `hash` | TEXT | PRIMARY KEY | Hash of request parameters |
| `result` | TEXT | NOT NULL | Cached API response |
| `created_at` | TEXT | NOT NULL | ISO timestamp |

---

## 3. Virtual Tables

### 3.1 documents_fts (FTS5)

Full-text search index using SQLite FTS5 with BM25 ranking.

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, body,
    tokenize='porter unicode61'
);
```

**Configuration**:
- **Tokenizer**: `porter unicode61`
  - `unicode61`: Unicode-aware tokenization (case-insensitive, accent-folding)
  - `porter`: Porter stemming algorithm (running → run)
- **Indexed Columns**:
  - `path`: Document path (enables path-based queries)
  - `body`: Full document content

**Linking**: `rowid` corresponds to `documents.id` for JOIN operations.

**Ranking**: BM25 algorithm (negative scores; lower = better match).

### 3.2 content_vectors_vec (sqlite-vec)

Vector storage and similarity search using sqlite-vec extension.

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS content_vectors_vec USING vec0(
    hash TEXT PRIMARY KEY,
    seq INTEGER,
    embedding FLOAT[768]
);
```

**Configuration**:
- **Dimension**: 768 (configurable; 384 for multilingual-e5-small)
- **Distance Metric**: L2 (Euclidean)
- **Key Format**: `"{content_hash}:{seq}"` (composite)

**Availability**: Optional; vector search disabled if extension unavailable.

**Binary Format**: Single-precision floats via `struct.pack()`.

---

## 4. Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors(hash);
CREATE INDEX IF NOT EXISTS idx_source_metadata_uri ON source_metadata(source_uri);
```

| Index | Table | Column(s) | Purpose |
|-------|-------|-----------|--------|
| `idx_documents_collection` | documents | collection_id | Fast collection-scoped queries |
| `idx_documents_hash` | documents | hash | Content deduplication lookups |
| `idx_content_vectors_hash` | content_vectors | hash | Embedding retrieval by content |
| `idx_source_metadata_uri` | source_metadata | source_uri | Metadata by URI |

---

## 5. Entity Relationships

```
┌─────────────────┐
│   collections   │
│─────────────────│
│ id (PK)         │◄─────────────────────────┐
│ name (UNIQUE)   │                          │
│ pwd             │                          │
│ source_type     │                          │
└────────┬────────┘                          │
         │                                   │
         │ 1:N                               │
         ▼                                   │
┌─────────────────┐                          │
│    documents    │                          │
│─────────────────│                          │
│ id (PK)         │◄───────┐                 │
│ collection_id   │───────►│                 │
│ path            │        │                 │
│ hash ───────────┼──┐     │                 │
│ active          │  │     │                 │
│ title           │  │     │                 │
└────────┬────────┘  │     │                 │
         │           │     │                 │
         │ 1:1       │     │                 │
         ▼           │     │                 │
┌─────────────────┐  │     │                 │
│ source_metadata │  │     │                 │
│─────────────────│  │     │                 │
│ id (PK)         │  │     │                 │
│ document_id ────┼──┘     │                 │
│ etag            │        │                 │
│ last_modified   │        │                 │
└─────────────────┘        │                 │
                           │                 │
         ┌─────────────────┘                 │
         │ N:1                               │
         ▼                                   │
┌─────────────────┐                          │
│     content     │                          │
│─────────────────│                          │
│ hash (PK)       │◄─────────────────┐       │
│ doc             │                  │       │
│ created_at      │                  │       │
└────────┬────────┘                  │       │
         │                           │       │
         │ 1:N                       │       │
         ▼                           │       │
┌─────────────────┐                  │       │
│ content_vectors │                  │       │
│─────────────────│                  │       │
│ hash ───────────┼──────────────────┘       │
│ seq             │                          │
│ pos             │                          │
│ model           │                          │
│ (PK: hash, seq) │                          │
└────────┬────────┘                          │
         │                                   │
         │ 1:1                               │
         ▼                                   │
┌─────────────────────┐                      │
│ content_vectors_vec │                      │
│─────────────────────│                      │
│ hash (PK, composite)│                      │
│ seq                 │                      │
│ embedding           │                      │
└─────────────────────┘                      │

┌─────────────────┐          ┌─────────────────┐
│  documents_fts  │          │   ollama_cache  │
│─────────────────│          │─────────────────│
│ rowid → doc.id  │          │ hash (PK)       │
│ path            │          │ result          │
│ body            │          │ created_at      │
└─────────────────┘          └─────────────────┘
```

**Key Relationships**:
- `documents.collection_id` → `collections.id` (many-to-one)
- `documents.hash` → `content.hash` (many-to-one, enables dedup)
- `source_metadata.document_id` → `documents.id` (one-to-one)
- `content_vectors.hash` → `content.hash` (many-to-one)
- `documents_fts.rowid` → `documents.id` (implicit, one-to-one)

---

## 6. Data Ingestion Pipeline

The following diagram shows how files become searchable:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COLLECTION CONFIGURATION                     │
├─────────────────────────────────────────────────────────────────┤
│  collection add <name> <path>                                   │
│    └─► Store in collections table                               │
│    └─► Configure source_type, glob_pattern, source_config       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT DISCOVERY                            │
├─────────────────────────────────────────────────────────────────┤
│  source.list_documents()                                        │
│    ├─► FileSystem: Glob pattern matching (e.g., **/*.md)        │
│    ├─► HTTP: Sitemap parsing or explicit URL list               │
│    └─► Entity: Custom resolver enumeration                      │
│                                                                 │
│  Yields: DocumentReference(uri, path, title, metadata)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CHANGE DETECTION                              │
├─────────────────────────────────────────────────────────────────┤
│  source.check_modified(ref, stored_metadata)                    │
│    ├─► FileSystem: Compare mtime_ns (nanosecond precision)      │
│    ├─► HTTP: Compare ETag or Last-Modified via HEAD request     │
│    └─► Returns: True (possibly changed) or False (unchanged)    │
│                                                                 │
│  If unchanged → SKIP (no fetch needed)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT FETCHING                              │
├─────────────────────────────────────────────────────────────────┤
│  source.fetch_content(ref)                                      │
│    ├─► FileSystem: Read file with encoding detection            │
│    ├─► HTTP: GET request with conditional headers               │
│    └─► Returns: FetchResult(content, content_type, metadata)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUALITY VALIDATION                            │
├─────────────────────────────────────────────────────────────────┤
│  is_indexable(content)                                          │
│    ├─► Reject empty documents                                   │
│    ├─► Reject title-only documents (# Heading only)             │
│    └─► Accept documents with meaningful content                 │
│                                                                 │
│  extract_title(content)                                         │
│    ├─► Look for first # Heading                                 │
│    └─► Fallback to filename stem                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT HASHING & DEDUP                       │
├─────────────────────────────────────────────────────────────────┤
│  hash = sha256_hash(content)                                    │
│                                                                 │
│  INSERT OR IGNORE INTO content(hash, doc, created_at)           │
│    └─► Deduplicates: identical content stored once              │
│                                                                 │
│  Compare new_hash vs stored document.hash                       │
│    ├─► Same hash → SKIP indexing (content unchanged)            │
│    └─► Different → PROCEED to indexing                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT STORAGE                              │
├─────────────────────────────────────────────────────────────────┤
│  INSERT OR UPDATE documents                                     │
│    ├─► collection_id, path (unique constraint)                  │
│    ├─► title, hash (reference to content)                       │
│    ├─► active = 1, modified_at = now()                          │
│    └─► Returns: document_id                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FTS5 INDEXING                                 │
├─────────────────────────────────────────────────────────────────┤
│  if is_indexable(content):                                      │
│    fts_repo.index_document(doc_id, path, body)                  │
│      └─► INSERT OR REPLACE INTO documents_fts                   │
│      └─► rowid = document_id                                    │
│      └─► Porter stemming + Unicode61 tokenization               │
│  else:                                                          │
│    fts_repo.remove_from_index(doc_id)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    METADATA STORAGE                              │
├─────────────────────────────────────────────────────────────────┤
│  UPSERT INTO source_metadata                                    │
│    ├─► document_id (one-to-one)                                 │
│    ├─► source_uri, etag, last_modified                          │
│    ├─► last_fetched_at, fetch_duration_ms                       │
│    └─► http_status, content_type                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION (Optional)               │
├─────────────────────────────────────────────────────────────────┤
│  if embed=True and sqlite-vec available:                        │
│                                                                 │
│  1. CHUNKING                                                    │
│     chunk_document(content, config)                             │
│       ├─► Split by paragraph (\\n\\n)                           │
│       ├─► Split by sentence/line                                │
│       └─► Target: ~512 tokens per chunk                         │
│                                                                 │
│  2. EMBEDDING                                                   │
│     For each chunk:                                             │
│       llm_provider.embed(chunk_text)                            │
│         └─► Generate 768-dim vector (or configured dim)         │
│                                                                 │
│  3. STORAGE                                                     │
│     INSERT INTO content_vectors(hash, seq, pos, model)          │
│     INSERT INTO content_vectors_vec(hash:seq, embedding)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT IS SEARCHABLE                        │
├─────────────────────────────────────────────────────────────────┤
│  FTS Search: BM25 keyword matching via documents_fts            │
│  Vector Search: L2 similarity via content_vectors_vec           │
│  Hybrid Search: Combined ranking via Reciprocal Rank Fusion     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Content-Addressable Storage

PMD uses a content-addressable storage (CAS) pattern for document content:

### How It Works

1. **Content Hashing**: Each document's content is hashed using SHA256
2. **Deduplication**: Content is stored once in the `content` table, keyed by hash
3. **Reference**: Documents point to content via the `hash` foreign key

### Benefits

- **Storage Efficiency**: Identical documents share storage
- **Fast Change Detection**: Compare hashes instead of content
- **Version Tracking**: Different versions have different hashes

### Example

```
Document A: /docs/readme.md   → hash: abc123
Document B: /backup/readme.md → hash: abc123  (same content)
Document C: /docs/guide.md    → hash: def456  (different content)

content table:
┌──────────┬───────────────────────┐
│ hash     │ doc                   │
├──────────┼───────────────────────┤
│ abc123   │ "# Welcome to PMD..." │
│ def456   │ "# User Guide..."     │
└──────────┴───────────────────────┘

documents table:
┌────┬─────────────────────┬──────────┐
│ id │ path                │ hash     │
├────┼─────────────────────┼──────────┤
│ 1  │ /docs/readme.md     │ abc123   │ ◄─┐
│ 2  │ /backup/readme.md   │ abc123   │ ◄─┴─ Same content
│ 3  │ /docs/guide.md      │ def456   │
└────┴─────────────────────┴──────────┘
```

---

## 8. Incremental Update Strategy

PMD supports efficient incremental updates through multiple levels of change detection:

### Level 1: Source-Level Check (Fastest)

Before fetching content, check if the source reports changes:

| Source Type | Method | Data Compared |
|-------------|--------|---------------|
| FileSystem | `stat()` | `mtime_ns` (nanosecond precision) |
| HTTP | HEAD request | `ETag` or `Last-Modified` header |
| Entity | Custom | Resolver-specific metadata |

**If unchanged**: Skip fetch entirely (no I/O).

### Level 2: Content Hash Check (Fast)

After fetching, compare content hashes:

```python
new_hash = sha256_hash(fetched_content)
if new_hash == stored_document.hash:
    # Content unchanged, skip re-indexing
    # But update source_metadata (last_fetched_at)
```

**If unchanged**: Skip FTS/embedding regeneration.

### Level 3: Full Reindex (Fallback)

Triggered when:
- `force=True` flag is set
- No stored metadata exists
- Source metadata is stale

### Incremental Update Flow

```
index_collection(collection_name, source, force=False)
  │
  ├─► For each document in source.list_documents():
  │     │
  │     ├─► Get stored_metadata for (collection_id, path)
  │     │
  │     ├─► source.check_modified(ref, stored_metadata)
  │     │     │
  │     │     ├─► False (unchanged)
  │     │     │     └─► SKIP to next document
  │     │     │
  │     │     └─► True (possibly changed)
  │     │           └─► CONTINUE to fetch
  │     │
  │     ├─► Fetch content
  │     │
  │     ├─► new_hash = sha256_hash(content)
  │     │     │
  │     │     ├─► Same as stored hash
  │     │     │     └─► Update metadata, SKIP indexing
  │     │     │
  │     │     └─► Different from stored hash
  │     │           └─► PROCEED to full indexing
  │     │
  │     ├─► Store document with new hash
  │     ├─► Index in FTS5
  │     ├─► Update source_metadata
  │     └─► (Optional) Generate embeddings
  │
  └─► Return IndexResult(indexed=N, skipped=M, errors=[...])
```

### Metadata Tracked for Incremental Updates

| Metadata | Source Type | Purpose |
|----------|-------------|---------|
| `mtime_ns` | FileSystem | Nanosecond-precision modification time |
| `etag` | HTTP | HTTP ETag header for conditional requests |
| `last_modified` | HTTP | HTTP Last-Modified header |
| `last_fetched_at` | All | Timestamp for staleness detection |
| `hash` | All (stored in documents) | Content change detection |

---

## Appendix: SQL Schema Reference

Complete schema definition from `src/pmd/store/schema.py`:

```sql
-- Version tracking
-- SCHEMA_VERSION = 1
-- EMBEDDING_DIMENSION = 768

-- Core tables
CREATE TABLE IF NOT EXISTS content (
    hash TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    pwd TEXT NOT NULL,
    glob_pattern TEXT NOT NULL DEFAULT '**/*.md',
    source_type TEXT NOT NULL DEFAULT 'filesystem',
    source_config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    hash TEXT NOT NULL REFERENCES content(hash),
    active INTEGER NOT NULL DEFAULT 1,
    modified_at TEXT NOT NULL,
    UNIQUE(collection_id, path)
);

CREATE TABLE IF NOT EXISTS content_vectors (
    hash TEXT NOT NULL,
    seq INTEGER NOT NULL,
    pos INTEGER NOT NULL,
    model TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (hash, seq)
);

CREATE TABLE IF NOT EXISTS ollama_cache (
    hash TEXT PRIMARY KEY,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id),
    source_uri TEXT NOT NULL,
    etag TEXT,
    last_modified TEXT,
    last_fetched_at TEXT NOT NULL,
    fetch_duration_ms INTEGER,
    http_status INTEGER,
    content_type TEXT,
    extra_metadata TEXT
);

-- Virtual tables
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, body,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS content_vectors_vec USING vec0(
    hash TEXT PRIMARY KEY,
    seq INTEGER,
    embedding FLOAT[768]
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors(hash);
CREATE INDEX IF NOT EXISTS idx_source_metadata_uri ON source_metadata(source_uri);
```
