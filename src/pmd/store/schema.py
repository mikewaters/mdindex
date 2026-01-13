"""Database schema constants for PMD.

NOTE: Schema DDL is now managed by migrations in pmd.store.migrations.
This file contains only constants and the vector table schema
(which requires the sqlite-vec extension to be loaded first).

See migrations/versions/ for the authoritative schema definitions.
"""

# Default embedding dimension
# - nomic/modernbert-embed-base uses 768 (default)
# - multilingual-e5-small uses 384
EMBEDDING_DIMENSION = 768

SCHEMA_SQL = """\
-- Content-addressable storage
CREATE TABLE IF NOT EXISTS content (
    hash TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Source collections (indexed directories or remote sources)
CREATE TABLE IF NOT EXISTS source_collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    pwd TEXT NOT NULL,
    glob_pattern TEXT NOT NULL DEFAULT '**/*.md',
    source_type TEXT NOT NULL DEFAULT 'filesystem',
    source_config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Documents (file-to-content mappings)
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_collection_id INTEGER NOT NULL REFERENCES source_collections(id),
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    hash TEXT NOT NULL REFERENCES content(hash),
    active INTEGER NOT NULL DEFAULT 1,
    modified_at TEXT NOT NULL,
    UNIQUE(source_collection_id, path)
);

-- Full-text search index (stores content for DELETE/UPDATE support)
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, body,
    tokenize='porter unicode61'
);

-- Vector embeddings metadata
CREATE TABLE IF NOT EXISTS content_vectors (
    hash TEXT NOT NULL,
    seq INTEGER NOT NULL,
    pos INTEGER NOT NULL,
    model TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (hash, seq)
);

-- Source metadata for remote documents
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

-- Document metadata (extracted tags, attributes)
CREATE TABLE IF NOT EXISTS document_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id),
    profile_name TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    source_tags_json TEXT NOT NULL,
    attributes_json TEXT,
    extracted_at TEXT NOT NULL
);

-- Document tags junction table (inverted index for fast tag lookups)
CREATE TABLE IF NOT EXISTS document_tags (
    document_id INTEGER NOT NULL REFERENCES documents(id),
    tag TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_source_collection ON documents(source_collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors(hash);
CREATE INDEX IF NOT EXISTS idx_source_metadata_uri ON source_metadata(source_uri);
CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag);
"""

# Vector storage table (requires sqlite-vec extension)
# Created separately since it needs the extension loaded first
VECTOR_TABLE_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS content_vectors_vec USING vec0(
    hash TEXT PRIMARY KEY,
    seq INTEGER,
    embedding FLOAT[{dimension}]
);
"""


def get_schema() -> str:
    """Get the SQL schema string."""
    return SCHEMA_SQL


def get_vector_schema(dimension: int = EMBEDDING_DIMENSION) -> str:
    """Get the vector table schema SQL.

    Args:
        dimension: Embedding vector dimension.

    Returns:
        SQL string for creating the vector virtual table.
    """
    return VECTOR_TABLE_SQL.format(dimension=dimension)
