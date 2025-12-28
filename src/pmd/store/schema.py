"""Database schema definitions for PMD."""

SCHEMA_VERSION = 1

# Default embedding dimension (nomic-embed-text uses 768)
EMBEDDING_DIMENSION = 768

SCHEMA_SQL = """\
-- Content-addressable storage
CREATE TABLE IF NOT EXISTS content (
    hash TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Collections (indexed directories)
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    pwd TEXT NOT NULL,
    glob_pattern TEXT NOT NULL DEFAULT '**/*.md',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Documents (file-to-content mappings)
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

-- Context descriptions
CREATE TABLE IF NOT EXISTS path_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    path_prefix TEXT NOT NULL,
    context TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(collection_id, path_prefix)
);

-- Ollama API response cache
CREATE TABLE IF NOT EXISTS ollama_cache (
    hash TEXT PRIMARY KEY,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors(hash);
CREATE INDEX IF NOT EXISTS idx_path_contexts_collection ON path_contexts(collection_id);
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
