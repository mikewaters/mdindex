"""Initial database schema.

Creates all core tables for PMD:
- content: Content-addressable storage
- collections: Indexed directories or remote sources
- documents: File-to-content mappings
- documents_fts: Full-text search index
- content_vectors: Vector embeddings metadata
- path_contexts: Context descriptions
- ollama_cache: API response cache
- source_metadata: Remote document metadata
- document_metadata: Extracted tags and attributes
- document_tags: Inverted index for tag lookups
"""

VERSION = 1
DESCRIPTION = "Initial schema with all core tables"


def up(conn):
    """Apply initial schema migration."""
    conn.executescript(
        """
        -- Content-addressable storage
        CREATE TABLE IF NOT EXISTS content (
            hash TEXT PRIMARY KEY,
            doc TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        -- Collections (indexed directories or remote sources)
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
        CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors(hash);
        CREATE INDEX IF NOT EXISTS idx_path_contexts_collection ON path_contexts(collection_id);
        CREATE INDEX IF NOT EXISTS idx_source_metadata_uri ON source_metadata(source_uri);
        CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag);
        """
    )
