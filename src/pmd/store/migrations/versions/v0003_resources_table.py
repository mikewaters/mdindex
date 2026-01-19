"""Add resources table for tracking fetch/index state.

Creates the resources table which tracks:
- URI-based resource identification
- Content hashing and caching references
- Source timestamps (created/modified at source)
- Load state (pending/loaded/error) with method and error tracking
- Index state (pending/indexed/error) with method and error tracking
- Extensible metadata via JSON blob

Also adds a nullable resource_id foreign key to the documents table,
linking documents to their source resource. The column is nullable
because existing documents won't have a resource_id until backfill.

This separates fetch/index lifecycle management from the documents table,
allowing more granular control over resource processing pipelines.

Backfill logic (Phase 1b):
- For filesystem collections: creates Resource rows with file:// URIs
- For HTTP/remote collections: uses source_metadata.source_uri if available
- Updates documents.resource_id to link to the new Resource rows
"""

from datetime import datetime, timezone

VERSION = 3
DESCRIPTION = "Add resources table and resource_id FK on documents"


def up(conn):
    """Apply migration: create resources table."""
    conn.executescript(
        """
        -- Resources table for tracking fetch/index state
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_collection_id INTEGER NOT NULL REFERENCES source_collections(id),
            uri TEXT NOT NULL,
            resource_type TEXT,
            hash TEXT,
            content_ref TEXT,
            source_created_at TEXT,
            source_modified_at TEXT,
            loaded_at TEXT,
            load_method TEXT,
            load_status TEXT DEFAULT 'pending',
            load_error TEXT,
            indexed_at TEXT,
            index_state TEXT DEFAULT 'pending',
            index_method TEXT,
            index_error TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(source_collection_id, uri)
        );

        -- Indexes for efficient querying by state
        CREATE INDEX IF NOT EXISTS idx_resources_collection_index_state
            ON resources(source_collection_id, index_state);
        CREATE INDEX IF NOT EXISTS idx_resources_collection_load_status
            ON resources(source_collection_id, load_status);

        -- Add foreign key from documents to resources
        -- Nullable because existing documents won't have a resource_id until backfill
        ALTER TABLE documents ADD COLUMN resource_id INTEGER REFERENCES resources(id);

        -- Index for efficient document lookups by resource
        CREATE INDEX IF NOT EXISTS idx_documents_resource_id ON documents(resource_id);
        """
    )

    # Run data backfill after DDL
    _backfill_resources(conn)


def _backfill_resources(conn):
    """Backfill Resource rows for existing documents.

    Creates Resource rows for all active documents and links them via resource_id.

    Strategy:
    1. For filesystem collections (source_type = 'filesystem'):
       - URI = 'file://' + pwd + '/' + document.path
    2. For non-filesystem collections (HTTP, entity, etc.):
       - Use source_metadata.source_uri if available
       - Skip documents without source_metadata (they'll get Resources on next sync)

    All backfilled Resources are marked as loaded + indexed since the documents
    already exist and are searchable.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Step 1: Backfill filesystem collections
    # These are straightforward: URI = file:// + pwd + path
    cursor = conn.execute(
        """
        SELECT
            d.id AS doc_id,
            d.source_collection_id,
            d.path,
            d.hash,
            d.modified_at,
            sc.pwd,
            sc.source_type
        FROM documents d
        JOIN source_collections sc ON sc.id = d.source_collection_id
        WHERE d.active = 1
          AND d.resource_id IS NULL
          AND sc.source_type = 'filesystem'
        """
    )
    filesystem_docs = cursor.fetchall()

    for row in filesystem_docs:
        doc_id = row[0]
        collection_id = row[1]
        path = row[2]
        doc_hash = row[3]
        modified_at = row[4]
        pwd = row[5]

        # Construct file:// URI
        # Ensure pwd doesn't have trailing slash and path doesn't have leading slash
        pwd = pwd.rstrip("/")
        path = path.lstrip("/")
        uri = f"file://{pwd}/{path}"

        # Insert Resource
        conn.execute(
            """
            INSERT INTO resources (
                source_collection_id, uri, hash,
                load_status, load_method, loaded_at,
                index_state, index_method, indexed_at,
                source_modified_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'loaded', 'backfill', ?, 'indexed', 'backfill', ?, ?, ?, ?)
            """,
            (collection_id, uri, doc_hash, modified_at, modified_at, modified_at, now, now),
        )

        # Get the new resource ID
        resource_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Update document with resource_id
        conn.execute(
            "UPDATE documents SET resource_id = ? WHERE id = ?",
            (resource_id, doc_id),
        )

    # Step 2: Backfill non-filesystem collections using source_metadata
    # These documents have source_uri stored in the source_metadata table
    cursor = conn.execute(
        """
        SELECT
            d.id AS doc_id,
            d.source_collection_id,
            d.hash,
            d.modified_at,
            sm.source_uri
        FROM documents d
        JOIN source_collections sc ON sc.id = d.source_collection_id
        JOIN source_metadata sm ON sm.document_id = d.id
        WHERE d.active = 1
          AND d.resource_id IS NULL
          AND sc.source_type != 'filesystem'
        """
    )
    remote_docs = cursor.fetchall()

    for row in remote_docs:
        doc_id = row[0]
        collection_id = row[1]
        doc_hash = row[2]
        modified_at = row[3]
        source_uri = row[4]

        # Use source_uri as the Resource URI
        uri = source_uri

        # Insert Resource (use INSERT OR IGNORE in case of duplicate URIs)
        conn.execute(
            """
            INSERT OR IGNORE INTO resources (
                source_collection_id, uri, hash,
                load_status, load_method, loaded_at,
                index_state, index_method, indexed_at,
                source_modified_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'loaded', 'backfill', ?, 'indexed', 'backfill', ?, ?, ?, ?)
            """,
            (collection_id, uri, doc_hash, modified_at, modified_at, modified_at, now, now),
        )

        # Get the resource ID (either newly inserted or existing)
        resource_cursor = conn.execute(
            "SELECT id FROM resources WHERE source_collection_id = ? AND uri = ?",
            (collection_id, uri),
        )
        resource_row = resource_cursor.fetchone()
        if resource_row:
            resource_id = resource_row[0]
            # Update document with resource_id
            conn.execute(
                "UPDATE documents SET resource_id = ? WHERE id = ?",
                (resource_id, doc_id),
            )


def down(conn):
    """Rollback migration: remove resources table and FK.

    This removes the resource_id column from documents and drops the
    resources table entirely. All resource tracking data will be lost.
    """
    # Clear the foreign key references first
    conn.execute("UPDATE documents SET resource_id = NULL")

    # Drop the index on resource_id
    conn.execute("DROP INDEX IF EXISTS idx_documents_resource_id")

    # SQLite doesn't support DROP COLUMN directly in older versions,
    # but since resource_id is nullable and we've set it to NULL,
    # we can leave the column in place. For a clean rollback:
    # 1. Create new table without resource_id
    # 2. Copy data
    # 3. Drop old table
    # 4. Rename new table
    # For simplicity, we'll just leave the NULL column and drop resources table

    # Drop indexes on resources table
    conn.execute("DROP INDEX IF EXISTS idx_resources_collection_index_state")
    conn.execute("DROP INDEX IF EXISTS idx_resources_collection_load_status")

    # Drop the resources table
    conn.execute("DROP TABLE IF EXISTS resources")


def verify(conn) -> dict:
    """Verify migration integrity.

    Returns a dict with verification results:
    - active_docs_without_resource: Count of active documents missing resource_id
    - total_resources: Total resource count
    - orphaned_resources: Resources not linked to any document
    - all_ok: True if all active documents have resources

    The verification query from the implementation plan:
    SELECT COUNT(*) FROM documents WHERE resource_id IS NULL AND active = 1
    should return 0 or acceptably small.
    """
    # Count active documents without resource_id
    cursor = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE resource_id IS NULL AND active = 1"
    )
    active_docs_without_resource = cursor.fetchone()[0]

    # Count total resources
    cursor = conn.execute("SELECT COUNT(*) FROM resources")
    total_resources = cursor.fetchone()[0]

    # Count orphaned resources (resources not linked to any document)
    cursor = conn.execute(
        """
        SELECT COUNT(*) FROM resources r
        WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE d.resource_id = r.id
        )
        """
    )
    orphaned_resources = cursor.fetchone()[0]

    # Count active documents with resources
    cursor = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE resource_id IS NOT NULL AND active = 1"
    )
    active_docs_with_resource = cursor.fetchone()[0]

    return {
        "active_docs_without_resource": active_docs_without_resource,
        "active_docs_with_resource": active_docs_with_resource,
        "total_resources": total_resources,
        "orphaned_resources": orphaned_resources,
        "all_ok": active_docs_without_resource == 0,
    }
