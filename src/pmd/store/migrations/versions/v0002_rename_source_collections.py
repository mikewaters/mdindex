"""Rename collections table to source_collections.

Renames:
- Table: collections -> source_collections
- Column: documents.collection_id -> documents.source_collection_id
- Indexes: updated to reflect new names
"""

VERSION = 2
DESCRIPTION = "Rename collections to source_collections"


def up(conn):
    """Apply migration: rename collections to source_collections.

    Note: This migration is safe to run on databases that were created with
    the updated v0001 schema (which already uses source_collections). It checks
    for the existence of the old table before attempting the rename.
    """
    # Check if the old table exists (for existing databases)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='collections'"
    )
    old_table_exists = cursor.fetchone() is not None

    if not old_table_exists:
        # Database was created with updated v0001 schema, no rename needed
        return

    # Rename the main table
    conn.execute("ALTER TABLE collections RENAME TO source_collections")

    # Rename foreign key columns
    conn.execute(
        "ALTER TABLE documents RENAME COLUMN collection_id TO source_collection_id"
    )

    # Drop old indexes and create new ones with updated names
    conn.execute("DROP INDEX IF EXISTS idx_documents_collection")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_source_collection "
        "ON documents(source_collection_id)"
    )
