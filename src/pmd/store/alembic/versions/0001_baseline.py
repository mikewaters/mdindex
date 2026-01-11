"""Baseline schema with all core tables.

Creates the full PMD database schema with source_metadata columns merged
into the documents table:
- content: Content-addressable storage
- source_collections: Indexed directories or remote sources
- documents: File-to-content mappings with source metadata
- documents_fts: Full-text search index (FTS5)
- content_vectors: Vector embeddings metadata
- document_metadata: Extracted tags and attributes
- document_tags: Inverted index for tag lookups

Revision ID: 0001
Revises: None
Create Date: 2026-01-11
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create full baseline schema."""
    # Content-addressable storage
    op.create_table(
        "content",
        sa.Column("hash", sa.Text(), nullable=False),
        sa.Column("doc", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("hash"),
    )

    # Source collections (indexed directories or remote sources)
    op.create_table(
        "source_collections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("pwd", sa.Text(), nullable=False),
        sa.Column("glob_pattern", sa.Text(), nullable=False, server_default="**/*.md"),
        sa.Column(
            "source_type", sa.Text(), nullable=False, server_default="filesystem"
        ),
        sa.Column("source_config", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # Documents with merged source_metadata columns
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_collection_id", sa.Integer(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("hash", sa.Text(), nullable=False),
        sa.Column("active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("modified_at", sa.Text(), nullable=False),
        # Merged source_metadata columns
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("etag", sa.Text(), nullable=True),
        sa.Column("last_modified", sa.Text(), nullable=True),
        sa.Column("last_fetched_at", sa.Text(), nullable=True),
        sa.Column("fetch_duration_ms", sa.Integer(), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("content_type", sa.Text(), nullable=True),
        sa.Column("extra_metadata", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["hash"], ["content.hash"]),
        sa.ForeignKeyConstraint(["source_collection_id"], ["source_collections.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_collection_id", "path"),
    )

    # Vector embeddings metadata (no FK on hash - embeddings can be created independently)
    op.create_table(
        "content_vectors",
        sa.Column("hash", sa.Text(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("pos", sa.Integer(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("embedded_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("hash", "seq"),
    )

    # Document metadata (extracted tags, attributes)
    op.create_table(
        "document_metadata",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("profile_name", sa.Text(), nullable=False),
        sa.Column("tags_json", sa.Text(), nullable=False),
        sa.Column("source_tags_json", sa.Text(), nullable=False),
        sa.Column("attributes_json", sa.Text(), nullable=True),
        sa.Column("extracted_at", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id"),
    )

    # Document tags junction table (inverted index for fast tag lookups)
    op.create_table(
        "document_tags",
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.PrimaryKeyConstraint("document_id", "tag"),
    )

    # Create indexes for performance
    op.create_index(
        "idx_documents_source_collection", "documents", ["source_collection_id"]
    )
    op.create_index("idx_documents_hash", "documents", ["hash"])
    op.create_index("idx_documents_source_uri", "documents", ["source_uri"])
    op.create_index("idx_content_vectors_hash", "content_vectors", ["hash"])
    op.create_index("idx_document_tags_tag", "document_tags", ["tag"])

    # Full-text search index (FTS5 virtual table - SQLite specific)
    # Must use op.execute() since FTS5 is SQLite-specific syntax
    op.execute(
        """
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            path, body,
            tokenize='porter unicode61'
        )
        """
    )


def downgrade() -> None:
    """Drop all tables in reverse order."""
    # Drop FTS5 virtual table
    op.execute("DROP TABLE IF EXISTS documents_fts")

    # Drop indexes
    op.drop_index("idx_document_tags_tag", table_name="document_tags")
    op.drop_index("idx_content_vectors_hash", table_name="content_vectors")
    op.drop_index("idx_documents_source_uri", table_name="documents")
    op.drop_index("idx_documents_hash", table_name="documents")
    op.drop_index("idx_documents_source_collection", table_name="documents")

    # Drop tables in reverse dependency order
    op.drop_table("document_tags")
    op.drop_table("document_metadata")
    op.drop_table("content_vectors")
    op.drop_table("documents")
    op.drop_table("source_collections")
    op.drop_table("content")
