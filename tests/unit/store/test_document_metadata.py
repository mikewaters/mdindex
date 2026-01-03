"""Tests for document metadata storage."""

import pytest
from datetime import datetime

from pmd.store.database import Database
from pmd.store.documents import DocumentRepository
from pmd.store.collections import CollectionRepository
from pmd.store.document_metadata import DocumentMetadataRepository, StoredDocumentMetadata


@pytest.fixture
def metadata_repo(db: Database) -> DocumentMetadataRepository:
    """Provide a DocumentMetadataRepository instance."""
    return DocumentMetadataRepository(db)


class TestDocumentMetadataUpsert:
    """Tests for upserting document metadata."""

    def test_upsert_new_metadata(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """Upserting new metadata should store it correctly."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id, "test.md", "Test", "content"
        )
        doc_id = _get_doc_id(document_repo, sample_collection.id, "test.md")

        metadata = StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="generic",
            tags={"python", "coding"},
            source_tags=["python", "coding"],
            extracted_at=datetime.utcnow().isoformat(),
        )
        metadata_repo.upsert(metadata)

        stored = metadata_repo.get_by_document(doc_id)
        assert stored is not None
        assert stored.profile_name == "generic"
        assert stored.tags == {"python", "coding"}

    def test_upsert_update_existing(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """Upserting existing metadata should update it."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id, "test.md", "Test", "content"
        )
        doc_id = _get_doc_id(document_repo, sample_collection.id, "test.md")

        # First upsert
        metadata1 = StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="generic",
            tags={"python"},
            source_tags=["python"],
            extracted_at=datetime.utcnow().isoformat(),
        )
        metadata_repo.upsert(metadata1)

        # Second upsert with different tags
        metadata2 = StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="obsidian",
            tags={"rust", "async"},
            source_tags=["#rust", "#async"],
            extracted_at=datetime.utcnow().isoformat(),
        )
        metadata_repo.upsert(metadata2)

        stored = metadata_repo.get_by_document(doc_id)
        assert stored.profile_name == "obsidian"
        assert stored.tags == {"rust", "async"}

    def test_upsert_maintains_junction_table(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
        db: Database,
    ):
        """Upserting should also update the document_tags junction table."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id, "test.md", "Test", "content"
        )
        doc_id = _get_doc_id(document_repo, sample_collection.id, "test.md")

        metadata = StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="generic",
            tags={"python", "coding", "test"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        )
        metadata_repo.upsert(metadata)

        # Check junction table directly
        cursor = db.execute(
            "SELECT tag FROM document_tags WHERE document_id = ? ORDER BY tag",
            (doc_id,),
        )
        tags = [row["tag"] for row in cursor.fetchall()]
        assert tags == ["coding", "python", "test"]


class TestDocumentMetadataGet:
    """Tests for retrieving document metadata."""

    def test_get_existing_metadata(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """get_by_document should return existing metadata."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id, "test.md", "Test", "content"
        )
        doc_id = _get_doc_id(document_repo, sample_collection.id, "test.md")

        metadata = StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="obsidian",
            tags={"topic/subtopic"},
            source_tags=["#topic/subtopic"],
            attributes={"title": "My Doc"},
            extracted_at="2024-01-01T00:00:00",
        )
        metadata_repo.upsert(metadata)

        stored = metadata_repo.get_by_document(doc_id)
        assert stored is not None
        assert stored.profile_name == "obsidian"
        assert stored.attributes == {"title": "My Doc"}

    def test_get_nonexistent_metadata(
        self,
        metadata_repo: DocumentMetadataRepository,
    ):
        """get_by_document should return None for nonexistent document."""
        stored = metadata_repo.get_by_document(99999)
        assert stored is None

    def test_get_tags(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """get_tags should return tag set for document."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id, "test.md", "Test", "content"
        )
        doc_id = _get_doc_id(document_repo, sample_collection.id, "test.md")

        metadata = StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="generic",
            tags={"python", "coding"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        )
        metadata_repo.upsert(metadata)

        tags = metadata_repo.get_tags(doc_id)
        assert tags == {"python", "coding"}

    def test_get_tags_empty(
        self,
        metadata_repo: DocumentMetadataRepository,
    ):
        """get_tags should return empty set for nonexistent document."""
        tags = metadata_repo.get_tags(99999)
        assert tags == set()


class TestDocumentMetadataFindByTag:
    """Tests for finding documents by tag."""

    def test_find_documents_with_tag(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """find_documents_with_tag should return matching documents."""
        # Create documents with different tags
        doc1, _ = document_repo.add_or_update(
            sample_collection.id, "doc1.md", "Doc 1", "content"
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id, "doc2.md", "Doc 2", "content"
        )
        doc3, _ = document_repo.add_or_update(
            sample_collection.id, "doc3.md", "Doc 3", "content"
        )

        doc1_id = _get_doc_id(document_repo, sample_collection.id, "doc1.md")
        doc2_id = _get_doc_id(document_repo, sample_collection.id, "doc2.md")
        doc3_id = _get_doc_id(document_repo, sample_collection.id, "doc3.md")

        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc1_id,
            profile_name="generic",
            tags={"python", "coding"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))
        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc2_id,
            profile_name="generic",
            tags={"python", "web"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))
        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc3_id,
            profile_name="generic",
            tags={"rust"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))

        python_docs = metadata_repo.find_documents_with_tag("python")
        assert set(python_docs) == {doc1_id, doc2_id}

        rust_docs = metadata_repo.find_documents_with_tag("rust")
        assert rust_docs == [doc3_id]

    def test_find_documents_with_tag_no_matches(
        self,
        metadata_repo: DocumentMetadataRepository,
    ):
        """find_documents_with_tag should return empty list when no matches."""
        docs = metadata_repo.find_documents_with_tag("nonexistent")
        assert docs == []

    def test_find_documents_with_any_tag(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """find_documents_with_any_tag should return documents with any matching tag."""
        doc1, _ = document_repo.add_or_update(
            sample_collection.id, "doc1.md", "Doc 1", "content"
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id, "doc2.md", "Doc 2", "content"
        )

        doc1_id = _get_doc_id(document_repo, sample_collection.id, "doc1.md")
        doc2_id = _get_doc_id(document_repo, sample_collection.id, "doc2.md")

        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc1_id,
            profile_name="generic",
            tags={"python"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))
        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc2_id,
            profile_name="generic",
            tags={"rust"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))

        docs = metadata_repo.find_documents_with_any_tag({"python", "rust"})
        assert set(docs) == {doc1_id, doc2_id}

    def test_find_documents_with_all_tags(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """find_documents_with_all_tags should return documents with ALL matching tags."""
        doc1, _ = document_repo.add_or_update(
            sample_collection.id, "doc1.md", "Doc 1", "content"
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id, "doc2.md", "Doc 2", "content"
        )

        doc1_id = _get_doc_id(document_repo, sample_collection.id, "doc1.md")
        doc2_id = _get_doc_id(document_repo, sample_collection.id, "doc2.md")

        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc1_id,
            profile_name="generic",
            tags={"python", "web", "api"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))
        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc2_id,
            profile_name="generic",
            tags={"python", "cli"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))

        # Only doc1 has both "python" AND "web"
        docs = metadata_repo.find_documents_with_all_tags({"python", "web"})
        assert docs == [doc1_id]


class TestDocumentMetadataDelete:
    """Tests for deleting document metadata."""

    def test_delete_existing_metadata(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """delete_by_document should remove metadata."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id, "test.md", "Test", "content"
        )
        doc_id = _get_doc_id(document_repo, sample_collection.id, "test.md")

        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="generic",
            tags={"python"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))

        result = metadata_repo.delete_by_document(doc_id)
        assert result is True
        assert metadata_repo.get_by_document(doc_id) is None

    def test_delete_nonexistent_metadata(
        self,
        metadata_repo: DocumentMetadataRepository,
    ):
        """delete_by_document should return False for nonexistent."""
        result = metadata_repo.delete_by_document(99999)
        assert result is False

    def test_delete_clears_junction_table(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
        db: Database,
    ):
        """delete_by_document should also clear junction table."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id, "test.md", "Test", "content"
        )
        doc_id = _get_doc_id(document_repo, sample_collection.id, "test.md")

        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc_id,
            profile_name="generic",
            tags={"python", "coding"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))

        metadata_repo.delete_by_document(doc_id)

        cursor = db.execute(
            "SELECT COUNT(*) as count FROM document_tags WHERE document_id = ?",
            (doc_id,),
        )
        assert cursor.fetchone()["count"] == 0


class TestDocumentMetadataAggregations:
    """Tests for tag aggregation functions."""

    def test_get_all_tags(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """get_all_tags should return all unique tags."""
        doc1, _ = document_repo.add_or_update(
            sample_collection.id, "doc1.md", "Doc 1", "content"
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id, "doc2.md", "Doc 2", "content"
        )

        doc1_id = _get_doc_id(document_repo, sample_collection.id, "doc1.md")
        doc2_id = _get_doc_id(document_repo, sample_collection.id, "doc2.md")

        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc1_id,
            profile_name="generic",
            tags={"python", "web"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))
        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc2_id,
            profile_name="generic",
            tags={"python", "cli"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))

        all_tags = metadata_repo.get_all_tags()
        assert all_tags == {"python", "web", "cli"}

    def test_count_documents_by_tag(
        self,
        metadata_repo: DocumentMetadataRepository,
        document_repo: DocumentRepository,
        sample_collection,
    ):
        """count_documents_by_tag should return correct counts."""
        doc1, _ = document_repo.add_or_update(
            sample_collection.id, "doc1.md", "Doc 1", "content"
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id, "doc2.md", "Doc 2", "content"
        )
        doc3, _ = document_repo.add_or_update(
            sample_collection.id, "doc3.md", "Doc 3", "content"
        )

        doc1_id = _get_doc_id(document_repo, sample_collection.id, "doc1.md")
        doc2_id = _get_doc_id(document_repo, sample_collection.id, "doc2.md")
        doc3_id = _get_doc_id(document_repo, sample_collection.id, "doc3.md")

        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc1_id,
            profile_name="generic",
            tags={"python"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))
        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc2_id,
            profile_name="generic",
            tags={"python", "web"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))
        metadata_repo.upsert(StoredDocumentMetadata(
            document_id=doc3_id,
            profile_name="generic",
            tags={"python", "web", "api"},
            source_tags=[],
            extracted_at=datetime.utcnow().isoformat(),
        ))

        counts = metadata_repo.count_documents_by_tag()
        assert counts["python"] == 3
        assert counts["web"] == 2
        assert counts["api"] == 1


def _get_doc_id(document_repo: DocumentRepository, collection_id: int, path: str) -> int:
    """Helper to get document ID from repository's database."""
    cursor = document_repo.db.execute(
        "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
        (collection_id, path),
    )
    return cursor.fetchone()["id"]
