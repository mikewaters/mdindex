"""Tests for document storage and retrieval."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.documents import DocumentRepository
from pmd.store.collections import CollectionRepository
from pmd.core.types import DocumentResult
from pmd.utils.hashing import sha256_hash


class TestDocumentAddOrUpdate:
    """Tests for adding and updating documents."""

    def test_add_new_document(self, document_repo: DocumentRepository, sample_collection):
        """Adding new document should return is_new=True."""
        content = "# Hello\n\nWorld"
        doc, is_new = document_repo.add_or_update(
            sample_collection.id,
            "hello.md",
            "Hello",
            content,
        )

        assert is_new is True
        assert isinstance(doc, DocumentResult)
        assert doc.title == "Hello"
        assert doc.filepath == "hello.md"
        assert doc.body == content

    def test_update_existing_document(self, document_repo: DocumentRepository, sample_collection):
        """Updating existing document should return is_new=False."""
        # First add
        document_repo.add_or_update(
            sample_collection.id,
            "hello.md",
            "Hello",
            "Original content",
        )

        # Then update
        doc, is_new = document_repo.add_or_update(
            sample_collection.id,
            "hello.md",
            "Hello Updated",
            "Updated content",
        )

        assert is_new is False
        assert doc.title == "Hello Updated"
        assert doc.body == "Updated content"

    def test_add_document_computes_hash(self, document_repo: DocumentRepository, sample_collection):
        """Document hash should be SHA256 of content."""
        content = "Test content for hashing"
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            content,
        )

        expected_hash = sha256_hash(content)
        assert doc.hash == expected_hash

    def test_add_document_stores_body_length(self, document_repo: DocumentRepository, sample_collection):
        """Document should have correct body length."""
        content = "A" * 100
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            content,
        )

        assert doc.body_length == 100

    def test_add_document_sets_modified_at(self, document_repo: DocumentRepository, sample_collection):
        """Document should have modified_at timestamp."""
        doc, _ = document_repo.add_or_update(
            sample_collection.id,
            "test.md",
            "Test",
            "content",
        )

        assert doc.modified_at is not None


class TestDocumentContentAddressableStorage:
    """Tests for content-addressable storage."""

    def test_identical_content_shares_hash(self, document_repo: DocumentRepository, sample_collection):
        """Identical content should share the same hash."""
        content = "Shared content"

        doc1, _ = document_repo.add_or_update(
            sample_collection.id,
            "doc1.md",
            "Doc 1",
            content,
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id,
            "doc2.md",
            "Doc 2",
            content,
        )

        assert doc1.hash == doc2.hash

    def test_different_content_different_hash(self, document_repo: DocumentRepository, sample_collection):
        """Different content should have different hashes."""
        doc1, _ = document_repo.add_or_update(
            sample_collection.id,
            "doc1.md",
            "Doc 1",
            "Content 1",
        )
        doc2, _ = document_repo.add_or_update(
            sample_collection.id,
            "doc2.md",
            "Doc 2",
            "Content 2",
        )

        assert doc1.hash != doc2.hash

    def test_content_deduplicated_in_storage(self, document_repo: DocumentRepository, sample_collection, db: Database):
        """Content should be deduplicated in content table."""
        content = "Deduplicated content"

        document_repo.add_or_update(sample_collection.id, "doc1.md", "Doc 1", content)
        document_repo.add_or_update(sample_collection.id, "doc2.md", "Doc 2", content)

        # Should only have one content entry
        cursor = db.execute("SELECT COUNT(*) as count FROM content WHERE doc = ?", (content,))
        assert cursor.fetchone()["count"] == 1


class TestDocumentGet:
    """Tests for document retrieval."""

    def test_get_existing_document(self, document_repo: DocumentRepository, sample_collection):
        """get should return existing document."""
        content = "Test content"
        document_repo.add_or_update(sample_collection.id, "test.md", "Test", content)

        doc = document_repo.get(sample_collection.id, "test.md")

        assert doc is not None
        assert doc.title == "Test"
        assert doc.body == content

    def test_get_nonexistent_document(self, document_repo: DocumentRepository, sample_collection):
        """get should return None for nonexistent document."""
        doc = document_repo.get(sample_collection.id, "nonexistent.md")

        assert doc is None

    def test_get_inactive_document_returns_none(self, document_repo: DocumentRepository, sample_collection):
        """get should not return inactive (deleted) documents."""
        document_repo.add_or_update(sample_collection.id, "test.md", "Test", "content")
        document_repo.delete(sample_collection.id, "test.md")

        doc = document_repo.get(sample_collection.id, "test.md")

        assert doc is None

    def test_get_by_hash_existing(self, document_repo: DocumentRepository, sample_collection):
        """get_by_hash should return content for existing hash."""
        content = "Unique test content"
        doc, _ = document_repo.add_or_update(sample_collection.id, "test.md", "Test", content)

        retrieved = document_repo.get_by_hash(doc.hash)

        assert retrieved == content

    def test_get_by_hash_nonexistent(self, document_repo: DocumentRepository):
        """get_by_hash should return None for nonexistent hash."""
        retrieved = document_repo.get_by_hash("nonexistent_hash")

        assert retrieved is None


class TestDocumentListByCollection:
    """Tests for listing documents in a collection."""

    def test_list_empty_collection(self, document_repo: DocumentRepository, sample_collection):
        """list_by_collection should return empty list for empty collection."""
        docs = document_repo.list_by_collection(sample_collection.id)

        assert docs == []

    def test_list_documents_in_collection(self, document_repo: DocumentRepository, sample_collection):
        """list_by_collection should return all documents."""
        document_repo.add_or_update(sample_collection.id, "doc1.md", "Doc 1", "content 1")
        document_repo.add_or_update(sample_collection.id, "doc2.md", "Doc 2", "content 2")
        document_repo.add_or_update(sample_collection.id, "doc3.md", "Doc 3", "content 3")

        docs = document_repo.list_by_collection(sample_collection.id)

        assert len(docs) == 3
        paths = {d.filepath for d in docs}
        assert paths == {"doc1.md", "doc2.md", "doc3.md"}

    def test_list_excludes_inactive_by_default(self, document_repo: DocumentRepository, sample_collection):
        """list_by_collection should exclude inactive documents by default."""
        document_repo.add_or_update(sample_collection.id, "active.md", "Active", "content")
        document_repo.add_or_update(sample_collection.id, "deleted.md", "Deleted", "content")
        document_repo.delete(sample_collection.id, "deleted.md")

        docs = document_repo.list_by_collection(sample_collection.id)

        assert len(docs) == 1
        assert docs[0].filepath == "active.md"

    def test_list_includes_inactive_when_specified(self, document_repo: DocumentRepository, sample_collection):
        """list_by_collection should include inactive when active_only=False."""
        document_repo.add_or_update(sample_collection.id, "active.md", "Active", "content")
        document_repo.add_or_update(sample_collection.id, "deleted.md", "Deleted", "content")
        document_repo.delete(sample_collection.id, "deleted.md")

        docs = document_repo.list_by_collection(sample_collection.id, active_only=False)

        assert len(docs) == 2

    def test_list_ordered_by_path(self, document_repo: DocumentRepository, sample_collection):
        """list_by_collection should return documents ordered by path."""
        document_repo.add_or_update(sample_collection.id, "z.md", "Z", "content")
        document_repo.add_or_update(sample_collection.id, "a.md", "A", "content")
        document_repo.add_or_update(sample_collection.id, "m.md", "M", "content")

        docs = document_repo.list_by_collection(sample_collection.id)
        paths = [d.filepath for d in docs]

        assert paths == ["a.md", "m.md", "z.md"]


class TestDocumentDelete:
    """Tests for document deletion (soft delete)."""

    def test_delete_existing_document(self, document_repo: DocumentRepository, sample_collection):
        """delete should soft-delete existing document."""
        document_repo.add_or_update(sample_collection.id, "test.md", "Test", "content")

        result = document_repo.delete(sample_collection.id, "test.md")

        assert result is True
        assert document_repo.get(sample_collection.id, "test.md") is None

    def test_delete_nonexistent_document(self, document_repo: DocumentRepository, sample_collection):
        """delete should return False for nonexistent document."""
        result = document_repo.delete(sample_collection.id, "nonexistent.md")

        assert result is False

    def test_delete_already_deleted(self, document_repo: DocumentRepository, sample_collection):
        """delete should return False for already deleted document."""
        document_repo.add_or_update(sample_collection.id, "test.md", "Test", "content")
        document_repo.delete(sample_collection.id, "test.md")

        result = document_repo.delete(sample_collection.id, "test.md")

        assert result is False

    def test_delete_preserves_content(self, document_repo: DocumentRepository, sample_collection, db: Database):
        """delete should preserve content in content table."""
        content = "Preserved content"
        doc, _ = document_repo.add_or_update(sample_collection.id, "test.md", "Test", content)

        document_repo.delete(sample_collection.id, "test.md")

        # Content should still exist
        cursor = db.execute("SELECT doc FROM content WHERE hash = ?", (doc.hash,))
        assert cursor.fetchone()["doc"] == content


class TestDocumentCheckIfModified:
    """Tests for modification detection."""

    def test_check_new_document_is_modified(self, document_repo: DocumentRepository, sample_collection):
        """check_if_modified should return True for new document."""
        result = document_repo.check_if_modified(
            sample_collection.id,
            "new.md",
            "any_hash",
        )

        assert result is True

    def test_check_same_content_not_modified(self, document_repo: DocumentRepository, sample_collection):
        """check_if_modified should return False for same content."""
        content = "Same content"
        doc, _ = document_repo.add_or_update(sample_collection.id, "test.md", "Test", content)

        result = document_repo.check_if_modified(
            sample_collection.id,
            "test.md",
            doc.hash,
        )

        assert result is False

    def test_check_different_content_is_modified(self, document_repo: DocumentRepository, sample_collection):
        """check_if_modified should return True for different content."""
        document_repo.add_or_update(sample_collection.id, "test.md", "Test", "Original")

        result = document_repo.check_if_modified(
            sample_collection.id,
            "test.md",
            "different_hash",
        )

        assert result is True


class TestDocumentCount:
    """Tests for document counting."""

    def test_count_empty_collection(self, document_repo: DocumentRepository, sample_collection):
        """count_by_collection should return 0 for empty collection."""
        count = document_repo.count_by_collection(sample_collection.id)

        assert count == 0

    def test_count_documents(self, document_repo: DocumentRepository, sample_collection):
        """count_by_collection should return correct count."""
        document_repo.add_or_update(sample_collection.id, "doc1.md", "Doc 1", "content")
        document_repo.add_or_update(sample_collection.id, "doc2.md", "Doc 2", "content")

        count = document_repo.count_by_collection(sample_collection.id)

        assert count == 2

    def test_count_excludes_inactive_by_default(self, document_repo: DocumentRepository, sample_collection):
        """count_by_collection should exclude inactive by default."""
        document_repo.add_or_update(sample_collection.id, "active.md", "Active", "content")
        document_repo.add_or_update(sample_collection.id, "deleted.md", "Deleted", "content")
        document_repo.delete(sample_collection.id, "deleted.md")

        count = document_repo.count_by_collection(sample_collection.id)

        assert count == 1


class TestDocumentGetByPathPrefix:
    """Tests for retrieving documents by path prefix."""

    def test_get_by_prefix_no_matches(self, document_repo: DocumentRepository, sample_collection):
        """get_by_path_prefix should return empty list when no matches."""
        document_repo.add_or_update(sample_collection.id, "other/doc.md", "Doc", "content")

        docs = document_repo.get_by_path_prefix(sample_collection.id, "folder/")

        assert docs == []

    def test_get_by_prefix_matches(self, document_repo: DocumentRepository, sample_collection):
        """get_by_path_prefix should return matching documents."""
        document_repo.add_or_update(sample_collection.id, "folder/doc1.md", "Doc 1", "content")
        document_repo.add_or_update(sample_collection.id, "folder/doc2.md", "Doc 2", "content")
        document_repo.add_or_update(sample_collection.id, "other/doc.md", "Other", "content")

        docs = document_repo.get_by_path_prefix(sample_collection.id, "folder/")

        assert len(docs) == 2
        paths = {d.filepath for d in docs}
        assert paths == {"folder/doc1.md", "folder/doc2.md"}

    def test_get_by_prefix_ordered(self, document_repo: DocumentRepository, sample_collection):
        """get_by_path_prefix should return documents ordered by path."""
        document_repo.add_or_update(sample_collection.id, "folder/z.md", "Z", "content")
        document_repo.add_or_update(sample_collection.id, "folder/a.md", "A", "content")

        docs = document_repo.get_by_path_prefix(sample_collection.id, "folder/")
        paths = [d.filepath for d in docs]

        assert paths == ["folder/a.md", "folder/z.md"]


class TestDocumentContentLength:
    """Tests for content length retrieval."""

    def test_get_content_length_existing(self, document_repo: DocumentRepository, sample_collection):
        """get_content_length should return correct length."""
        content = "A" * 500
        document_repo.add_or_update(sample_collection.id, "test.md", "Test", content)

        length = document_repo.get_content_length(sample_collection.id, "test.md")

        assert length == 500

    def test_get_content_length_nonexistent(self, document_repo: DocumentRepository, sample_collection):
        """get_content_length should return None for nonexistent."""
        length = document_repo.get_content_length(sample_collection.id, "nonexistent.md")

        assert length is None
