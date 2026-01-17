"""Tests for collection CRUD operations."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.repositories.collections import CollectionRepository
from pmd.core.exceptions import SourceCollectionExistsError, SourceCollectionNotFoundError
from pmd.core.types import SourceCollection


class TestCollectionCreate:
    """Tests for collection creation."""

    def test_create_collection(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Create should return a Collection object."""
        collection = collection_repo.create("notes", str(tmp_path), "**/*.md")

        assert isinstance(collection, SourceCollection)
        assert collection.name == "notes"
        assert collection.pwd == str(tmp_path)
        assert collection.glob_pattern == "**/*.md"
        assert collection.id is not None

    def test_create_collection_default_pattern(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Create should use default glob pattern."""
        collection = collection_repo.create("notes", str(tmp_path))

        assert collection.glob_pattern == "**/*.md"

    def test_create_collection_custom_pattern(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Create should accept custom glob pattern."""
        collection = collection_repo.create("notes", str(tmp_path), "*.txt")

        assert collection.glob_pattern == "*.txt"

    def test_create_duplicate_raises(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Creating duplicate collection should raise."""
        collection_repo.create("notes", str(tmp_path))

        with pytest.raises(SourceCollectionExistsError, match="already exists"):
            collection_repo.create("notes", str(tmp_path))

    def test_create_sets_timestamps(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Create should set created_at and updated_at."""
        collection = collection_repo.create("notes", str(tmp_path))

        assert collection.created_at is not None
        assert collection.updated_at is not None
        assert collection.created_at == collection.updated_at


class TestCollectionGet:
    """Tests for collection retrieval."""

    def test_get_by_name_exists(self, collection_repo: CollectionRepository, tmp_path: Path):
        """get_by_name should return collection if exists."""
        created = collection_repo.create("notes", str(tmp_path))
        found = collection_repo.get_by_name("notes")

        assert found is not None
        assert found.id == created.id
        assert found.name == "notes"

    def test_get_by_name_not_exists(self, collection_repo: CollectionRepository):
        """get_by_name should return None if not exists."""
        found = collection_repo.get_by_name("nonexistent")

        assert found is None

    def test_get_by_id_exists(self, collection_repo: CollectionRepository, tmp_path: Path):
        """get_by_id should return collection if exists."""
        created = collection_repo.create("notes", str(tmp_path))
        found = collection_repo.get_by_id(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.name == "notes"

    def test_get_by_id_not_exists(self, collection_repo: CollectionRepository):
        """get_by_id should return None if not exists."""
        found = collection_repo.get_by_id(99999)

        assert found is None


class TestCollectionListAll:
    """Tests for listing collections."""

    def test_list_all_empty(self, collection_repo: CollectionRepository):
        """list_all should return empty list when no collections."""
        collections = collection_repo.list_all()

        assert collections == []

    def test_list_all_single(self, collection_repo: CollectionRepository, tmp_path: Path):
        """list_all should return single collection."""
        collection_repo.create("notes", str(tmp_path))
        collections = collection_repo.list_all()

        assert len(collections) == 1
        assert collections[0].name == "notes"

    def test_list_all_multiple(self, collection_repo: CollectionRepository, tmp_path: Path):
        """list_all should return all collections."""
        collection_repo.create("notes", str(tmp_path))
        collection_repo.create("docs", str(tmp_path))
        collection_repo.create("blog", str(tmp_path))

        collections = collection_repo.list_all()

        assert len(collections) == 3
        names = {c.name for c in collections}
        assert names == {"notes", "docs", "blog"}

    def test_list_all_ordered_by_name(self, collection_repo: CollectionRepository, tmp_path: Path):
        """list_all should return collections ordered by name."""
        collection_repo.create("zebra", str(tmp_path))
        collection_repo.create("apple", str(tmp_path))
        collection_repo.create("mango", str(tmp_path))

        collections = collection_repo.list_all()
        names = [c.name for c in collections]

        assert names == ["apple", "mango", "zebra"]


class TestCollectionRename:
    """Tests for renaming collections."""

    def test_rename_success(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Rename should update collection name."""
        collection = collection_repo.create("old-name", str(tmp_path))
        collection_repo.rename(collection.id, "new-name")

        # Old name should not exist
        assert collection_repo.get_by_name("old-name") is None

        # New name should exist
        found = collection_repo.get_by_name("new-name")
        assert found is not None
        assert found.id == collection.id

    def test_rename_updates_timestamp(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Rename should update updated_at timestamp."""
        collection = collection_repo.create("old-name", str(tmp_path))
        original_updated = collection.updated_at

        collection_repo.rename(collection.id, "new-name")
        found = collection_repo.get_by_name("new-name")

        assert found.updated_at >= original_updated

    def test_rename_same_name(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Rename to same name should succeed."""
        collection = collection_repo.create("same-name", str(tmp_path))
        collection_repo.rename(collection.id, "same-name")  # Should not raise

        found = collection_repo.get_by_name("same-name")
        assert found is not None

    def test_rename_to_existing_raises(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Rename to existing name should raise."""
        collection_repo.create("target-name", str(tmp_path))
        collection = collection_repo.create("source-name", str(tmp_path))

        with pytest.raises(SourceCollectionExistsError, match="already exists"):
            collection_repo.rename(collection.id, "target-name")

    def test_rename_nonexistent_raises(self, collection_repo: CollectionRepository):
        """Rename nonexistent collection should raise."""
        with pytest.raises(SourceCollectionNotFoundError):
            collection_repo.rename(99999, "new-name")


class TestCollectionRemove:
    """Tests for removing collections."""

    def test_remove_success(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Remove should delete collection."""
        collection = collection_repo.create("to-delete", str(tmp_path))
        collection_repo.remove(collection.id)

        assert collection_repo.get_by_id(collection.id) is None

    def test_remove_returns_counts(self, collection_repo: CollectionRepository, tmp_path: Path):
        """Remove should return deletion counts."""
        collection = collection_repo.create("to-delete", str(tmp_path))
        docs_deleted, hashes_cleaned = collection_repo.remove(collection.id)

        assert isinstance(docs_deleted, int)
        assert isinstance(hashes_cleaned, int)

    def test_remove_nonexistent_raises(self, collection_repo: CollectionRepository):
        """Remove nonexistent collection should raise."""
        with pytest.raises(SourceCollectionNotFoundError):
            collection_repo.remove(99999)

    def test_remove_cleans_up_documents(
        self,
        collection_repo: CollectionRepository,
        document_repo,
        tmp_path: Path,
    ):
        """Remove should delete associated documents."""
        from pmd.store.repositories.documents import DocumentRepository

        collection = collection_repo.create("with-docs", str(tmp_path))

        # Add some documents
        document_repo.add_or_update(collection.id, "doc1.md", "Doc 1", "content 1")
        document_repo.add_or_update(collection.id, "doc2.md", "Doc 2", "content 2")

        # Remove collection
        docs_deleted, _ = collection_repo.remove(collection.id)

        assert docs_deleted == 2


class TestCollectionUpdatePath:
    """Tests for updating collection path."""

    def test_update_path_success(self, collection_repo: CollectionRepository, tmp_path: Path):
        """update_path should update pwd."""
        collection = collection_repo.create("notes", str(tmp_path))
        new_path = str(tmp_path / "new-location")

        collection_repo.update_path(collection.id, new_path)

        found = collection_repo.get_by_id(collection.id)
        assert found.pwd == new_path

    def test_update_path_with_pattern(self, collection_repo: CollectionRepository, tmp_path: Path):
        """update_path should update glob pattern."""
        collection = collection_repo.create("notes", str(tmp_path), "**/*.md")
        new_path = str(tmp_path / "new-location")

        collection_repo.update_path(collection.id, new_path, "*.txt")

        found = collection_repo.get_by_id(collection.id)
        assert found.pwd == new_path
        assert found.glob_pattern == "*.txt"

    def test_update_path_nonexistent_raises(self, collection_repo: CollectionRepository):
        """update_path on nonexistent should raise."""
        with pytest.raises(SourceCollectionNotFoundError):
            collection_repo.update_path(99999, "/new/path")

    def test_update_path_updates_timestamp(self, collection_repo: CollectionRepository, tmp_path: Path):
        """update_path should update updated_at."""
        collection = collection_repo.create("notes", str(tmp_path))
        original_updated = collection.updated_at

        collection_repo.update_path(collection.id, str(tmp_path / "new"))

        found = collection_repo.get_by_id(collection.id)
        assert found.updated_at >= original_updated
