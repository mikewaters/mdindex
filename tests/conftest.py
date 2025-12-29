"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.collections import CollectionRepository
from pmd.store.documents import DocumentRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.search import FTS5SearchRepository


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path for tests."""
    return tmp_path / "test.db"


@pytest.fixture
def fixtures_dir() -> Path:
    """Provide path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def db(test_db_path: Path) -> Database:
    """Provide a connected database instance."""
    database = Database(test_db_path)
    database.connect()
    yield database
    database.close()


@pytest.fixture
def collection_repo(db: Database) -> CollectionRepository:
    """Provide a CollectionRepository instance."""
    return CollectionRepository(db)


@pytest.fixture
def document_repo(db: Database) -> DocumentRepository:
    """Provide a DocumentRepository instance."""
    return DocumentRepository(db)


@pytest.fixture
def embedding_repo(db: Database) -> EmbeddingRepository:
    """Provide an EmbeddingRepository instance."""
    return EmbeddingRepository(db)


@pytest.fixture
def fts_repo(db: Database) -> FTS5SearchRepository:
    """Provide a FTS5SearchRepository instance."""
    return FTS5SearchRepository(db)


# Legacy alias for backwards compatibility with existing tests
@pytest.fixture
def search_repo(fts_repo: FTS5SearchRepository) -> FTS5SearchRepository:
    """Provide a FTS5SearchRepository instance (legacy alias for fts_repo)."""
    return fts_repo


@pytest.fixture
def sample_collection(collection_repo: CollectionRepository, tmp_path: Path):
    """Create a sample collection for testing."""
    return collection_repo.create("test-collection", str(tmp_path), "**/*.md")


@pytest.fixture
def sample_document(
    document_repo: DocumentRepository,
    sample_collection,
) -> tuple:
    """Create a sample document for testing."""
    content = "# Test Document\n\nThis is test content for search testing."
    doc, is_new = document_repo.add_or_update(
        sample_collection.id,
        "test.md",
        "Test Document",
        content,
    )
    return doc, is_new, content
