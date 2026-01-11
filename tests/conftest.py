"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.collections import SourceCollectionRepository
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
def source_collection_repo(db: Database) -> SourceCollectionRepository:
    """Provide a SourceCollectionRepository instance."""
    return SourceCollectionRepository(db)


# Backwards compatibility alias
@pytest.fixture
def collection_repo(source_collection_repo: SourceCollectionRepository) -> SourceCollectionRepository:
    """Provide a SourceCollectionRepository instance (deprecated alias for source_collection_repo)."""
    return source_collection_repo


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
def sample_source_collection(source_collection_repo: SourceCollectionRepository, tmp_path: Path):
    """Create a sample source collection for testing."""
    return source_collection_repo.create("test-collection", str(tmp_path), "**/*.md")


# Backwards compatibility alias
@pytest.fixture
def sample_collection(sample_source_collection):
    """Create a sample collection for testing (deprecated alias for sample_source_collection)."""
    return sample_source_collection


@pytest.fixture
def sample_document(
    document_repo: DocumentRepository,
    sample_source_collection,
) -> tuple:
    """Create a sample document for testing."""
    content = "# Test Document\n\nThis is test content for search testing."
    doc, is_new = document_repo.add_or_update(
        sample_source_collection.id,
        "test.md",
        "Test Document",
        content,
    )
    return doc, is_new, content
