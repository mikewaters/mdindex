"""Pytest configuration and fixtures for integration tests."""

import pytest
from pathlib import Path

from pmd.core.config import Config
from pmd.services import ServiceContainer
from pmd.store.collections import CollectionRepository
from pmd.store.database import Database
from pmd.store.documents import DocumentRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.search import FTS5SearchRepository


@pytest.fixture
def test_corpus_path() -> Path:
    """Provide path to the test corpus directory."""
    return Path(__file__).parent.parent / "fixtures" / "test_corpus"


@pytest.fixture
def integration_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path for integration tests."""
    return tmp_path / "integration_test.db"


@pytest.fixture
def db(integration_db_path: Path) -> Database:
    """Provide a connected database instance for integration tests."""
    database = Database(integration_db_path)
    database.connect()
    yield database
    database.close()


@pytest.fixture
def config(integration_db_path: Path) -> Config:
    """Provide a Config instance pointing to the test database."""
    cfg = Config()
    cfg.db_path = integration_db_path
    return cfg


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
def test_corpus_collection(
    collection_repo: CollectionRepository,
    test_corpus_path: Path,
) -> "SourceCollection":
    """Create a collection pointing to the test corpus."""
    return collection_repo.create("test-corpus", str(test_corpus_path), "**/*.md")


@pytest.fixture
async def services(config: Config):
    """Provide a ServiceContainer instance for integration tests."""
    async with ServiceContainer(config) as svc:
        yield svc
