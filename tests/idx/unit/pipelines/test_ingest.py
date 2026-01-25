"""Tests for idx.pipelines.ingest module."""

import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

from idx.core.settings import get_settings
from idx.ingest.pipelines import IngestPipeline
from idx.ingest.schemas import IngestDirectoryConfig, IngestObsidianConfig
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import FTSManager, create_fts_table
from idx.store.repositories import DatasetRepository, DocumentRepository


def _clear_pipeline_cache(dataset_names: list[str]) -> None:
    """Clear LlamaIndex pipeline cache for specific datasets.

    This ensures test isolation by removing persisted docstores
    that would otherwise cause documents to be skipped.
    """
    settings = get_settings()
    pipeline_dir = settings.cache_path / "pipeline_storage"
    for name in dataset_names:
        cache_path = pipeline_dir / name
        if cache_path.exists():
            shutil.rmtree(cache_path)






class TestIngestPipeline:
    """Tests for IngestPipeline class."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear pipeline cache before each test for isolation."""
        _clear_pipeline_cache(["test-docs", "test-vault", "my-dataset"])
        yield
        # Also clear after test
        _clear_pipeline_cache(["test-docs", "test-vault", "my-dataset"])

    @pytest.fixture(autouse=True)
    def disable_pipeline_cache(self) -> None:
        """Disable pipeline cache loading to ensure documents reach transforms.

        Without this, LlamaIndex's docstore would filter duplicates before
        they reach PersistenceTransform, causing skipped count to be 0.
        """
        with patch("idx.ingest.pipelines.load_pipeline", lambda name, pipeline: pipeline):
            yield

    @pytest.fixture(autouse=True)
    def use_mock_embedding(self, patched_embedding) -> None:
        """Use mock embedding model for all tests."""
        yield

    @pytest.fixture
    def test_db(self, tmp_path: Path):
        """Create a test database and patch get_session."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)
        create_fts_table(engine)

        factory = sessionmaker(bind=engine, expire_on_commit=False)

        @contextmanager
        def get_test_session():
            session = factory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        # Patch get_session to use our test database
        with patch("idx.ingest.pipelines.get_session", get_test_session):
            yield get_test_session

    @pytest.fixture
    def db_session(self, test_db):
        """Create a test database session for verification."""
        with test_db() as session:
            yield session

    @pytest.fixture
    def sample_directory(self, tmp_path: Path) -> Path:
        """Create a sample directory with test files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create some markdown files
        (docs_dir / "readme.md").write_text("# Readme\n\nThis is a test.")
        (docs_dir / "notes.md").write_text("# Notes\n\nSome notes here.")

        subdir = docs_dir / "subdir"
        subdir.mkdir()
        (subdir / "deep.md").write_text("# Deep\n\nNested file.")

        return docs_dir

    def test_ingest_creates_dataset(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Ingestion creates a new dataset."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.dataset_name == "test-docs"
        assert result.dataset_id > 0

        # Verify dataset exists
        repo = DatasetRepository(db_session)
        dataset = repo.get_by_name("test-docs")
        assert dataset is not None
        assert dataset.source_type == "directory"

    def test_ingest_creates_documents(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Ingestion creates documents for matching files."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.documents_created == 3  # readme.md, notes.md, subdir/deep.md
        assert result.documents_updated == 0
        assert result.documents_skipped == 0
        assert result.documents_failed == 0

        # Verify documents exist
        doc_repo = DocumentRepository(db_session)
        docs = doc_repo.list_by_dataset(result.dataset_id)
        assert len(docs) == 3

        paths = {doc.path for doc in docs}
        assert "readme.md" in paths
        assert "notes.md" in paths
        assert "subdir/deep.md" in paths

    def test_ingest_updates_fts(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Ingestion updates the FTS index."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        # Verify FTS index
        fts = FTSManager(db_session)
        assert fts.count() == 3

        # Search should find results
        results = fts.search("readme")
        assert len(results) >= 1
        assert any("readme.md" in r.path for r in results)

    def test_ingest_skips_unchanged_documents(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Re-ingestion skips unchanged documents."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()

        # First ingestion
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 3

        # Second ingestion - should skip all
        result2 = pipeline.ingest(config)
        assert result2.documents_created == 0
        assert result2.documents_updated == 0
        assert result2.documents_skipped == 3

    def test_ingest_updates_changed_documents(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Re-ingestion updates changed documents."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()

        # First ingestion
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 3

        # Modify a file
        (sample_directory / "readme.md").write_text("# Updated Readme\n\nNew content.")

        # Second ingestion - should update one
        result2 = pipeline.ingest(config)
        assert result2.documents_created == 0
        assert result2.documents_updated == 1
        assert result2.documents_skipped == 2

    def test_ingest_force_updates_all(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Force mode updates all documents."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()

        # First ingestion
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 3

        # Force ingestion - should update all
        config_force = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
            force=True,
        )
        result2 = pipeline.ingest(config_force)
        assert result2.documents_created == 0
        assert result2.documents_updated == 3
        assert result2.documents_skipped == 0

    def test_ingest_with_exclusion_patterns(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Exclusion patterns are respected."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md", "!**/subdir/**"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        # Should only include files not in subdir
        assert result.documents_created == 2

        doc_repo = DocumentRepository(db_session)
        docs = doc_repo.list_by_dataset(result.dataset_id)
        paths = {doc.path for doc in docs}
        assert "readme.md" in paths
        assert "notes.md" in paths
        assert "subdir/deep.md" not in paths

    def test_ingest_normalizes_dataset_name(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Dataset name is normalized."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="My Test Docs!",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.dataset_name == "my-test-docs"

    def test_ingest_reuses_existing_dataset(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """Ingestion reuses existing dataset with same name."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()

        result1 = pipeline.ingest(config)
        result2 = pipeline.ingest(config)

        assert result1.dataset_id == result2.dataset_id

    def test_ingest_result_properties(
        self, test_db, db_session, sample_directory: Path
    ) -> None:
        """IngestResult properties work correctly."""
        config = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.total_processed == 3
        assert result.success is True
        assert result.completed_at is not None
        assert result.started_at <= result.completed_at

    def test_ingest_handles_empty_directory(
        self, test_db, db_session, tmp_path: Path
    ) -> None:
        """Ingestion handles empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = IngestDirectoryConfig(
            source_path=empty_dir,
            dataset_name="empty-dataset",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.documents_created == 0
        assert result.success is True

    def test_ingest_handles_missing_directory(
        self, test_db, db_session, tmp_path: Path
    ) -> None:
        """Ingestion raises error for missing directory."""
        from pydantic import ValidationError

        missing_dir = tmp_path / "nonexistent"

        # Validation happens at config creation time
        with pytest.raises(ValidationError, match="does not exist"):
            IngestDirectoryConfig(
                source_path=missing_dir,
                dataset_name="test",
                patterns=["**/*.md"],
            )

    # NOTE: Stale document detection tests were removed.
    # Stale document handling is now in idx.store.cleanup module.
    # See tests for cleanup_stale_documents() function instead.


class TestObsidianIngest:
    """Tests for Obsidian vault ingestion."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear pipeline cache before each test for isolation."""
        _clear_pipeline_cache(["my-vault", "test-vault"])
        yield
        # Also clear after test
        _clear_pipeline_cache(["my-vault", "test-vault"])

    @pytest.fixture(autouse=True)
    def disable_pipeline_cache(self) -> None:
        """Disable pipeline cache loading to ensure documents reach transforms."""
        with patch("idx.ingest.pipelines.load_pipeline", lambda name, pipeline: pipeline):
            yield

    @pytest.fixture(autouse=True)
    def use_mock_embedding(self, patched_embedding) -> None:
        """Use mock embedding model for all tests."""
        yield

    @pytest.fixture
    def test_db(self, tmp_path: Path):
        """Create a test database and patch get_session."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)
        create_fts_table(engine)

        factory = sessionmaker(bind=engine, expire_on_commit=False)

        @contextmanager
        def get_test_session():
            session = factory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        # Patch get_session to use our test database
        with patch("idx.ingest.pipelines.get_session", get_test_session):
            yield get_test_session

    @pytest.fixture
    def db_session(self, test_db):
        """Create a test database session for verification."""
        with test_db() as session:
            yield session

    @pytest.fixture
    def obsidian_vault(self, tmp_path: Path) -> Path:
        """Create a sample Obsidian vault."""
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()

        # Create .obsidian directory (required for valid vault)
        obsidian_dir = vault_dir / ".obsidian"
        obsidian_dir.mkdir()
        (obsidian_dir / "app.json").write_text("{}")

        # Create markdown files with frontmatter
        (vault_dir / "note1.md").write_text(
            """---
tags:
  - work
  - important
aliases:
  - First Note
---

# Note 1

This is the first note.
"""
        )

        (vault_dir / "note2.md").write_text(
            """---
tags: personal
---

# Note 2

This is a personal note.
"""
        )

        # Note without frontmatter
        (vault_dir / "plain.md").write_text("# Plain Note\n\nNo frontmatter here.")

        # Subdirectory
        subdir = vault_dir / "folder"
        subdir.mkdir()
        (subdir / "nested.md").write_text(
            """---
tags:
  - nested
  - folder
---

# Nested Note

In a subfolder.
"""
        )

        return vault_dir

    def test_obsidian_ingest_creates_dataset(
        self, test_db, db_session, obsidian_vault: Path
    ) -> None:
        """Obsidian ingestion creates a dataset."""
        config = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.dataset_name == "my-vault"
        assert result.dataset_id > 0

        repo = DatasetRepository(db_session)
        dataset = repo.get_by_name("my-vault")
        assert dataset is not None
        assert dataset.source_type == "obsidian"

    def test_obsidian_ingest_creates_documents(
        self, test_db, db_session, obsidian_vault: Path
    ) -> None:
        """Obsidian ingestion creates documents."""
        config = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.documents_created == 4  # note1, note2, plain, nested
        assert result.documents_failed == 0

    def test_obsidian_ingest_extracts_metadata(
        self, test_db, db_session, obsidian_vault: Path
    ) -> None:
        """Obsidian ingestion extracts frontmatter metadata."""
        import json

        config = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        doc_repo = DocumentRepository(db_session)
        doc1 = doc_repo.get_by_path(result.dataset_id, "note1.md")
        assert doc1 is not None
        assert doc1.metadata_json is not None

        metadata = json.loads(doc1.metadata_json)
        # Frontmatter is stored under "frontmatter" key
        assert "frontmatter" in metadata
        frontmatter = metadata["frontmatter"]
        assert "tags" in frontmatter
        assert "work" in frontmatter["tags"]
        assert "important" in frontmatter["tags"]
        assert "aliases" in frontmatter
        assert "First Note" in frontmatter["aliases"]

    def test_obsidian_ingest_handles_no_frontmatter(
        self, test_db, db_session, obsidian_vault: Path
    ) -> None:
        """Obsidian ingestion handles documents without frontmatter."""
        import json

        config = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        doc_repo = DocumentRepository(db_session)
        plain_doc = doc_repo.get_by_path(result.dataset_id, "plain.md")
        assert plain_doc is not None
        # Metadata should be empty or minimal - frontmatter should be absent or empty
        if plain_doc.metadata_json:
            metadata = json.loads(plain_doc.metadata_json)
            frontmatter = metadata.get("frontmatter", {})
            assert not frontmatter.get("tags")
            assert not frontmatter.get("aliases")

    def test_obsidian_ingest_updates_fts(
        self, test_db, db_session, obsidian_vault: Path
    ) -> None:
        """Obsidian ingestion updates the FTS index."""
        config = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        fts = FTSManager(db_session)
        assert fts.count() == 4

        # Search should find results
        results = fts.search("nested")
        assert len(results) >= 1

    def test_obsidian_ingest_force_mode(
        self, test_db, db_session, obsidian_vault: Path
    ) -> None:
        """Obsidian force mode updates all documents."""
        config = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
        )
        pipeline = IngestPipeline()

        # First ingestion
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 4

        # Force ingestion
        config_force = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
            force=True,
        )
        result2 = pipeline.ingest(config_force)
        assert result2.documents_updated == 4
        assert result2.documents_skipped == 0

    def test_obsidian_ingest_invalid_vault(
        self, test_db, db_session, tmp_path: Path
    ) -> None:
        """Obsidian ingestion raises error for invalid vault."""
        from pydantic import ValidationError

        not_a_vault = tmp_path / "not_vault"
        not_a_vault.mkdir()

        # Validation happens at config creation time
        with pytest.raises(ValidationError, match="missing .obsidian"):
            IngestObsidianConfig(
                source_path=not_a_vault,
                dataset_name="test",
            )
