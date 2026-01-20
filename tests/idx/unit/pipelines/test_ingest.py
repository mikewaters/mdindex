"""Tests for idx.pipelines.ingest module."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

from idx.pipelines.ingest import IngestPipeline, compute_content_hash, source_doc_to_llama_doc
from idx.pipelines.schemas import IngestDirectoryConfig, IngestObsidianConfig
from idx.source.directory import SourceDocument
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import FTSManager, create_fts_table
from idx.store.repositories import DatasetRepository, DocumentRepository


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_hash_basic_text(self) -> None:
        """Hash is computed correctly for basic text."""
        hash1 = compute_content_hash("Hello, World!")
        hash2 = compute_content_hash("Hello, World!")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_hash_different_content(self) -> None:
        """Different content produces different hashes."""
        hash1 = compute_content_hash("Hello")
        hash2 = compute_content_hash("World")
        assert hash1 != hash2

    def test_hash_empty_string(self) -> None:
        """Empty string produces valid hash."""
        hash_val = compute_content_hash("")
        assert len(hash_val) == 64

    def test_hash_unicode(self) -> None:
        """Unicode content is hashed correctly."""
        hash_val = compute_content_hash("Hello 世界 ")
        assert len(hash_val) == 64


class TestSourceDocToLlamaDoc:
    """Tests for source_doc_to_llama_doc function."""

    def test_basic_conversion(self) -> None:
        """SourceDocument is converted to LlamaIndex Document."""
        from datetime import datetime, timezone

        source_doc = SourceDocument(
            path=Path("/tmp/docs/test.md"),
            relative_path="test.md",
            last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
            content="# Test\n\nHello world.",
            etag="abc123",
        )

        llama_doc = source_doc_to_llama_doc(source_doc)

        assert llama_doc.text == "# Test\n\nHello world."
        assert llama_doc.doc_id == "test.md"
        assert llama_doc.metadata["relative_path"] == "test.md"
        assert llama_doc.metadata["file_path"] == "/tmp/docs/test.md"
        assert llama_doc.metadata["etag"] == "abc123"

    def test_extra_metadata(self) -> None:
        """Extra metadata is merged into the document."""
        source_doc = SourceDocument(
            path=Path("/tmp/docs/test.md"),
            relative_path="test.md",
            content="Hello",
        )

        llama_doc = source_doc_to_llama_doc(
            source_doc,
            extra_metadata={"custom_key": "custom_value"},
        )

        assert llama_doc.metadata["custom_key"] == "custom_value"


class TestIngestPipeline:
    """Tests for IngestPipeline class."""

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
        with patch("idx.pipelines.ingest.get_session", get_test_session):
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
        result = pipeline.ingest_directory(config)

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
        result = pipeline.ingest_directory(config)

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
        result = pipeline.ingest_directory(config)

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
        result1 = pipeline.ingest_directory(config)
        assert result1.documents_created == 3

        # Second ingestion - should skip all
        result2 = pipeline.ingest_directory(config)
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
        result1 = pipeline.ingest_directory(config)
        assert result1.documents_created == 3

        # Modify a file
        (sample_directory / "readme.md").write_text("# Updated Readme\n\nNew content.")

        # Second ingestion - should update one
        result2 = pipeline.ingest_directory(config)
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
        result1 = pipeline.ingest_directory(config)
        assert result1.documents_created == 3

        # Force ingestion - should update all
        config_force = IngestDirectoryConfig(
            source_path=sample_directory,
            dataset_name="test-docs",
            patterns=["**/*.md"],
            force=True,
        )
        result2 = pipeline.ingest_directory(config_force)
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
        result = pipeline.ingest_directory(config)

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
        result = pipeline.ingest_directory(config)

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

        result1 = pipeline.ingest_directory(config)
        result2 = pipeline.ingest_directory(config)

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
        result = pipeline.ingest_directory(config)

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
        result = pipeline.ingest_directory(config)

        assert result.documents_created == 0
        assert result.success is True

    def test_ingest_handles_missing_directory(
        self, test_db, db_session, tmp_path: Path
    ) -> None:
        """Ingestion raises error for missing directory."""
        missing_dir = tmp_path / "nonexistent"

        config = IngestDirectoryConfig(
            source_path=missing_dir,
            dataset_name="test",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()

        with pytest.raises(FileNotFoundError):
            pipeline.ingest_directory(config)

    # NOTE: Stale document detection tests were removed.
    # Stale document handling is now in idx.store.cleanup module.
    # See tests for cleanup_stale_documents() function instead.


class TestObsidianIngest:
    """Tests for Obsidian vault ingestion."""

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
        with patch("idx.pipelines.ingest.get_session", get_test_session):
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
        result = pipeline.ingest_obsidian(config)

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
        result = pipeline.ingest_obsidian(config)

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
        result = pipeline.ingest_obsidian(config)

        doc_repo = DocumentRepository(db_session)
        doc1 = doc_repo.get_by_path(result.dataset_id, "note1.md")
        assert doc1 is not None
        assert doc1.metadata_json is not None

        metadata = json.loads(doc1.metadata_json)
        assert "tags" in metadata
        assert "work" in metadata["tags"]
        assert "important" in metadata["tags"]
        assert "aliases" in metadata
        assert "First Note" in metadata["aliases"]

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
        result = pipeline.ingest_obsidian(config)

        doc_repo = DocumentRepository(db_session)
        plain_doc = doc_repo.get_by_path(result.dataset_id, "plain.md")
        assert plain_doc is not None
        # Metadata should be empty or minimal
        if plain_doc.metadata_json:
            metadata = json.loads(plain_doc.metadata_json)
            assert not metadata.get("tags")
            assert not metadata.get("aliases")

    def test_obsidian_ingest_updates_fts(
        self, test_db, db_session, obsidian_vault: Path
    ) -> None:
        """Obsidian ingestion updates the FTS index."""
        config = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest_obsidian(config)

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
        result1 = pipeline.ingest_obsidian(config)
        assert result1.documents_created == 4

        # Force ingestion
        config_force = IngestObsidianConfig(
            source_path=obsidian_vault,
            dataset_name="my-vault",
            force=True,
        )
        result2 = pipeline.ingest_obsidian(config_force)
        assert result2.documents_updated == 4
        assert result2.documents_skipped == 0

    def test_obsidian_ingest_invalid_vault(
        self, test_db, db_session, tmp_path: Path
    ) -> None:
        """Obsidian ingestion raises error for invalid vault."""
        not_a_vault = tmp_path / "not_vault"
        not_a_vault.mkdir()

        config = IngestObsidianConfig(
            source_path=not_a_vault,
            dataset_name="test",
        )
        pipeline = IngestPipeline()

        with pytest.raises(ValueError, match="missing .obsidian"):
            pipeline.ingest_obsidian(config)
