"""Integration tests for ingest pipeline + FTS search.

Tests end-to-end flow: ingest dataset, refresh with add/change/delete,
verify FTS results and soft-delete behavior.
"""

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from idx.core.settings import get_settings
from idx.ingest.pipelines import IngestPipeline
from idx.ingest.schemas import IngestDirectoryConfig, IngestObsidianConfig
from idx.search.fts import FTSSearch
from idx.search.models import SearchCriteria
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import create_fts_table


def _clear_pipeline_cache(dataset_names: list[str]) -> None:
    """Clear LlamaIndex pipeline cache for specific datasets."""
    settings = get_settings()
    pipeline_dir = settings.cache_path / "pipeline_storage"
    for name in dataset_names:
        cache_path = pipeline_dir / name
        if cache_path.exists():
            shutil.rmtree(cache_path)


@pytest.fixture
def test_engine(tmp_path: Path) -> Engine:
    """Create a temporary database and return the engine."""
    db_path = tmp_path / "test.db"
    engine = create_engine_for_path(db_path)
    Base.metadata.create_all(engine)
    create_fts_table(engine)
    return engine


@pytest.fixture
def session_factory(test_engine: Engine):
    """Create a session factory for the test database."""
    return sessionmaker(bind=test_engine, expire_on_commit=False)


@contextmanager
def create_session(factory) -> Generator[Session, None, None]:
    """Create a session that auto-commits on exit."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear pipeline cache before and after each test for isolation."""
    dataset_names = ["test-vault", "obsidian-vault", "vault1", "vault2"]
    _clear_pipeline_cache(dataset_names)
    yield
    _clear_pipeline_cache(dataset_names)


@pytest.fixture(autouse=True)
def disable_pipeline_cache() -> None:
    """Disable pipeline cache loading to ensure documents reach transforms.

    Without this, LlamaIndex's docstore would filter duplicates before
    they reach PersistenceTransform, causing skipped count to be 0.
    """
    with patch("idx.ingest.pipelines.load_pipeline", lambda name, pipeline: pipeline):
        yield


@pytest.fixture(autouse=True)
def use_mock_embedding(patched_embedding) -> None:
    """Use mock embedding model for all tests."""
    yield


@pytest.fixture
def patched_get_session(session_factory):
    """Patch get_session to use the test database."""
    @contextmanager
    def get_test_session():
        with create_session(session_factory) as session:
            yield session

    with patch("idx.ingest.pipelines.get_session", get_test_session):
        yield get_test_session


@pytest.fixture
def sample_vault(tmp_path: Path) -> Path:
    """Create a sample Obsidian vault with markdown files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create .obsidian directory to mark as vault
    (vault / ".obsidian").mkdir()

    # Create some markdown files
    (vault / "note1.md").write_text("""---
title: Python Tutorial
tags: [python, tutorial]
---

# Python Tutorial

This is a tutorial about Python programming.
Learn about functions, classes, and modules.
""")

    (vault / "note2.md").write_text("""---
title: JavaScript Guide
tags: [javascript, web]
---

# JavaScript Guide

This guide covers JavaScript basics.
Learn about async/await and promises.
""")

    (vault / "note3.md").write_text("""---
title: Database Design
tags: [sql, database]
---

# Database Design

Learn about SQL and database normalization.
Covers relational databases and indexes.
""")

    # Create a subdirectory with more files
    subdir = vault / "projects"
    subdir.mkdir()

    (subdir / "project1.md").write_text("""---
title: My Python Project
---

# My Python Project

Building a CLI tool with Python.
Uses argparse and pathlib.
""")

    return vault


class TestIngestAndSearch:
    """Integration tests for ingest + FTS search flow."""

    def test_ingest_directory_then_search(
        self,
        patched_get_session,
        session_factory,
        sample_vault: Path,
    ) -> None:
        """Ingest directory and verify FTS search works."""
        # Ingest the directory
        pipeline = IngestPipeline()
        config = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )

        result = pipeline.ingest(config)

        assert result.documents_created == 4
        assert result.documents_failed == 0

        # Search for Python content
        with create_session(session_factory) as session:
            search = FTSSearch(session)
            results = search.search(
                SearchCriteria(query="python", limit=10)
            )

            assert len(results.results) >= 2  # note1.md and project1.md
            paths = {r.path for r in results.results}
            assert "note1.md" in paths or "projects/project1.md" in paths

    def test_search_with_dataset_filter(
        self,
        patched_get_session,
        session_factory,
        sample_vault: Path,
        tmp_path: Path,
    ) -> None:
        """Search results can be filtered by dataset."""
        pipeline = IngestPipeline()

        # Ingest first vault
        config1 = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="vault1",
            patterns=["**/*.md"],
        )
        pipeline.ingest(config1)

        # Create and ingest second vault
        vault2 = tmp_path / "vault2"
        vault2.mkdir()
        (vault2 / ".obsidian").mkdir()
        (vault2 / "other.md").write_text("# Other Python Note\n\nMore python content here.")

        config2 = IngestDirectoryConfig(
            source_path=vault2,
            dataset_name="vault2",
            patterns=["**/*.md"],
        )
        pipeline.ingest(config2)

        # Search all datasets
        with create_session(session_factory) as session:
            search = FTSSearch(session)

            # Search without filter
            all_results = search.search(
                SearchCriteria(query="python", limit=10)
            )

            # Search with filter
            filtered_results = search.search(
                SearchCriteria(query="python", dataset_name="vault1", limit=10)
            )

            # Unfiltered should have results from both
            assert len(all_results.results) >= 2

            # Filtered should only have vault1 results
            for r in filtered_results.results:
                assert r.dataset_name == "vault1"


class TestRefreshBehavior:
    """Tests for refresh/re-ingest behavior."""

    def test_refresh_detects_unchanged_files(
        self,
        patched_get_session,
        sample_vault: Path,
    ) -> None:
        """Re-ingesting unchanged files skips them."""
        config = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )

        pipeline = IngestPipeline()

        # First ingest
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 4

        # Second ingest (no changes)
        result2 = pipeline.ingest(config)
        assert result2.documents_created == 0
        assert result2.documents_skipped == 4

    def test_refresh_detects_modified_files(
        self,
        patched_get_session,
        sample_vault: Path,
    ) -> None:
        """Re-ingesting modified files updates them."""
        config = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )

        pipeline = IngestPipeline()

        # First ingest
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 4

        # Modify a file
        import time
        time.sleep(0.1)  # Ensure mtime changes
        note1 = sample_vault / "note1.md"
        note1.write_text(note1.read_text() + "\n\nNew content added!")

        # Second ingest
        result2 = pipeline.ingest(config)
        assert result2.documents_updated >= 1
        assert result2.documents_skipped == 3

    def test_refresh_detects_added_files(
        self,
        patched_get_session,
        sample_vault: Path,
    ) -> None:
        """Re-ingesting with new files adds them."""
        config = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )

        pipeline = IngestPipeline()

        # First ingest
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 4

        # Add a new file
        (sample_vault / "new_note.md").write_text("# New Note\n\nBrand new content.")

        # Second ingest
        result2 = pipeline.ingest(config)
        assert result2.documents_created == 1
        assert result2.documents_skipped == 4


class TestSoftDeleteBehavior:
    """Tests for soft-delete and stale document handling.

    Note: Stale document detection has moved to idx.store.cleanup module.
    These tests now use cleanup_stale_documents() for marking stale docs.
    """

    def test_deleted_files_are_soft_deleted(
        self,
        patched_get_session,
        session_factory,
        sample_vault: Path,
    ) -> None:
        """Files removed from disk are cleaned up from database."""
        from idx.store.cleanup import cleanup_stale_documents

        config = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )

        pipeline = IngestPipeline()

        # First ingest
        result1 = pipeline.ingest(config)
        assert result1.documents_created == 4

        # Delete a file
        (sample_vault / "note2.md").unlink()

        # Second ingest (processes existing files)
        pipeline.ingest(config)

        # Run cleanup to remove stale documents
        with create_session(session_factory) as session:
            stale_count = cleanup_stale_documents(
                session,
                result1.dataset_id,
                source_path=sample_vault,
                patterns=["**/*.md"],
            )
            assert stale_count == 1

        # Check database - document should be deleted (hard delete)
        with create_session(session_factory) as session:
            result = session.execute(
                text("SELECT path FROM documents WHERE path LIKE '%note2.md'")
            )
            row = result.fetchone()
            assert row is None  # Document was hard-deleted

    def test_deleted_file_not_in_search(
        self,
        patched_get_session,
        session_factory,
        sample_vault: Path,
    ) -> None:
        """Deleted documents don't appear in search results after cleanup."""
        from idx.store.cleanup import cleanup_stale_documents

        config = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )

        pipeline = IngestPipeline()

        # Ingest
        result = pipeline.ingest(config)

        # Verify JavaScript file is searchable
        with create_session(session_factory) as session:
            search = FTSSearch(session)
            results = search.search(SearchCriteria(query="javascript", limit=10))
            assert len(results.results) >= 1

        # Delete the JavaScript file
        (sample_vault / "note2.md").unlink()

        # Re-ingest and cleanup stale
        pipeline.ingest(config)
        with create_session(session_factory) as session:
            cleanup_stale_documents(
                session,
                result.dataset_id,
                source_path=sample_vault,
                patterns=["**/*.md"],
            )

        # Search again - should not find JavaScript
        with create_session(session_factory) as session:
            search = FTSSearch(session)
            results = search.search(SearchCriteria(query="javascript", limit=10))
            # note2.md was deleted, so no JavaScript results
            assert len(results.results) == 0

    def test_reappeared_file_is_reindexed(
        self,
        patched_get_session,
        session_factory,
        sample_vault: Path,
    ) -> None:
        """File that reappears after cleanup is re-indexed."""
        from idx.store.cleanup import cleanup_stale_documents

        config = IngestDirectoryConfig(
            source_path=sample_vault,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )

        pipeline = IngestPipeline()

        # First ingest
        result = pipeline.ingest(config)

        # Save content and delete
        note2_path = sample_vault / "note2.md"
        original_content = note2_path.read_text()
        note2_path.unlink()

        # Ingest and cleanup to delete stale docs
        pipeline.ingest(config)
        with create_session(session_factory) as session:
            stale_count = cleanup_stale_documents(
                session,
                result.dataset_id,
                source_path=sample_vault,
                patterns=["**/*.md"],
            )
            assert stale_count == 1

        # Restore the file
        note2_path.write_text(original_content)

        # Ingest again - should create new document
        result3 = pipeline.ingest(config)
        # Should be created (since it was hard-deleted and now reappears)
        assert result3.documents_created >= 1

        # Verify searchable again
        with create_session(session_factory) as session:
            search = FTSSearch(session)
            results = search.search(SearchCriteria(query="javascript", limit=10))
            assert len(results.results) >= 1


class TestObsidianIngest:
    """Tests for Obsidian-specific ingest behavior."""

    def test_ingest_obsidian_extracts_frontmatter(
        self,
        patched_get_session,
        session_factory,
        sample_vault: Path,
    ) -> None:
        """Obsidian ingest extracts frontmatter metadata."""
        config = IngestObsidianConfig(
            source_path=sample_vault,
            dataset_name="obsidian-vault",
        )

        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        assert result.documents_created == 4
        assert result.documents_failed == 0

        # Verify metadata was stored
        with create_session(session_factory) as session:
            query_result = session.execute(
                text("SELECT metadata_json FROM documents WHERE path LIKE '%note1.md'")
            )
            row = query_result.fetchone()
            assert row is not None
            # Metadata should contain frontmatter under "frontmatter" key
            import json
            metadata = json.loads(row[0]) if row[0] else {}
            frontmatter = metadata.get("frontmatter", {})
            assert "title" in frontmatter or "tags" in frontmatter

    def test_obsidian_excludes_obsidian_dir(
        self,
        patched_get_session,
        session_factory,
        sample_vault: Path,
    ) -> None:
        """Obsidian ingest excludes .obsidian directory."""
        # Add a file in .obsidian
        (sample_vault / ".obsidian" / "config.json").write_text('{"theme": "dark"}')

        config = IngestObsidianConfig(
            source_path=sample_vault,
            dataset_name="obsidian-vault",
        )

        pipeline = IngestPipeline()
        result = pipeline.ingest(config)

        # Should only have the 4 markdown files, not the config
        assert result.documents_created == 4

        # Verify config.json not in database
        with create_session(session_factory) as session:
            result = session.execute(
                text("SELECT COUNT(*) FROM documents WHERE path LIKE '%.json%'")
            )
            count = result.scalar()
            assert count == 0
