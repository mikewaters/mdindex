"""Shared fixtures for pipeline scenario tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from idx.ingest.pipelines import IngestPipeline
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import create_fts_table

pytest_plugins = [
    "tests.idx.scenario.pipelines.steps.content_hash_steps",
    "tests.idx.scenario.pipelines.steps.directory_steps",
    "tests.idx.scenario.pipelines.steps.obsidian_steps",
]


@dataclass(slots=True)
class ScenarioContext:
    """Mutable scenario context for pytest-bdd steps.

    Attributes:
        session: SQLAlchemy session used by the pipeline for the scenario.
        pipeline: Ingest pipeline instance bound to the scenario session.
        directory: Directory path used for directory ingestion scenarios.
        obsidian_vault: Vault path used for Obsidian ingestion scenarios.
        last_result: Result returned from the most recent ingest call.
        errors: Captured exception objects from "attempt" steps.
    """

    session: Session | None = None
    pipeline: IngestPipeline | None = None
    directory: Path | None = None
    obsidian_vault: Path | None = None
    last_result: Any | None = None
    errors: list[BaseException] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def ctx() -> ScenarioContext:
    """Provide an empty scenario context for step definitions."""

    return ScenarioContext()


@pytest.fixture
def db_session(tmp_path: Path) -> Session:
    """Create a test database session with the FTS table available."""

    db_path = tmp_path / "test.db"
    engine = create_engine_for_path(db_path)
    Base.metadata.create_all(engine)
    create_fts_table(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def pipeline(db_session: Session) -> IngestPipeline:
    """Create an ingest pipeline bound to the scenario DB session."""

    return IngestPipeline(session=db_session)


@pytest.fixture
def sample_directory(tmp_path: Path) -> Path:
    """Create a sample directory with markdown test files."""

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "readme.md").write_text("# Readme\n\nThis is a test.")
    (docs_dir / "notes.md").write_text("# Notes\n\nSome notes here.")

    subdir = docs_dir / "subdir"
    subdir.mkdir()
    (subdir / "deep.md").write_text("# Deep\n\nNested file.")

    return docs_dir


@pytest.fixture
def obsidian_vault(tmp_path: Path) -> Path:
    """Create a sample Obsidian vault with a mix of frontmatter and plain notes."""

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    obsidian_dir = vault_dir / ".obsidian"
    obsidian_dir.mkdir()
    (obsidian_dir / "app.json").write_text("{}")

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

    (vault_dir / "plain.md").write_text("# Plain Note\n\nNo frontmatter here.")

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


@pytest.fixture
def invalid_obsidian_vault(tmp_path: Path) -> Path:
    """Create a directory that is *not* a valid Obsidian vault."""

    not_a_vault = tmp_path / "not_vault"
    not_a_vault.mkdir()
    return not_a_vault
