"""Unit tests for the ObsidianVaultSource and ObsidianDocument."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from idx.source.obsidian import (
    ObsidianDocument,
    ObsidianVaultSource,
    _extract_aliases,
    _extract_tags,
    _parse_frontmatter,
)


class TestParseFrontmatter:
    """Tests for the _parse_frontmatter function."""

    def test_no_frontmatter(self) -> None:
        """Content without frontmatter returns None and original content."""
        content = "Just regular content here."
        frontmatter, body = _parse_frontmatter(content)

        assert frontmatter is None
        assert body == content

    def test_empty_frontmatter(self) -> None:
        """Empty frontmatter returns empty dict."""
        content = "---\n---\nBody content."
        frontmatter, body = _parse_frontmatter(content)

        assert frontmatter == {}
        assert body == "Body content."

    def test_simple_frontmatter(self) -> None:
        """Simple frontmatter is parsed correctly."""
        content = """---
title: My Note
tags: python
---
Body content."""
        frontmatter, body = _parse_frontmatter(content)

        assert frontmatter is not None
        assert frontmatter["title"] == "My Note"
        assert frontmatter["tags"] == "python"
        assert body == "Body content."

    def test_frontmatter_with_list(self) -> None:
        """Frontmatter with YAML list is parsed correctly."""
        content = """---
tags:
  - python
  - web
---
Body content."""
        frontmatter, body = _parse_frontmatter(content)

        assert frontmatter is not None
        assert frontmatter["tags"] == ["python", "web"]

    def test_invalid_yaml_returns_none(self) -> None:
        """Invalid YAML in frontmatter returns None."""
        content = """---
tags: [python, web
invalid: yaml
---
Body content."""
        frontmatter, body = _parse_frontmatter(content)

        assert frontmatter is None
        assert body == content

    def test_non_dict_frontmatter_wrapped(self) -> None:
        """Non-dict YAML (scalar/list) is wrapped in _raw key."""
        content = """---
- item1
- item2
---
Body content."""
        frontmatter, body = _parse_frontmatter(content)

        assert frontmatter is not None
        assert "_raw" in frontmatter
        assert frontmatter["_raw"] == ["item1", "item2"]

    def test_frontmatter_preserves_content_after_delimiters(self) -> None:
        """Body content after frontmatter delimiters is preserved."""
        content = "---\ntitle: Test\n---\nBody content here."
        frontmatter, body = _parse_frontmatter(content)

        assert frontmatter is not None
        assert body == "Body content here."


class TestExtractTags:
    """Tests for the _extract_tags function."""

    def test_none_frontmatter(self) -> None:
        """None frontmatter returns empty list."""
        assert _extract_tags(None) == []

    def test_no_tags_field(self) -> None:
        """Missing tags field returns empty list."""
        assert _extract_tags({"title": "Test"}) == []

    def test_tags_list(self) -> None:
        """Tags as YAML list are extracted."""
        fm: dict[str, Any] = {"tags": ["python", "web", "api"]}
        tags = _extract_tags(fm)

        assert tags == ["python", "web", "api"]

    def test_tags_single_string(self) -> None:
        """Single tag string is extracted."""
        fm: dict[str, Any] = {"tags": "python"}
        tags = _extract_tags(fm)

        assert tags == ["python"]

    def test_tags_comma_separated(self) -> None:
        """Comma-separated tags are split."""
        fm: dict[str, Any] = {"tags": "python, web, api"}
        tags = _extract_tags(fm)

        assert tags == ["python", "web", "api"]

    def test_tags_null_value(self) -> None:
        """Null tags value returns empty list."""
        fm: dict[str, Any] = {"tags": None}
        tags = _extract_tags(fm)

        assert tags == []

    def test_tags_list_with_none_items(self) -> None:
        """None items in tag list are filtered out."""
        fm: dict[str, Any] = {"tags": ["python", None, "web"]}
        tags = _extract_tags(fm)

        assert tags == ["python", "web"]

    def test_tags_list_with_mixed_types(self) -> None:
        """Mixed type items are converted to strings."""
        fm: dict[str, Any] = {"tags": ["python", 123, True]}
        tags = _extract_tags(fm)

        assert tags == ["python", "123", "True"]

    def test_tags_strips_whitespace(self) -> None:
        """Whitespace is stripped from tags."""
        fm: dict[str, Any] = {"tags": ["  python  ", " web "]}
        tags = _extract_tags(fm)

        assert tags == ["python", "web"]

    def test_tags_empty_string(self) -> None:
        """Empty string tag returns empty list."""
        fm: dict[str, Any] = {"tags": ""}
        tags = _extract_tags(fm)

        assert tags == []


class TestExtractAliases:
    """Tests for the _extract_aliases function."""

    def test_none_frontmatter(self) -> None:
        """None frontmatter returns empty list."""
        assert _extract_aliases(None) == []

    def test_no_aliases_field(self) -> None:
        """Missing aliases field returns empty list."""
        assert _extract_aliases({"title": "Test"}) == []

    def test_aliases_list(self) -> None:
        """Aliases as YAML list are extracted."""
        fm: dict[str, Any] = {"aliases": ["Alt Title", "Another Name"]}
        aliases = _extract_aliases(fm)

        assert aliases == ["Alt Title", "Another Name"]

    def test_aliases_single_string(self) -> None:
        """Single alias string is extracted."""
        fm: dict[str, Any] = {"aliases": "Alt Title"}
        aliases = _extract_aliases(fm)

        assert aliases == ["Alt Title"]

    def test_aliases_comma_separated(self) -> None:
        """Comma-separated aliases are split."""
        fm: dict[str, Any] = {"aliases": "Alt Title, Another Name"}
        aliases = _extract_aliases(fm)

        assert aliases == ["Alt Title", "Another Name"]

    def test_aliases_null_value(self) -> None:
        """Null aliases value returns empty list."""
        fm: dict[str, Any] = {"aliases": None}
        aliases = _extract_aliases(fm)

        assert aliases == []


class TestObsidianDocument:
    """Tests for the ObsidianDocument model."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Basic ObsidianDocument creation works."""
        doc = ObsidianDocument(
            path=tmp_path / "note.md",
            relative_path="note.md",
            content="# Test\nContent",
            body="Content",
        )

        assert doc.path == tmp_path / "note.md"
        assert doc.content == "# Test\nContent"
        assert doc.body == "Content"
        assert doc.frontmatter is None
        assert doc.tags == []
        assert doc.aliases == []

    def test_with_frontmatter(self, tmp_path: Path) -> None:
        """ObsidianDocument with frontmatter fields."""
        doc = ObsidianDocument(
            path=tmp_path / "note.md",
            relative_path="note.md",
            content="---\ntitle: Test\n---\nContent",
            frontmatter={"title": "Test", "tags": ["python"]},
            tags=["python"],
            aliases=["Alt Title"],
            body="Content",
        )

        assert doc.frontmatter == {"title": "Test", "tags": ["python"]}
        assert doc.tags == ["python"]
        assert doc.aliases == ["Alt Title"]

    def test_extends_source_document_fields(self, tmp_path: Path) -> None:
        """ObsidianDocument has all SourceDocument fields."""
        from datetime import datetime, timezone

        doc = ObsidianDocument(
            path=tmp_path / "note.md",
            relative_path="subdir/note.md",
            content="Content",
            body="Content",
            last_modified=datetime(2024, 1, 15, tzinfo=timezone.utc),
            etag="abc123",
        )

        assert doc.relative_path == "subdir/note.md"
        assert doc.last_modified is not None
        assert doc.etag == "abc123"


class TestObsidianVaultSource:
    """Tests for the ObsidianVaultSource class."""

    @pytest.fixture
    def vault_dir(self, tmp_path: Path) -> Path:
        """Create a minimal Obsidian vault directory."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / ".obsidian").mkdir()
        return vault

    def test_init_valid_vault(self, vault_dir: Path) -> None:
        """Initialize with valid Obsidian vault."""
        source = ObsidianVaultSource(vault_dir)

        assert source.path == vault_dir

    def test_init_not_directory(self, tmp_path: Path) -> None:
        """Raises ValueError if path is not a directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="not a directory"):
            ObsidianVaultSource(file_path)

    def test_init_missing_obsidian_dir(self, tmp_path: Path) -> None:
        """Raises ValueError if .obsidian directory is missing."""
        vault = tmp_path / "not_a_vault"
        vault.mkdir()

        with pytest.raises(ValueError, match="missing .obsidian directory"):
            ObsidianVaultSource(vault)

    def test_init_string_path(self, vault_dir: Path) -> None:
        """Initialize with string path works."""
        source = ObsidianVaultSource(str(vault_dir))

        assert source.path == vault_dir

    def test_enumerate_empty_vault(self, vault_dir: Path) -> None:
        """Empty vault yields no documents."""
        source = ObsidianVaultSource(vault_dir)

        docs = list(source.enumerate())

        assert docs == []

    def test_enumerate_single_file(self, vault_dir: Path) -> None:
        """Single markdown file is enumerated."""
        (vault_dir / "note.md").write_text("# Hello\nWorld")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].path == vault_dir / "note.md"
        assert docs[0].content == "# Hello\nWorld"
        assert docs[0].body == "# Hello\nWorld"

    def test_enumerate_multiple_files(self, vault_dir: Path) -> None:
        """Multiple markdown files are enumerated."""
        (vault_dir / "note1.md").write_text("Note 1")
        (vault_dir / "note2.md").write_text("Note 2")
        (vault_dir / "subdir").mkdir()
        (vault_dir / "subdir" / "note3.md").write_text("Note 3")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 3
        paths = {d.relative_path for d in docs}
        assert "note1.md" in paths
        assert "note2.md" in paths
        assert "subdir/note3.md" in paths

    def test_enumerate_skips_obsidian_dir(self, vault_dir: Path) -> None:
        """Files in .obsidian directory are skipped."""
        (vault_dir / "note.md").write_text("Note")
        (vault_dir / ".obsidian" / "config.md").write_text("Config")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].relative_path == "note.md"

    def test_enumerate_skips_non_markdown(self, vault_dir: Path) -> None:
        """Non-markdown files are skipped."""
        (vault_dir / "note.md").write_text("Note")
        (vault_dir / "image.png").write_bytes(b"\x89PNG")
        (vault_dir / "document.pdf").write_bytes(b"%PDF")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].relative_path == "note.md"

    def test_enumerate_parses_frontmatter(self, vault_dir: Path) -> None:
        """Frontmatter is parsed from markdown files."""
        content = """---
title: My Note
tags:
  - python
  - web
aliases:
  - Alt Title
---
# Heading
Body content."""
        (vault_dir / "note.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        doc = docs[0]
        assert doc.frontmatter is not None
        assert doc.frontmatter["title"] == "My Note"
        assert doc.tags == ["python", "web"]
        assert doc.aliases == ["Alt Title"]
        assert doc.body == "# Heading\nBody content."

    def test_enumerate_file_without_frontmatter(self, vault_dir: Path) -> None:
        """Files without frontmatter have None frontmatter."""
        (vault_dir / "note.md").write_text("# Just content\nNo frontmatter.")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        doc = docs[0]
        assert doc.frontmatter is None
        assert doc.tags == []
        assert doc.aliases == []
        assert doc.body == "# Just content\nNo frontmatter."

    def test_enumerate_invalid_frontmatter(self, vault_dir: Path) -> None:
        """Files with invalid YAML frontmatter are handled gracefully."""
        content = """---
tags: [python, web
invalid: yaml
---
Body content."""
        (vault_dir / "note.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        doc = docs[0]
        # Invalid YAML means frontmatter is None
        assert doc.frontmatter is None
        # Body is the full content since frontmatter parsing failed
        assert doc.body == content

    def test_iter_protocol(self, vault_dir: Path) -> None:
        """ObsidianVaultSource supports iteration protocol."""
        (vault_dir / "note.md").write_text("Content")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source)

        assert len(docs) == 1

    def test_enumerate_complex_vault(self, vault_dir: Path) -> None:
        """Complex vault structure is enumerated correctly."""
        # Create nested structure
        (vault_dir / "inbox").mkdir()
        (vault_dir / "projects").mkdir()
        (vault_dir / "projects" / "active").mkdir()
        (vault_dir / "archive").mkdir()

        # Create files
        (vault_dir / "index.md").write_text("Index")
        (vault_dir / "inbox" / "quick-note.md").write_text("Quick note")
        (vault_dir / "projects" / "README.md").write_text("Projects readme")
        (vault_dir / "projects" / "active" / "project-a.md").write_text("Project A")
        (vault_dir / "archive" / "old-note.md").write_text("Old note")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 5
        paths = {d.relative_path for d in docs}
        assert "index.md" in paths
        assert "inbox/quick-note.md" in paths
        assert "projects/README.md" in paths
        assert "projects/active/project-a.md" in paths
        assert "archive/old-note.md" in paths

    def test_enumerate_real_obsidian_note(self, vault_dir: Path) -> None:
        """Full Obsidian note with typical content is parsed correctly."""
        content = """---
title: Python Web Development
author: John Doe
date: 2024-01-15
tags:
  - python
  - web-development
  - project/active
aliases:
  - Python Guide
  - Web Dev Tutorial
---
# Introduction

This guide covers [[Flask]] and [[Django]] for #backend development.

## Getting Started

See also: [[Related Note]]

%% Private comment %%
"""
        (vault_dir / "python-web.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        doc = docs[0]

        # Check frontmatter extraction
        assert doc.frontmatter is not None
        assert doc.frontmatter["title"] == "Python Web Development"
        assert doc.frontmatter["author"] == "John Doe"

        # Check tags (from frontmatter, not inline)
        assert doc.tags == ["python", "web-development", "project/active"]

        # Check aliases
        assert doc.aliases == ["Python Guide", "Web Dev Tutorial"]

        # Check body doesn't include frontmatter
        assert "---" not in doc.body
        assert "# Introduction" in doc.body
        assert "[[Flask]]" in doc.body

    def test_enumerate_preserves_last_modified(self, vault_dir: Path) -> None:
        """last_modified from SourceDocument is preserved."""
        (vault_dir / "note.md").write_text("Content")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        # last_modified should be set from file stat
        assert docs[0].last_modified is not None


class TestObsidianVaultSourceEdgeCases:
    """Edge case tests for ObsidianVaultSource."""

    @pytest.fixture
    def vault_dir(self, tmp_path: Path) -> Path:
        """Create a minimal Obsidian vault directory."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / ".obsidian").mkdir()
        return vault

    def test_empty_frontmatter_block(self, vault_dir: Path) -> None:
        """Empty frontmatter block is handled."""
        content = """---
---
Body content."""
        (vault_dir / "note.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].frontmatter == {}
        assert docs[0].body == "Body content."

    def test_frontmatter_with_null_tags(self, vault_dir: Path) -> None:
        """Null tags in frontmatter return empty list."""
        content = """---
tags: null
---
Body content."""
        (vault_dir / "note.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].tags == []

    def test_frontmatter_with_empty_tags_list(self, vault_dir: Path) -> None:
        """Empty tags list in frontmatter."""
        content = """---
tags: []
---
Body content."""
        (vault_dir / "note.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].tags == []

    def test_unicode_content(self, vault_dir: Path) -> None:
        """Unicode content is handled correctly."""
        content = """---
title: 日本語タイトル
tags:
  - 中文标签
---
Content with unicode: 日本語 中文 한국어"""
        (vault_dir / "unicode.md").write_text(content, encoding="utf-8")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].frontmatter is not None
        assert docs[0].frontmatter["title"] == "日本語タイトル"
        assert docs[0].tags == ["中文标签"]
        assert "日本語" in docs[0].body

    def test_file_with_only_frontmatter(self, vault_dir: Path) -> None:
        """File with only frontmatter, no body."""
        content = """---
title: Just Frontmatter
tags: python
---"""
        (vault_dir / "note.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].tags == ["python"]
        assert docs[0].body == ""

    def test_deeply_nested_subdirectories(self, vault_dir: Path) -> None:
        """Deeply nested subdirectories are enumerated."""
        deep = vault_dir / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        (deep / "note.md").write_text("Deep note")

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].relative_path == "a/b/c/d/e/note.md"

    def test_tags_as_comma_separated_string(self, vault_dir: Path) -> None:
        """Tags as comma-separated string in frontmatter."""
        content = """---
tags: python, web, api
---
Content."""
        (vault_dir / "note.md").write_text(content)

        source = ObsidianVaultSource(vault_dir)
        docs = list(source.enumerate())

        assert len(docs) == 1
        assert docs[0].tags == ["python", "web", "api"]
