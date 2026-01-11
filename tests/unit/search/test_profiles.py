"""Tests for metadata profile implementations."""

import pytest

from pmd.metadata import (
    ExtractedMetadata,
    GenericProfile,
    ObsidianProfile,
    DraftsProfile,
)


class TestGenericProfile:
    """Tests for the GenericProfile implementation."""

    def test_profile_name(self):
        """Profile should have correct name."""
        profile = GenericProfile()
        assert profile.name == "generic"

    def test_extract_frontmatter_tags_array(self):
        """Should extract tags from YAML array."""
        profile = GenericProfile()
        content = """---
tags: [python, web, api]
---
Content here.
"""
        result = profile.extract_metadata(content, "test.md")

        assert "python" in result.tags
        assert "web" in result.tags
        assert "api" in result.tags
        assert result.extraction_source == "generic"

    def test_extract_frontmatter_tags_string(self):
        """Should extract tags from comma-separated string."""
        profile = GenericProfile()
        content = """---
tags: python, web, api
---
Content here.
"""
        result = profile.extract_metadata(content, "test.md")

        assert "python" in result.tags
        assert "web" in result.tags
        assert "api" in result.tags

    def test_extract_from_multiple_fields(self):
        """Should check multiple tag field names."""
        profile = GenericProfile()
        content = """---
keywords: [keyword1, keyword2]
categories: [cat1]
---
Content here.
"""
        result = profile.extract_metadata(content, "test.md")

        assert "keyword1" in result.tags
        assert "keyword2" in result.tags
        assert "cat1" in result.tags

    def test_extract_inline_tags(self):
        """Should extract inline hashtags from content."""
        profile = GenericProfile()
        content = """---
title: Test
---
This document discusses #python and #web development.
"""
        result = profile.extract_metadata(content, "test.md")

        assert "python" in result.tags
        assert "web" in result.tags

    def test_extract_both_frontmatter_and_inline(self):
        """Should combine frontmatter and inline tags."""
        profile = GenericProfile()
        content = """---
tags: [backend]
---
Talking about #frontend and #backend today.
"""
        result = profile.extract_metadata(content, "test.md")

        assert "backend" in result.tags
        assert "frontend" in result.tags
        # backend appears twice (frontmatter and inline) but set deduplicates
        assert len([t for t in result.tags if t == "backend"]) == 1

    def test_extract_common_attributes(self):
        """Should extract common metadata attributes."""
        profile = GenericProfile()
        content = """---
title: My Document
author: John Doe
date: 2024-01-15
description: A test document
---
Content here.
"""
        result = profile.extract_metadata(content, "test.md")

        assert result.attributes.get("title") == "My Document"
        assert result.attributes.get("author") == "John Doe"
        assert result.attributes.get("date") == "2024-01-15"
        assert result.attributes.get("description") == "A test document"

    def test_normalize_tag_lowercase(self):
        """Should normalize tags to lowercase."""
        profile = GenericProfile()

        assert profile.normalize_tag("Python") == "python"
        assert profile.normalize_tag("RUST") == "rust"
        assert profile.normalize_tag("CamelCase") == "camelcase"

    def test_normalize_tag_strips_hash(self):
        """Should strip leading # from tags."""
        profile = GenericProfile()

        assert profile.normalize_tag("#python") == "python"
        assert profile.normalize_tag("##double") == "double"

    def test_normalize_tag_strips_whitespace(self):
        """Should strip whitespace from tags."""
        profile = GenericProfile()

        assert profile.normalize_tag("  python  ") == "python"
        assert profile.normalize_tag("\ttab\t") == "tab"

    def test_source_tags_preserved(self):
        """Should preserve original source tags."""
        profile = GenericProfile()
        content = """---
tags: [Python, WEB-api]
---
Using #CamelCase tags.
"""
        result = profile.extract_metadata(content, "test.md")

        # Source tags should include original format
        assert "Python" in result.source_tags
        assert "WEB-api" in result.source_tags
        assert "#CamelCase" in result.source_tags

    def test_no_frontmatter_no_tags(self):
        """Should handle documents without frontmatter."""
        profile = GenericProfile()
        content = """# Just a heading

Regular content without any tags.
"""
        result = profile.extract_metadata(content, "test.md")

        assert result.tags == set()
        assert result.source_tags == []

    def test_empty_document(self):
        """Should handle empty documents."""
        profile = GenericProfile()
        result = profile.extract_metadata("", "test.md")

        assert result.tags == set()
        assert result.attributes == {}


class TestObsidianProfile:
    """Tests for the ObsidianProfile implementation."""

    def test_profile_name(self):
        """Profile should have correct name."""
        profile = ObsidianProfile()
        assert profile.name == "obsidian"

    def test_extract_frontmatter_tags(self):
        """Should extract tags from Obsidian frontmatter."""
        profile = ObsidianProfile()
        content = """---
tags: [python, rust]
---
Content here.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        assert "python" in result.tags
        assert "rust" in result.tags
        assert result.extraction_source == "obsidian"

    def test_expand_nested_tags(self):
        """Should expand nested tags to include parents."""
        profile = ObsidianProfile()
        content = """---
tags: [project/active/sprint1]
---
Content here.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        # Should expand to all levels
        assert "project" in result.tags
        assert "project/active" in result.tags
        assert "project/active/sprint1" in result.tags

    def test_expand_multiple_nested_tags(self):
        """Should expand multiple nested tags independently."""
        profile = ObsidianProfile()
        content = """---
tags: [project/alpha, status/todo]
---
Content here.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        assert "project" in result.tags
        assert "project/alpha" in result.tags
        assert "status" in result.tags
        assert "status/todo" in result.tags

    def test_extract_inline_nested_tags(self):
        """Should expand inline nested tags."""
        profile = ObsidianProfile()
        content = """---
title: Test
---
This document has #project/beta inline tags.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        assert "project" in result.tags
        assert "project/beta" in result.tags

    def test_extract_aliases(self):
        """Should extract Obsidian aliases."""
        profile = ObsidianProfile()
        content = """---
aliases: [Test Doc, My Alias]
---
Content here.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        assert "aliases" in result.attributes
        assert "Test Doc" in result.attributes["aliases"]
        assert "My Alias" in result.attributes["aliases"]

    def test_extract_cssclass(self):
        """Should extract Obsidian cssclass."""
        profile = ObsidianProfile()
        content = """---
cssclass: custom-class
---
Content here.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        assert result.attributes.get("cssclass") == "custom-class"

    def test_extract_cssclass_list(self):
        """Should extract cssclass as list."""
        profile = ObsidianProfile()
        content = """---
cssclass: [class1, class2]
---
Content here.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        assert "class1" in result.attributes["cssclass"]
        assert "class2" in result.attributes["cssclass"]

    def test_simple_tag_not_expanded(self):
        """Simple tags without / should not be expanded."""
        profile = ObsidianProfile()
        content = """---
tags: [python]
---
Content here.
"""
        result = profile.extract_metadata(content, "vault/test.md")

        assert result.tags == {"python"}

    def test_normalize_preserves_hierarchy(self):
        """Normalization should preserve / hierarchy."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("Project/Active") == "project/active"
        assert profile.normalize_tag("#parent/child") == "parent/child"


class TestDraftsProfile:
    """Tests for the DraftsProfile implementation."""

    def test_profile_name(self):
        """Profile should have correct name."""
        profile = DraftsProfile()
        assert profile.name == "drafts"

    def test_extract_basic_tags(self):
        """Should extract basic tags."""
        profile = DraftsProfile()
        content = """---
tags: [python, web]
---
Content here.
"""
        result = profile.extract_metadata(content, "drafts/test.md")

        assert "python" in result.tags
        assert "web" in result.tags
        assert result.extraction_source == "drafts"

    def test_hyphen_not_expanded_by_default(self):
        """By default, hyphen hierarchy should NOT be expanded."""
        profile = DraftsProfile(expand_hyphen_hierarchy=False)
        content = """---
tags: [project-active]
---
Content here.
"""
        result = profile.extract_metadata(content, "drafts/test.md")

        # Should keep as single tag
        assert "project-active" in result.tags
        assert "project" not in result.tags

    def test_hyphen_expanded_when_enabled(self):
        """Should expand hyphen hierarchy when enabled."""
        profile = DraftsProfile(expand_hyphen_hierarchy=True)
        content = """---
tags: [project-active-sprint1]
---
Content here.
"""
        result = profile.extract_metadata(content, "drafts/test.md")

        # Should expand to all levels
        assert "project" in result.tags
        assert "project-active" in result.tags
        assert "project-active-sprint1" in result.tags

    def test_extract_drafts_metadata(self):
        """Should extract Drafts-specific metadata."""
        profile = DraftsProfile()
        content = """---
uuid: 12345678-1234-1234-1234-123456789abc
flagged: true
created_at: 2024-01-15T10:30:00Z
modified_at: 2024-01-16T11:00:00Z
---
Content here.
"""
        result = profile.extract_metadata(content, "drafts/test.md")

        assert "uuid" in result.attributes
        assert "flagged" in result.attributes
        assert "created_at" in result.attributes
        assert "modified_at" in result.attributes

    def test_extract_geolocation(self):
        """Should extract Drafts geolocation fields."""
        profile = DraftsProfile()
        content = """---
created_latitude: 37.7749
created_longitude: -122.4194
---
Content here.
"""
        result = profile.extract_metadata(content, "drafts/test.md")

        assert "created_latitude" in result.attributes
        assert "created_longitude" in result.attributes

    def test_extract_inline_tags(self):
        """Should extract inline tags from content."""
        profile = DraftsProfile()
        content = """---
title: Test
---
Note about #meeting and #todo items.
"""
        result = profile.extract_metadata(content, "drafts/test.md")

        assert "meeting" in result.tags
        assert "todo" in result.tags

    def test_inline_tags_with_hyphen_expansion(self):
        """Should expand inline tags when hyphen expansion enabled."""
        profile = DraftsProfile(expand_hyphen_hierarchy=True)
        content = """# Note
Working on #project-backend today.
"""
        result = profile.extract_metadata(content, "drafts/test.md")

        assert "project" in result.tags
        assert "project-backend" in result.tags

    def test_normalize_tag(self):
        """Should normalize tags correctly."""
        profile = DraftsProfile()

        assert profile.normalize_tag("Python") == "python"
        assert profile.normalize_tag("#Tag") == "tag"
        assert profile.normalize_tag("  spaced  ") == "spaced"


class TestExtractedMetadata:
    """Tests for the ExtractedMetadata dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        metadata = ExtractedMetadata()

        assert metadata.tags == set()
        assert metadata.source_tags == []
        assert metadata.attributes == {}
        assert metadata.extraction_source == ""

    def test_with_values(self):
        """Should accept all values."""
        metadata = ExtractedMetadata(
            tags={"python", "web"},
            source_tags=["Python", "#web"],
            attributes={"title": "Test"},
            extraction_source="generic",
        )

        assert metadata.tags == {"python", "web"}
        assert metadata.source_tags == ["Python", "#web"]
        assert metadata.attributes == {"title": "Test"}
        assert metadata.extraction_source == "generic"

    def test_tags_are_set(self):
        """Tags should be stored as a set (deduplicated)."""
        metadata = ExtractedMetadata(
            tags={"python", "python", "rust"},
        )

        assert len(metadata.tags) == 2
        assert "python" in metadata.tags
        assert "rust" in metadata.tags
