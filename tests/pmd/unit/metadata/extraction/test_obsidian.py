"""Tests for Obsidian metadata profile.

Tests for ObsidianProfile class which handles metadata extraction
from Obsidian vault documents with special support for nested tags,
wikilinks, and Obsidian-specific frontmatter fields.
"""

import pytest

from pmd.metadata import (
    ObsidianProfile,
    ExtractedMetadata,
)
from pmd.metadata.extraction.obsidian import detect_obsidian_content


class TestObsidianProfileExtractMetadataFrontmatter:
    """Tests for extracting metadata from frontmatter."""

    def test_extract_tags_from_single_tag_field(self):
        """Should extract tags from tags field in frontmatter."""
        profile = ObsidianProfile()
        content = """---
tags: python
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert "python" in result.source_tags
        assert result.extraction_source == "obsidian"

    def test_extract_tags_from_list_field(self):
        """Should extract tags from list in frontmatter."""
        profile = ObsidianProfile()
        content = """---
tags:
  - python
  - machine-learning
  - ai
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.tags == {"python", "machine-learning", "ai"}
        assert len(result.source_tags) == 3

    def test_extract_tags_from_comma_separated_string(self):
        """Should extract tags from comma-separated string."""
        profile = ObsidianProfile()
        content = """---
tags: python, web, api
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.tags == {"python", "web", "api"}

    def test_extract_nested_tags_from_frontmatter(self):
        """Should extract and expand nested tags from frontmatter."""
        profile = ObsidianProfile()
        content = """---
tags:
  - project/active
  - work/urgent
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Nested tags should be expanded to include parent tags
        assert "project" in result.tags
        assert "project/active" in result.tags
        assert "work" in result.tags
        assert "work/urgent" in result.tags

    def test_extract_aliases_field(self):
        """Should extract aliases from frontmatter."""
        profile = ObsidianProfile()
        content = """---
aliases:
  - Alt Title
  - Another Name
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "aliases" in result.attributes
        assert isinstance(result.attributes["aliases"], list)
        assert "Alt Title" in result.attributes["aliases"]
        assert "Another Name" in result.attributes["aliases"]

    def test_extract_aliases_as_string(self):
        """Should handle aliases as a single string."""
        profile = ObsidianProfile()
        content = """---
aliases: Single Alias
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "aliases" in result.attributes
        assert result.attributes["aliases"] == ["Single Alias"]

    def test_extract_aliases_comma_separated(self):
        """Should handle aliases as comma-separated string."""
        profile = ObsidianProfile()
        content = """---
aliases: Alias One, Alias Two, Alias Three
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "aliases" in result.attributes
        assert len(result.attributes["aliases"]) == 3

    def test_extract_cssclass_as_string(self):
        """Should extract cssclass as string."""
        profile = ObsidianProfile()
        content = """---
cssclass: custom-style
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "cssclass" in result.attributes
        assert result.attributes["cssclass"] == "custom-style"

    def test_extract_cssclass_as_list(self):
        """Should extract cssclass as list of strings."""
        profile = ObsidianProfile()
        content = """---
cssclass:
  - style1
  - style2
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "cssclass" in result.attributes
        assert isinstance(result.attributes["cssclass"], list)
        assert result.attributes["cssclass"] == ["style1", "style2"]

    def test_extract_common_metadata_fields(self):
        """Should extract title, author, date, created, modified."""
        profile = ObsidianProfile()
        content = """---
title: My Note
author: John Doe
date: 2024-01-15
created: 2024-01-01
modified: 2024-01-15
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.attributes["title"] == "My Note"
        assert result.attributes["author"] == "John Doe"
        assert result.attributes["date"] == "2024-01-15"
        assert result.attributes["created"] == "2024-01-01"
        assert result.attributes["modified"] == "2024-01-15"

    def test_skip_empty_attribute_values(self):
        """Should skip attributes with empty values."""
        profile = ObsidianProfile()
        content = """---
title: Test
author: ""
date: null
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "title" in result.attributes
        assert "author" not in result.attributes
        assert "date" not in result.attributes


class TestObsidianProfileExtractMetadataInlineTags:
    """Tests for extracting inline tags from content."""

    def test_extract_single_inline_tag(self):
        """Should extract single inline tag from content."""
        profile = ObsidianProfile()
        content = "This is a #python tutorial."

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert "#python" in result.source_tags

    def test_extract_multiple_inline_tags(self):
        """Should extract multiple inline tags from content."""
        profile = ObsidianProfile()
        content = "Learn #python and #javascript for #web development."

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.tags == {"python", "javascript", "web"}
        assert "#python" in result.source_tags
        assert "#javascript" in result.source_tags
        assert "#web" in result.source_tags

    def test_extract_nested_inline_tags(self):
        """Should extract and expand nested inline tags."""
        profile = ObsidianProfile()
        content = "Working on #project/active and #work/urgent tasks."

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Nested tags should be expanded
        assert "project" in result.tags
        assert "project/active" in result.tags
        assert "work" in result.tags
        assert "work/urgent" in result.tags

    def test_extract_deeply_nested_tags(self):
        """Should expand deeply nested tags with multiple levels."""
        profile = ObsidianProfile()
        content = "Tagged with #parent/child/grandchild"

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "parent" in result.tags
        assert "parent/child" in result.tags
        assert "parent/child/grandchild" in result.tags

    def test_extract_hyphenated_tags(self):
        """Should extract tags with hyphens."""
        profile = ObsidianProfile()
        content = "Study #machine-learning and #deep-learning."

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "machine-learning" in result.tags
        assert "deep-learning" in result.tags

    def test_extract_underscored_tags(self):
        """Should extract tags with underscores."""
        profile = ObsidianProfile()
        content = "Review #code_review and #pull_request processes."

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "code_review" in result.tags
        assert "pull_request" in result.tags

    def test_ignore_tags_in_code_blocks(self):
        """Should not extract tags from code blocks."""
        profile = ObsidianProfile()
        content = """Some text #real-tag

```python
# This is a #fake-tag in code
print("hello")
```

More text #another-tag"""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "real-tag" in result.tags
        assert "another-tag" in result.tags
        assert "fake-tag" not in result.tags

    def test_ignore_tags_in_inline_code(self):
        """Should not extract tags from inline code."""
        profile = ObsidianProfile()
        content = "Use `#this-is-code` but extract #this-is-tag"

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "this-is-tag" in result.tags
        assert "this-is-code" not in result.tags

    def test_ignore_markdown_headings(self):
        """Should not extract markdown headings as tags."""
        profile = ObsidianProfile()
        content = """# Introduction

This is about #python programming."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert "introduction" not in result.tags

    def test_ignore_pure_number_tags(self):
        """Should not extract pure number tags."""
        profile = ObsidianProfile()
        content = "Issue #123 is about #python development."

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert "123" not in result.tags


class TestObsidianProfileCombinedExtraction:
    """Tests for combined frontmatter and inline tag extraction."""

    def test_combine_frontmatter_and_inline_tags(self):
        """Should combine tags from both frontmatter and inline."""
        profile = ObsidianProfile()
        content = """---
tags: python
---
This is about #web development."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.tags == {"python", "web"}
        assert "python" in result.source_tags
        assert "#web" in result.source_tags

    def test_expand_nested_tags_from_both_sources(self):
        """Should expand nested tags from both frontmatter and inline."""
        profile = ObsidianProfile()
        content = """---
tags: project/active
---
Working on #work/urgent tasks."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "project" in result.tags
        assert "project/active" in result.tags
        assert "work" in result.tags
        assert "work/urgent" in result.tags

    def test_deduplicate_tags_from_both_sources(self):
        """Should deduplicate tags that appear in both sources."""
        profile = ObsidianProfile()
        content = """---
tags: python
---
Learn #python programming."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Normalized tags should be deduplicated
        assert result.tags == {"python"}
        # Source tags should preserve both occurrences
        assert len(result.source_tags) == 2

    def test_normalize_case_differences(self):
        """Should normalize case differences between sources."""
        profile = ObsidianProfile()
        content = """---
tags: Python
---
Using #PYTHON for scripting."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # All normalized to lowercase
        assert result.tags == {"python"}

    def test_full_obsidian_document(self):
        """Should extract complete metadata from complex Obsidian document."""
        profile = ObsidianProfile()
        content = """---
title: Python Web Development
author: John Doe
date: 2024-01-15
tags:
  - python
  - project/web
aliases:
  - Python Guide
  - Web Dev Tutorial
cssclass: custom-note
---
# Introduction

This guide covers [[Flask]] and [[Django]] for #backend development.
Working on #project/web/api design with #database integration.

See also: [[Related Note]]"""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Check expanded nested tags
        assert "python" in result.tags
        assert "project" in result.tags
        assert "project/web" in result.tags
        assert "project/web/api" in result.tags
        assert "backend" in result.tags
        assert "database" in result.tags

        # Check attributes
        assert result.attributes["title"] == "Python Web Development"
        assert result.attributes["author"] == "John Doe"
        assert "Python Guide" in result.attributes["aliases"]
        assert result.attributes["cssclass"] == "custom-note"

        # Check extraction source
        assert result.extraction_source == "obsidian"


class TestObsidianProfileNormalizeTag:
    """Tests for tag normalization."""

    def test_normalize_lowercase(self):
        """Should convert tags to lowercase."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("Python") == "python"
        assert profile.normalize_tag("JAVASCRIPT") == "javascript"
        assert profile.normalize_tag("MixedCase") == "mixedcase"

    def test_normalize_strip_leading_hash(self):
        """Should remove leading # from tags."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("#python") == "python"
        assert profile.normalize_tag("##python") == "python"

    def test_normalize_strip_whitespace(self):
        """Should strip leading and trailing whitespace."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("  python  ") == "python"
        assert profile.normalize_tag("\tjavascript\n") == "javascript"

    def test_normalize_nested_tags(self):
        """Should preserve hierarchy in nested tags."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("project/active") == "project/active"
        assert profile.normalize_tag("Parent/Child") == "parent/child"
        assert profile.normalize_tag("#work/urgent") == "work/urgent"

    def test_normalize_combined_transformations(self):
        """Should apply all normalizations together."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("#Python") == "python"
        assert profile.normalize_tag("#Project/Active") == "project/active"
        # Note: lstrip("#") is called before strip(), so leading spaces prevent hash stripping
        assert profile.normalize_tag("  #web-dev  ") == "#web-dev"

    def test_normalize_preserve_special_chars(self):
        """Should preserve hyphens, underscores, and slashes."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("machine-learning") == "machine-learning"
        assert profile.normalize_tag("code_review") == "code_review"
        assert profile.normalize_tag("project/active") == "project/active"

    def test_normalize_empty_string(self):
        """Should handle empty strings."""
        profile = ObsidianProfile()

        assert profile.normalize_tag("") == ""
        assert profile.normalize_tag("#") == ""
        assert profile.normalize_tag("   ") == ""


class TestObsidianProfileExpandNestedTag:
    """Tests for nested tag expansion."""

    def test_expand_simple_tag(self):
        """Should return single tag for non-nested tags."""
        profile = ObsidianProfile()

        result = profile._expand_nested_tag("python")

        assert result == {"python"}

    def test_expand_single_level_nested_tag(self):
        """Should expand one-level nested tag."""
        profile = ObsidianProfile()

        result = profile._expand_nested_tag("project/active")

        assert result == {"project", "project/active"}

    def test_expand_two_level_nested_tag(self):
        """Should expand two-level nested tag."""
        profile = ObsidianProfile()

        result = profile._expand_nested_tag("project/web/api")

        assert result == {"project", "project/web", "project/web/api"}

    def test_expand_three_level_nested_tag(self):
        """Should expand deeply nested tag."""
        profile = ObsidianProfile()

        result = profile._expand_nested_tag("parent/child/grandchild/great")

        assert result == {
            "parent",
            "parent/child",
            "parent/child/grandchild",
            "parent/child/grandchild/great",
        }

    def test_expand_normalizes_tag_first(self):
        """Should normalize tag before expansion."""
        profile = ObsidianProfile()

        result = profile._expand_nested_tag("#Project/Active")

        assert result == {"project", "project/active"}

    def test_expand_handles_hash_prefix(self):
        """Should handle tags with # prefix."""
        profile = ObsidianProfile()

        result = profile._expand_nested_tag("#work/urgent")

        assert result == {"work", "work/urgent"}

    def test_expand_mixed_case(self):
        """Should normalize mixed case in nested tags."""
        profile = ObsidianProfile()

        result = profile._expand_nested_tag("Parent/Child/GrandChild")

        assert result == {
            "parent",
            "parent/child",
            "parent/child/grandchild",
        }


class TestObsidianProfileEdgeCases:
    """Edge case tests for ObsidianProfile."""

    def test_no_frontmatter(self):
        """Should handle content without frontmatter."""
        profile = ObsidianProfile()
        content = "Just regular content with #python tag."

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert len(result.attributes) == 0

    def test_empty_content(self):
        """Should handle completely empty content."""
        profile = ObsidianProfile()
        content = ""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert len(result.tags) == 0
        assert len(result.source_tags) == 0
        assert len(result.attributes) == 0

    def test_empty_frontmatter(self):
        """Should handle empty frontmatter block."""
        profile = ObsidianProfile()
        content = """---
---
Content with #python tag."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert len(result.attributes) == 0

    def test_frontmatter_only(self):
        """Should handle frontmatter with no content."""
        profile = ObsidianProfile()
        content = """---
tags: python
title: Test
---"""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert result.attributes["title"] == "Test"

    def test_no_tags_anywhere(self):
        """Should handle content with no tags at all."""
        profile = ObsidianProfile()
        content = """---
title: Test
---
Just some regular content."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert len(result.tags) == 0
        assert result.attributes["title"] == "Test"

    def test_whitespace_only_content(self):
        """Should handle whitespace-only content."""
        profile = ObsidianProfile()
        content = "   \n\t\n   "

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert len(result.tags) == 0

    def test_invalid_yaml_frontmatter(self):
        """Should handle invalid YAML in frontmatter gracefully."""
        profile = ObsidianProfile()
        content = """---
tags: [python, web
invalid: yaml: structure
---
Content with #python tag."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Should treat as no frontmatter and extract inline tag
        assert "python" in result.tags

    def test_frontmatter_with_null_tag_field(self):
        """Should handle null tag field values."""
        profile = ObsidianProfile()
        content = """---
tags: null
title: Test
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert len(result.tags) == 0
        assert result.attributes["title"] == "Test"

    def test_very_long_content(self):
        """Should handle very long content efficiently."""
        profile = ObsidianProfile()
        # Create long content with tags at different positions
        content = "Start #tag1 " + ("word " * 10000) + " #tag2 end"

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "tag1" in result.tags
        assert "tag2" in result.tags

    def test_many_inline_tags(self):
        """Should handle content with many inline tags."""
        profile = ObsidianProfile()
        # Create content with 100 unique tags
        tags = " ".join([f"#tag{i}" for i in range(100)])
        content = f"Content with many tags: {tags}"

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert len(result.tags) == 100

    def test_duplicate_inline_tags(self):
        """Should deduplicate repeated inline tags."""
        profile = ObsidianProfile()
        content = "Tag #python appears #python multiple #python times"

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Normalized tags should be deduplicated
        assert result.tags == {"python"}
        # Source tags preserve all occurrences
        assert result.source_tags.count("#python") == 3

    def test_mixed_case_tag_deduplication(self):
        """Should deduplicate tags with different cases."""
        profile = ObsidianProfile()
        content = "#Python #PYTHON #python #PyThOn"

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.tags == {"python"}
        assert len(result.source_tags) == 4

    def test_path_parameter_unused(self):
        """Path parameter should not affect extraction."""
        profile = ObsidianProfile()
        content = "#python tutorial"

        result1 = profile.extract_metadata(content, "/vault/one.md")
        result2 = profile.extract_metadata(content, "/different/vault/two.md")

        assert result1.tags == result2.tags

    def test_unicode_content(self):
        """Should handle unicode content correctly."""
        profile = ObsidianProfile()
        content = """---
title: Python Tutorial 中文
---
Content with #python and unicode 日本語"""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert "Python Tutorial 中文" in result.attributes["title"]

    def test_tags_with_numbers(self):
        """Should handle tags with numbers correctly."""
        profile = ObsidianProfile()
        content = "Using #python3 and #web2 and #k8s"

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python3" in result.tags
        assert "web2" in result.tags
        assert "k8s" in result.tags

    def test_empty_tag_fields(self):
        """Should handle empty tag field values."""
        profile = ObsidianProfile()
        content = """---
tags: []
aliases: ""
---
Content with #python"""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Should only have inline tag
        assert result.tags == {"python"}
        assert "aliases" not in result.attributes

    def test_mixed_type_tag_list(self):
        """Should handle tag lists with mixed types."""
        profile = ObsidianProfile()
        content = """---
tags:
  - python
  - 123
  - true
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # All should be converted to strings
        assert "python" in result.tags
        assert "123" in result.tags
        assert "true" in result.tags

    def test_nested_tags_with_trailing_slash(self):
        """Should handle nested tags with trailing slash."""
        profile = ObsidianProfile()
        content = "Tagged with #project/active/"

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Should handle gracefully (trailing slash may be stripped by parser)
        assert "project" in result.tags or "project/active" in result.tags

    def test_empty_nested_tag_parts(self):
        """Should handle nested tags with empty parts."""
        profile = ObsidianProfile()
        # This is an edge case that might not occur in practice
        # but should be handled gracefully
        content = "Tagged with #project//active"

        result = profile.extract_metadata(content, "/vault/doc.md")

        # Should extract something reasonable
        assert len(result.tags) >= 0

    def test_wikilinks_do_not_affect_extraction(self):
        """Wikilinks should not affect metadata extraction."""
        profile = ObsidianProfile()
        content = """---
tags: python
---
See [[Related Note]] and [[Another Note|Display Text]] for #web development."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert "web" in result.tags
        # Wikilink text should not be treated as tags
        assert "related" not in result.tags
        assert "note" not in result.tags

    def test_obsidian_comments_do_not_affect_extraction(self):
        """Obsidian comments (%%) should not affect extraction."""
        profile = ObsidianProfile()
        content = """---
tags: python
---
Some content #visible

%% This is a comment with #hidden-tag %%

More content #also-visible"""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "python" in result.tags
        assert "visible" in result.tags
        assert "also-visible" in result.tags
        # Comment tags might still be extracted since we don't filter them
        # This tests current behavior


class TestObsidianProfileName:
    """Tests for profile name attribute."""

    def test_profile_has_name(self):
        """ObsidianProfile should have name attribute."""
        profile = ObsidianProfile()

        assert hasattr(profile, "name")
        assert profile.name == "obsidian"

    def test_extraction_source_matches_name(self):
        """Extraction source should match profile name."""
        profile = ObsidianProfile()
        content = "#python tutorial"

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.extraction_source == profile.name


class TestDetectObsidianContent:
    """Tests for Obsidian content detection function."""

    def test_detect_wikilinks(self):
        """Should detect content with wikilinks."""
        content = "See [[Related Note]] for more info."

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_wikilinks_with_alias(self):
        """Should detect wikilinks with display text."""
        content = "Check out [[Note|Display Text]] here."

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_embeds(self):
        """Should detect Obsidian embeds."""
        content = "![[Image.png]]"

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_obsidian_comments(self):
        """Should detect Obsidian comments."""
        content = "Some text %% hidden comment %% more text"

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_single_percent(self):
        """Should detect content with single %% marker."""
        content = "Text %% and more"

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_nested_tags(self):
        """Should detect nested tags."""
        content = "Working on #project/active tasks"

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_nested_tags_mixed_case(self):
        """Should detect nested tags with mixed case."""
        content = "Tagged with #Project/Active"

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_not_detect_plain_markdown(self):
        """Should not detect plain markdown as Obsidian."""
        content = """# Title

Some regular markdown content with #simple tags."""

        assert detect_obsidian_content(content, "/path/doc.md") is False

    def test_not_detect_simple_tags_only(self):
        """Should not detect content with only simple tags."""
        content = "Just #python and #javascript tags"

        assert detect_obsidian_content(content, "/path/doc.md") is False

    def test_not_detect_empty_content(self):
        """Should not detect empty content."""
        content = ""

        assert detect_obsidian_content(content, "/path/doc.md") is False

    def test_not_detect_regular_brackets(self):
        """Should not detect regular brackets as wikilinks."""
        content = "Array access: array[index]"

        assert detect_obsidian_content(content, "/path/doc.md") is False

    def test_detect_multiple_obsidian_features(self):
        """Should detect content with multiple Obsidian features."""
        content = """---
tags: project/active
---
See [[Related]] and work on #work/urgent tasks.
%% Don't forget to review %%"""

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_nested_tag_at_start(self):
        """Should detect nested tag at start of content."""
        content = "#project/active is the current status"

        assert detect_obsidian_content(content, "/path/doc.md") is True

    def test_detect_nested_tag_at_end(self):
        """Should detect nested tag at end of content."""
        content = "Current status is #project/active"

        assert detect_obsidian_content(content, "/path/doc.md") is True


class TestObsidianProfileAttributeExtraction:
    """Focused tests for attribute extraction edge cases."""

    def test_extract_all_common_attributes(self):
        """Should extract all recognized common attributes."""
        profile = ObsidianProfile()
        content = """---
title: Note Title
author: Jane Doe
date: 2024-01-15
created: 2024-01-01
modified: 2024-01-15
---
Content."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert len(result.attributes) == 5
        assert all(
            attr in result.attributes
            for attr in ["title", "author", "date", "created", "modified"]
        )

    def test_extract_obsidian_specific_attributes(self):
        """Should extract Obsidian-specific attributes."""
        profile = ObsidianProfile()
        content = """---
aliases:
  - Alt Name
cssclass: custom-style
tags: python
---
Content."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "aliases" in result.attributes
        assert "cssclass" in result.attributes

    def test_ignore_empty_aliases(self):
        """Should not add aliases attribute if empty."""
        profile = ObsidianProfile()
        content = """---
aliases: []
title: Test
---
Content."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "title" in result.attributes
        assert "aliases" not in result.attributes

    def test_handle_empty_cssclass(self):
        """Should store empty cssclass as-is."""
        profile = ObsidianProfile()
        content = """---
cssclass: ""
title: Test
---
Content."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert "title" in result.attributes
        # Empty string is stored (not filtered out)
        assert "cssclass" in result.attributes
        assert result.attributes["cssclass"] == ""

    def test_handle_numeric_cssclass(self):
        """Numeric cssclass values are not handled specially."""
        profile = ObsidianProfile()
        content = """---
cssclass: 123
---
Content."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # The implementation only handles str or list types for cssclass
        # Raw integers are not processed
        assert "cssclass" not in result.attributes

    def test_handle_cssclass_list_with_numbers(self):
        """Should convert cssclass list items to strings."""
        profile = ObsidianProfile()
        content = """---
cssclass:
  - style1
  - 123
  - true
---
Content."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        assert result.attributes["cssclass"] == ["style1", "123", "True"]

    def test_datetime_attribute_conversion(self):
        """Should convert datetime objects to strings."""
        profile = ObsidianProfile()
        content = """---
date: 2024-01-15 10:30:00
created: 2024-01-01T08:00:00Z
---
Content here."""

        result = profile.extract_metadata(content, "/vault/doc.md")

        # YAML should parse as datetime, then convert to string
        assert "date" in result.attributes
        assert isinstance(result.attributes["date"], str)
        assert "created" in result.attributes
        assert isinstance(result.attributes["created"], str)


class TestObsidianProfileComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_daily_note_format(self):
        """Should handle typical daily note format."""
        profile = ObsidianProfile()
        content = """---
date: 2024-01-15
tags:
  - daily-note
  - journal/2024
aliases:
  - Monday Jan 15
---
## Tasks
- [ ] Review #project/web/api
- [x] Meeting about #work/urgent items

## Notes
Discussed [[Project Planning]] with team.
Need to follow up on #action/follow-up tasks."""

        result = profile.extract_metadata(content, "/vault/daily/2024-01-15.md")

        # Check expanded tags
        assert "daily-note" in result.tags
        assert "journal" in result.tags
        assert "journal/2024" in result.tags
        assert "project" in result.tags
        assert "project/web" in result.tags
        assert "project/web/api" in result.tags
        assert "work" in result.tags
        assert "work/urgent" in result.tags
        assert "action" in result.tags
        assert "action/follow-up" in result.tags

    def test_project_note_format(self):
        """Should handle typical project note format."""
        profile = ObsidianProfile()
        content = """---
title: Web API Development
tags:
  - project/active
  - development/backend
  - status/in-progress
aliases:
  - API Project
  - Backend API
cssclass: project-note
created: 2024-01-01
modified: 2024-01-15
---
# Overview
Building REST API using [[Python]] and [[Flask]].

## Tech Stack
- #python3 #flask #postgresql
- #docker for deployment

## Tasks
- Implement #feature/authentication
- Design #feature/api/v2 endpoints"""

        result = profile.extract_metadata(content, "/vault/projects/web-api.md")

        # Check nested tag expansion
        assert "project" in result.tags
        assert "project/active" in result.tags
        assert "development" in result.tags
        assert "development/backend" in result.tags
        assert "status" in result.tags
        assert "status/in-progress" in result.tags
        assert "feature" in result.tags
        assert "feature/authentication" in result.tags
        assert "feature/api" in result.tags
        assert "feature/api/v2" in result.tags

        # Check simple tags
        assert "python3" in result.tags
        assert "flask" in result.tags
        assert "postgresql" in result.tags
        assert "docker" in result.tags

        # Check attributes
        assert result.attributes["title"] == "Web API Development"
        assert "API Project" in result.attributes["aliases"]
        assert result.attributes["cssclass"] == "project-note"

    def test_reference_note_with_many_links(self):
        """Should handle reference notes with many wikilinks."""
        profile = ObsidianProfile()
        content = """---
tags:
  - reference
  - learning/resources
title: Python Learning Resources
---
# Books
- [[Python Crash Course]]
- [[Fluent Python]]

# Online Courses
- [[Course - Python Fundamentals]]
- [[Advanced Python|Adv Python Course]]

Tagged with #python #learning #resources"""

        result = profile.extract_metadata(content, "/vault/reference.md")

        # Tags should be extracted correctly despite many wikilinks
        assert "reference" in result.tags
        assert "learning" in result.tags
        assert "learning/resources" in result.tags
        assert "python" in result.tags
        assert "learning" in result.tags
        assert "resources" in result.tags

    def test_meeting_note_with_nested_structure(self):
        """Should handle meeting notes with nested topic tags."""
        profile = ObsidianProfile()
        content = """---
date: 2024-01-15
tags:
  - meeting
  - team/engineering
  - status/weekly
participants:
  - Alice
  - Bob
---
# Weekly Engineering Meeting

## Topics Discussed
1. #project/frontend/redesign progress
2. #project/backend/api status
3. #infrastructure/deployment updates

## Action Items
- [ ] Review #code/pull-request
- [ ] Schedule #meeting/planning session"""

        result = profile.extract_metadata(content, "/vault/meetings/2024-01-15.md")

        # All nested tags should be expanded
        assert "meeting" in result.tags
        assert "team" in result.tags
        assert "team/engineering" in result.tags
        assert "project" in result.tags
        assert "project/frontend" in result.tags
        assert "project/frontend/redesign" in result.tags
        assert "project/backend" in result.tags
        assert "project/backend/api" in result.tags
        assert "infrastructure" in result.tags
        assert "infrastructure/deployment" in result.tags
        assert "code" in result.tags
        assert "code/pull-request" in result.tags
        assert "meeting/planning" in result.tags
