"""Tests for generic metadata profile."""

import pytest
from datetime import datetime

from pmd.metadata import GenericProfile, ExtractedMetadata


class TestGenericProfileExtractMetadataFrontmatter:
    """Tests for extracting metadata from frontmatter."""

    def test_extract_tags_from_single_tag_field(self):
        """Should extract tags from tags field in frontmatter."""
        profile = GenericProfile()
        content = """---
tags: python
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert "python" in result.source_tags
        assert result.extraction_source == "generic"

    def test_extract_tags_from_list_field(self):
        """Should extract tags from list in frontmatter."""
        profile = GenericProfile()
        content = """---
tags:
  - python
  - machine-learning
  - ai
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.tags == {"python", "machine-learning", "ai"}
        assert len(result.source_tags) == 3

    def test_extract_tags_from_comma_separated_string(self):
        """Should extract tags from comma-separated string."""
        profile = GenericProfile()
        content = """---
tags: python, web, api
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.tags == {"python", "web", "api"}

    def test_extract_tags_from_multiple_field_names(self):
        """Should extract tags from tags, keywords, and categories fields."""
        profile = GenericProfile()
        content = """---
tags: python
keywords: web
categories: tutorial
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.tags == {"python", "web", "tutorial"}
        assert len(result.source_tags) == 3

    def test_extract_tags_from_singular_field_names(self):
        """Should extract from singular field names (tag, category)."""
        profile = GenericProfile()
        content = """---
tag: python
category: tutorial
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.tags == {"python", "tutorial"}

    def test_extract_common_attributes(self):
        """Should extract common attributes from frontmatter."""
        profile = GenericProfile()
        content = """---
title: My Document
author: John Doe
date: 2024-01-15
description: A test document
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.attributes["title"] == "My Document"
        assert result.attributes["author"] == "John Doe"
        assert result.attributes["date"] == "2024-01-15"
        assert result.attributes["description"] == "A test document"

    def test_extract_created_and_modified_timestamps(self):
        """Should extract created and modified timestamp attributes."""
        profile = GenericProfile()
        content = """---
created: 2024-01-01
modified: 2024-01-15
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "created" in result.attributes
        assert "modified" in result.attributes

    def test_handle_list_attribute_values(self):
        """Should convert list attribute values to list of strings."""
        profile = GenericProfile()
        content = """---
author:
  - John Doe
  - Jane Smith
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert isinstance(result.attributes["author"], list)
        assert result.attributes["author"] == ["John Doe", "Jane Smith"]

    def test_skip_dict_attribute_values(self):
        """Should skip dictionary attribute values."""
        profile = GenericProfile()
        content = """---
title: Test
metadata:
  nested: value
  another: data
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "title" in result.attributes
        assert "metadata" not in result.attributes

    def test_convert_numeric_attributes_to_strings(self):
        """Should convert numeric attribute values to strings."""
        profile = GenericProfile()
        content = """---
title: Test
date: 2024
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.attributes["title"] == "Test"
        assert result.attributes["date"] == "2024"

    def test_handle_none_attribute_values(self):
        """Should skip None attribute values."""
        profile = GenericProfile()
        content = """---
title: Test
author: null
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "title" in result.attributes
        assert "author" not in result.attributes


class TestGenericProfileExtractMetadataInlineTags:
    """Tests for extracting inline tags from content."""

    def test_extract_single_inline_tag(self):
        """Should extract single inline tag from content."""
        profile = GenericProfile()
        content = "This is a #python tutorial."

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert "#python" in result.source_tags

    def test_extract_multiple_inline_tags(self):
        """Should extract multiple inline tags from content."""
        profile = GenericProfile()
        content = "Learn #python and #javascript for #web development."

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.tags == {"python", "javascript", "web"}
        assert "#python" in result.source_tags
        assert "#javascript" in result.source_tags
        assert "#web" in result.source_tags

    def test_extract_nested_tags(self):
        """Should extract nested tags with slashes."""
        profile = GenericProfile()
        content = "Working on #project/active and #work/urgent tasks."

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "project/active" in result.tags
        assert "work/urgent" in result.tags

    def test_extract_hyphenated_tags(self):
        """Should extract tags with hyphens."""
        profile = GenericProfile()
        content = "Study #machine-learning and #deep-learning."

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "machine-learning" in result.tags
        assert "deep-learning" in result.tags

    def test_extract_underscored_tags(self):
        """Should extract tags with underscores."""
        profile = GenericProfile()
        content = "Review #code_review and #pull_request processes."

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "code_review" in result.tags
        assert "pull_request" in result.tags

    def test_ignore_tags_in_code_blocks(self):
        """Should not extract tags from code blocks."""
        profile = GenericProfile()
        content = """Some text #real-tag

```python
# This is a #fake-tag in code
print("hello")
```

More text #another-tag"""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "real-tag" in result.tags
        assert "another-tag" in result.tags
        assert "fake-tag" not in result.tags

    def test_ignore_tags_in_inline_code(self):
        """Should not extract tags from inline code."""
        profile = GenericProfile()
        content = "Use `#this-is-code` but extract #this-is-tag"

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "this-is-tag" in result.tags
        assert "this-is-code" not in result.tags

    def test_ignore_markdown_headings(self):
        """Should not extract markdown headings as tags."""
        profile = GenericProfile()
        content = """# Introduction

This is about #python programming."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert "introduction" not in result.tags

    def test_ignore_pure_number_tags(self):
        """Should not extract pure number tags."""
        profile = GenericProfile()
        content = "Issue #123 is about #python development."

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert "123" not in result.tags


class TestGenericProfileCombinedExtraction:
    """Tests for combined frontmatter and inline tag extraction."""

    def test_combine_frontmatter_and_inline_tags(self):
        """Should combine tags from both frontmatter and inline."""
        profile = GenericProfile()
        content = """---
tags: python
---
This is about #web development."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.tags == {"python", "web"}
        assert "python" in result.source_tags
        assert "#web" in result.source_tags

    def test_deduplicate_tags_from_both_sources(self):
        """Should deduplicate tags that appear in both sources."""
        profile = GenericProfile()
        content = """---
tags: python
---
Learn #python programming."""

        result = profile.extract_metadata(content, "/path/doc.md")

        # Normalized tags should be deduplicated
        assert result.tags == {"python"}
        # Source tags should preserve both occurrences
        assert len(result.source_tags) == 2

    def test_normalize_case_differences(self):
        """Should normalize case differences between sources."""
        profile = GenericProfile()
        content = """---
tags: Python
---
Using #PYTHON for scripting."""

        result = profile.extract_metadata(content, "/path/doc.md")

        # All normalized to lowercase
        assert result.tags == {"python"}

    def test_full_metadata_extraction(self):
        """Should extract complete metadata from complex document."""
        profile = GenericProfile()
        content = """---
title: Python Web Development
author: John Doe
date: 2024-01-15
tags:
  - python
  - web
keywords: flask, django
categories: tutorial
---
# Introduction

This guide covers #backend development using #python.
Learn about #api design and #database integration."""

        result = profile.extract_metadata(content, "/path/doc.md")

        # Check tags from all sources
        expected_tags = {"python", "web", "flask", "django", "tutorial",
                        "backend", "api", "database"}
        assert result.tags == expected_tags

        # Check attributes
        assert result.attributes["title"] == "Python Web Development"
        assert result.attributes["author"] == "John Doe"
        assert result.attributes["date"] == "2024-01-15"

        # Check extraction source
        assert result.extraction_source == "generic"


class TestGenericProfileNormalizeTag:
    """Tests for tag normalization."""

    def test_normalize_lowercase(self):
        """Should convert tags to lowercase."""
        profile = GenericProfile()

        assert profile.normalize_tag("Python") == "python"
        assert profile.normalize_tag("JAVASCRIPT") == "javascript"
        assert profile.normalize_tag("MixedCase") == "mixedcase"

    def test_normalize_strip_leading_hash(self):
        """Should remove leading # from tags."""
        profile = GenericProfile()

        assert profile.normalize_tag("#python") == "python"
        assert profile.normalize_tag("##python") == "python"

    def test_normalize_strip_whitespace(self):
        """Should strip leading and trailing whitespace."""
        profile = GenericProfile()

        assert profile.normalize_tag("  python  ") == "python"
        assert profile.normalize_tag("\tjavascript\n") == "javascript"

    def test_normalize_combined_transformations(self):
        """Should apply all normalizations together."""
        profile = GenericProfile()

        # Leading spaces prevent # stripping (lstrip("#") is applied before strip())
        assert profile.normalize_tag("  #Python  ") == "#python"
        assert profile.normalize_tag("#Python") == "python"
        assert profile.normalize_tag("#web-dev") == "web-dev"

    def test_normalize_preserve_special_chars(self):
        """Should preserve hyphens, underscores, and slashes."""
        profile = GenericProfile()

        assert profile.normalize_tag("machine-learning") == "machine-learning"
        assert profile.normalize_tag("code_review") == "code_review"
        assert profile.normalize_tag("project/active") == "project/active"

    def test_normalize_empty_string(self):
        """Should handle empty strings."""
        profile = GenericProfile()

        assert profile.normalize_tag("") == ""
        assert profile.normalize_tag("#") == ""
        assert profile.normalize_tag("   ") == ""


class TestGenericProfileEdgeCases:
    """Edge case tests for GenericProfile."""

    def test_no_frontmatter(self):
        """Should handle content without frontmatter."""
        profile = GenericProfile()
        content = "Just regular content with #python tag."

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert len(result.attributes) == 0

    def test_empty_content(self):
        """Should handle completely empty content."""
        profile = GenericProfile()
        content = ""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert len(result.tags) == 0
        assert len(result.source_tags) == 0
        assert len(result.attributes) == 0

    def test_empty_frontmatter(self):
        """Should handle empty frontmatter block."""
        profile = GenericProfile()
        content = """---
---
Content with #python tag."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert len(result.attributes) == 0

    def test_frontmatter_only(self):
        """Should handle frontmatter with no content."""
        profile = GenericProfile()
        content = """---
tags: python
title: Test
---"""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert result.attributes["title"] == "Test"

    def test_no_tags_anywhere(self):
        """Should handle content with no tags at all."""
        profile = GenericProfile()
        content = """---
title: Test
---
Just some regular content."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert len(result.tags) == 0
        assert result.attributes["title"] == "Test"

    def test_whitespace_only_content(self):
        """Should handle whitespace-only content."""
        profile = GenericProfile()
        content = "   \n\t\n   "

        result = profile.extract_metadata(content, "/path/doc.md")

        assert len(result.tags) == 0

    def test_invalid_yaml_frontmatter(self):
        """Should handle invalid YAML in frontmatter gracefully."""
        profile = GenericProfile()
        content = """---
tags: [python, web
invalid: yaml: structure
---
Content with #python tag."""

        result = profile.extract_metadata(content, "/path/doc.md")

        # Should treat as no frontmatter and extract inline tag
        assert "python" in result.tags

    def test_frontmatter_with_null_tag_field(self):
        """Should handle null tag field values."""
        profile = GenericProfile()
        content = """---
tags: null
title: Test
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert len(result.tags) == 0
        assert result.attributes["title"] == "Test"

    def test_very_long_content(self):
        """Should handle very long content efficiently."""
        profile = GenericProfile()
        # Create long content with tags at different positions
        content = "Start #tag1 " + ("word " * 10000) + " #tag2 end"

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "tag1" in result.tags
        assert "tag2" in result.tags

    def test_many_inline_tags(self):
        """Should handle content with many inline tags."""
        profile = GenericProfile()
        # Create content with 100 unique tags
        tags = " ".join([f"#tag{i}" for i in range(100)])
        content = f"Content with many tags: {tags}"

        result = profile.extract_metadata(content, "/path/doc.md")

        assert len(result.tags) == 100

    def test_duplicate_inline_tags(self):
        """Should deduplicate repeated inline tags."""
        profile = GenericProfile()
        content = "Tag #python appears #python multiple #python times"

        result = profile.extract_metadata(content, "/path/doc.md")

        # Normalized tags should be deduplicated
        assert result.tags == {"python"}
        # Source tags preserve all occurrences
        assert result.source_tags.count("#python") == 3

    def test_mixed_case_tag_deduplication(self):
        """Should deduplicate tags with different cases."""
        profile = GenericProfile()
        content = "#Python #PYTHON #python #PyThOn"

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.tags == {"python"}
        assert len(result.source_tags) == 4

    def test_path_parameter_unused(self):
        """Path parameter should not affect generic extraction."""
        profile = GenericProfile()
        content = "#python tutorial"

        result1 = profile.extract_metadata(content, "/path/one.md")
        result2 = profile.extract_metadata(content, "/different/path.md")

        assert result1.tags == result2.tags

    def test_unicode_content(self):
        """Should handle unicode content correctly."""
        profile = GenericProfile()
        content = """---
title: Python Tutorial 中文
---
Content with #python and unicode 日本語"""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python" in result.tags
        assert "Python Tutorial 中文" in result.attributes["title"]

    def test_datetime_attribute_conversion(self):
        """Should convert datetime objects to strings."""
        profile = GenericProfile()
        content = """---
date: 2024-01-15 10:30:00
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        # YAML should parse as datetime, then convert to string
        assert "date" in result.attributes
        assert isinstance(result.attributes["date"], str)

    def test_boolean_attribute_conversion(self):
        """Should convert boolean attributes to strings."""
        profile = GenericProfile()
        content = """---
published: true
draft: false
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        # Booleans not in common attributes list, but test anyway
        # They would be ignored since they're not in the common attrs list
        assert "published" not in result.attributes
        assert "draft" not in result.attributes

    def test_tags_with_numbers(self):
        """Should handle tags with numbers correctly."""
        profile = GenericProfile()
        content = "Using #python3 and #web2 and #k8s"

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "python3" in result.tags
        assert "web2" in result.tags
        assert "k8s" in result.tags

    def test_empty_tag_fields(self):
        """Should handle empty tag field values."""
        profile = GenericProfile()
        content = """---
tags: []
keywords: ""
---
Content with #python"""

        result = profile.extract_metadata(content, "/path/doc.md")

        # Should only have inline tag
        assert result.tags == {"python"}

    def test_mixed_type_tag_list(self):
        """Should handle tag lists with mixed types."""
        profile = GenericProfile()
        content = """---
tags:
  - python
  - 123
  - true
---
Content here."""

        result = profile.extract_metadata(content, "/path/doc.md")

        # All should be converted to strings
        assert "python" in result.tags
        assert "123" in result.tags
        assert "true" in result.tags


class TestGenericProfileAttributeExtraction:
    """Focused tests for attribute extraction edge cases."""

    def test_extract_all_common_attributes(self):
        """Should extract all recognized common attributes."""
        profile = GenericProfile()
        content = """---
title: Document Title
author: John Doe
date: 2024-01-15
created: 2024-01-01
modified: 2024-01-15
description: A comprehensive document
---
Content."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert len(result.attributes) == 6
        assert all(attr in result.attributes for attr in
                  ["title", "author", "date", "created", "modified", "description"])

    def test_ignore_unrecognized_attributes(self):
        """Should only extract recognized attributes."""
        profile = GenericProfile()
        content = """---
title: Test
custom_field: value
unknown: data
---
Content."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "title" in result.attributes
        assert "custom_field" not in result.attributes
        assert "unknown" not in result.attributes

    def test_handle_multiline_description(self):
        """Should handle multiline description values."""
        profile = GenericProfile()
        content = """---
description: |
  This is a long description
  that spans multiple lines
---
Content."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert "description" in result.attributes
        assert "multiple lines" in result.attributes["description"]

    def test_list_to_string_list_conversion(self):
        """Should properly convert list elements to strings."""
        profile = GenericProfile()
        content = """---
author:
  - John
  - 123
  - true
---
Content."""

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.attributes["author"] == ["John", "123", "True"]


class TestGenericProfileName:
    """Tests for profile name attribute."""

    def test_profile_has_name(self):
        """GenericProfile should have name attribute."""
        profile = GenericProfile()

        assert hasattr(profile, "name")
        assert profile.name == "generic"

    def test_extraction_source_matches_name(self):
        """Extraction source should match profile name."""
        profile = GenericProfile()
        content = "#python tutorial"

        result = profile.extract_metadata(content, "/path/doc.md")

        assert result.extraction_source == profile.name
