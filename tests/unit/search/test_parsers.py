"""Tests for metadata parsing utilities."""

import pytest

from pmd.sources.metadata import (
    FrontmatterResult,
    parse_frontmatter,
    extract_inline_tags,
    extract_tags_from_field,
)


class TestParseFrontmatter:
    """Tests for the parse_frontmatter function."""

    def test_valid_yaml_frontmatter(self):
        """Should parse valid YAML frontmatter."""
        content = """---
title: My Document
tags: [python, code]
author: Test User
---
# Hello World

Content here.
"""
        result = parse_frontmatter(content)

        assert result.has_frontmatter is True
        assert result.data["title"] == "My Document"
        assert result.data["tags"] == ["python", "code"]
        assert result.data["author"] == "Test User"
        assert "# Hello World" in result.content

    def test_no_frontmatter(self):
        """Should handle documents without frontmatter."""
        content = """# Hello World

No frontmatter here.
"""
        result = parse_frontmatter(content)

        assert result.has_frontmatter is False
        assert result.data == {}
        assert result.content == content

    def test_empty_frontmatter(self):
        """Should handle empty frontmatter block."""
        content = """---
---
# Content
"""
        result = parse_frontmatter(content)

        assert result.has_frontmatter is True
        assert result.data == {}
        assert "# Content" in result.content

    def test_invalid_yaml(self):
        """Should treat invalid YAML as no frontmatter."""
        content = """---
invalid: yaml: syntax: here
: broken
---
# Content
"""
        result = parse_frontmatter(content)

        # Invalid YAML is treated as no frontmatter
        assert result.has_frontmatter is False
        assert result.content == content

    def test_frontmatter_with_scalar_yaml(self):
        """Should handle YAML that parses to a scalar."""
        content = """---
just a string value
---
# Content
"""
        result = parse_frontmatter(content)

        # Scalar YAML is wrapped in {"_raw": value}
        assert result.has_frontmatter is True
        assert "_raw" in result.data

    def test_frontmatter_with_various_types(self):
        """Should preserve various YAML value types."""
        content = """---
string: hello
number: 42
float: 3.14
boolean: true
null_value: null
list:
  - item1
  - item2
nested:
  key: value
---
Content
"""
        result = parse_frontmatter(content)

        assert result.data["string"] == "hello"
        assert result.data["number"] == 42
        assert result.data["float"] == 3.14
        assert result.data["boolean"] is True
        assert result.data["null_value"] is None
        assert result.data["list"] == ["item1", "item2"]
        assert result.data["nested"]["key"] == "value"

    def test_frontmatter_preserves_remaining_content(self):
        """Should preserve all content after frontmatter."""
        content = """---
title: Test
---
First line
Second line
Third line
"""
        result = parse_frontmatter(content)

        assert "First line" in result.content
        assert "Second line" in result.content
        assert "Third line" in result.content

    def test_frontmatter_not_at_start(self):
        """Frontmatter must be at the start of document."""
        content = """Some text first

---
title: This is not frontmatter
---
More content
"""
        result = parse_frontmatter(content)

        assert result.has_frontmatter is False
        assert result.content == content

    def test_frontmatter_with_multiline_strings(self):
        """Should handle multiline YAML strings."""
        content = """---
description: |
  This is a
  multiline description
  with multiple lines.
---
Content
"""
        result = parse_frontmatter(content)

        assert result.has_frontmatter is True
        assert "multiline description" in result.data["description"]

    def test_frontmatter_with_special_characters(self):
        """Should handle special characters in values."""
        # Note: #python must be quoted in YAML to avoid being treated as a comment
        content = """---
title: "Title: With Colon"
author: Name <email@example.com>
tags: ["#python", "@mention"]
---
Content
"""
        result = parse_frontmatter(content)

        assert result.data["title"] == "Title: With Colon"
        assert "<email@example.com>" in result.data["author"]
        assert result.data["tags"] == ["#python", "@mention"]


class TestExtractInlineTags:
    """Tests for the extract_inline_tags function."""

    def test_simple_tags(self):
        """Should extract simple hashtags."""
        content = "Check out #python and #rust for programming."

        tags = extract_inline_tags(content)

        assert "python" in tags
        assert "rust" in tags

    def test_nested_tags(self):
        """Should extract nested/hierarchical tags."""
        content = "This is a #project/active task with #status/todo label."

        tags = extract_inline_tags(content)

        assert "project/active" in tags
        assert "status/todo" in tags

    def test_tags_with_dashes(self):
        """Should extract tags with dashes."""
        content = "Using #my-tag and #another-long-tag here."

        tags = extract_inline_tags(content)

        assert "my-tag" in tags
        assert "another-long-tag" in tags

    def test_tags_with_underscores(self):
        """Should extract tags with underscores."""
        content = "Found #my_tag and #another_tag here."

        tags = extract_inline_tags(content)

        assert "my_tag" in tags
        assert "another_tag" in tags

    def test_ignores_headings(self):
        """Should not match markdown headings."""
        content = """# Heading One
## Heading Two
### Heading Three

Regular #tag here.
"""
        tags = extract_inline_tags(content)

        assert "tag" in tags
        assert "Heading" not in tags
        assert len(tags) == 1

    def test_ignores_pure_numbers(self):
        """Should not match pure numeric tags."""
        content = "Issue #123 is fixed. Also #456. But #code123 is valid."

        tags = extract_inline_tags(content)

        # Pure numbers not matched, but alphanumeric are
        assert "code123" in tags
        assert "123" not in tags
        assert "456" not in tags

    def test_ignores_tags_in_fenced_code_blocks(self):
        """Should not extract tags from fenced code blocks."""
        content = """Some text with #valid tag.

```python
# This is a comment, not a tag
x = "#notag"
```

More text with #another tag.
"""
        tags = extract_inline_tags(content)

        assert "valid" in tags
        assert "another" in tags
        assert "notag" not in tags
        assert "This" not in tags

    def test_ignores_tags_in_inline_code(self):
        """Should not extract tags from inline code."""
        content = "Use `#notag` in code, but #realtag outside."

        tags = extract_inline_tags(content)

        assert "realtag" in tags
        assert "notag" not in tags

    def test_tag_at_start_of_line(self):
        """Should extract tags at start of line."""
        content = """#firsttag is here
And #secondtag here.
"""
        tags = extract_inline_tags(content)

        assert "firsttag" in tags
        assert "secondtag" in tags

    def test_tag_in_parentheses(self):
        """Should extract tags inside parentheses."""
        content = "Check this (#python) for details."

        tags = extract_inline_tags(content)

        assert "python" in tags

    def test_multiple_same_tags(self):
        """Should preserve duplicate tags (for frequency analysis)."""
        content = "#python is great. I love #python so much. #python forever!"

        tags = extract_inline_tags(content)

        assert tags.count("python") == 3

    def test_empty_content(self):
        """Should return empty list for empty content."""
        tags = extract_inline_tags("")

        assert tags == []

    def test_no_tags(self):
        """Should return empty list when no tags present."""
        content = "This content has no hashtags at all."

        tags = extract_inline_tags(content)

        assert tags == []

    def test_mixed_valid_invalid(self):
        """Should only extract valid tags."""
        content = "#valid # invalid #123 #also-valid #"

        tags = extract_inline_tags(content)

        assert "valid" in tags
        assert "also-valid" in tags
        # These should not be present
        assert "invalid" not in tags
        assert "123" not in tags


class TestExtractTagsFromField:
    """Tests for the extract_tags_from_field function."""

    def test_list_of_strings(self):
        """Should extract tags from a list of strings."""
        value = ["python", "rust", "javascript"]

        tags = extract_tags_from_field(value)

        assert tags == ["python", "rust", "javascript"]

    def test_comma_separated_string(self):
        """Should split comma-separated tags."""
        value = "python, rust, javascript"

        tags = extract_tags_from_field(value)

        assert tags == ["python", "rust", "javascript"]

    def test_single_string(self):
        """Should return single tag for single string."""
        value = "python"

        tags = extract_tags_from_field(value)

        assert tags == ["python"]

    def test_none_value(self):
        """Should return empty list for None."""
        tags = extract_tags_from_field(None)

        assert tags == []

    def test_empty_string(self):
        """Should return empty list for empty string."""
        tags = extract_tags_from_field("")

        assert tags == []

    def test_list_with_whitespace(self):
        """Should strip whitespace from list items."""
        value = ["  python  ", " rust ", "  javascript"]

        tags = extract_tags_from_field(value)

        assert tags == ["python", "rust", "javascript"]

    def test_comma_string_with_whitespace(self):
        """Should handle comma-separated with extra whitespace."""
        value = "  python  ,   rust  , javascript  "

        tags = extract_tags_from_field(value)

        assert tags == ["python", "rust", "javascript"]

    def test_list_with_none_items(self):
        """Should skip None items in list."""
        value = ["python", None, "rust", None]

        tags = extract_tags_from_field(value)

        assert tags == ["python", "rust"]

    def test_list_with_non_string_items(self):
        """Should convert non-string items to strings."""
        value = ["python", 123, True]

        tags = extract_tags_from_field(value)

        assert "python" in tags
        assert "123" in tags
        assert "True" in tags

    def test_hashtag_prefixed_string(self):
        """Should handle space-separated #tag format."""
        value = "#python #rust #javascript"

        tags = extract_tags_from_field(value)

        assert "python" in tags
        assert "rust" in tags
        assert "javascript" in tags

    def test_mixed_format_not_split(self):
        """Single string without comma or #-prefix pattern stays intact."""
        value = "multi word tag"

        tags = extract_tags_from_field(value)

        # Should be kept as single tag since no commas or #patterns
        assert tags == ["multi word tag"]

    def test_empty_list(self):
        """Should return empty list for empty list input."""
        tags = extract_tags_from_field([])

        assert tags == []

    def test_list_with_empty_strings(self):
        """Should filter out empty strings from list."""
        value = ["python", "", "rust", "  ", ""]

        tags = extract_tags_from_field(value)

        # Empty strings after strip should be filtered
        assert "python" in tags
        assert "rust" in tags
        assert "" not in tags


class TestFrontmatterResult:
    """Tests for the FrontmatterResult dataclass."""

    def test_dataclass_fields(self):
        """Should have correct fields."""
        result = FrontmatterResult(
            data={"key": "value"},
            content="Content here",
            has_frontmatter=True,
        )

        assert result.data == {"key": "value"}
        assert result.content == "Content here"
        assert result.has_frontmatter is True

    def test_empty_result(self):
        """Should allow empty data."""
        result = FrontmatterResult(
            data={},
            content="No frontmatter",
            has_frontmatter=False,
        )

        assert result.data == {}
        assert result.has_frontmatter is False
