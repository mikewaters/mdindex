"""Tests for text normalization utilities."""

import pytest

from pmd.search.text import NormalizedContent, normalize_content


class TestNormalizeContent:
    """Tests for normalize_content function."""

    def test_empty_content(self):
        """Empty content should be detected as title-only."""
        result = normalize_content("")

        assert result.embeddable_body == ""
        assert result.fts_body == ""
        assert result.title is None
        assert result.is_title_only is True

    def test_whitespace_only(self):
        """Whitespace-only content should be detected as title-only."""
        result = normalize_content("   \n\n  \t  ")

        assert result.embeddable_body == ""
        assert result.fts_body == ""
        assert result.title is None
        assert result.is_title_only is True

    def test_title_only_heading(self):
        """Single heading with no body should be title-only."""
        content = "# My Title"
        result = normalize_content(content)

        assert result.embeddable_body == ""
        assert result.fts_body == "My Title"
        assert result.title == "My Title"
        assert result.is_title_only is True

    def test_title_only_with_trailing_whitespace(self):
        """Title with trailing whitespace should be title-only."""
        content = "# My Title\n\n  \n"
        result = normalize_content(content)

        assert result.embeddable_body == ""
        assert result.fts_body == "My Title"
        assert result.title == "My Title"
        assert result.is_title_only is True

    def test_title_with_body(self):
        """Heading with body content should not be title-only."""
        content = "# My Title\n\nThis is the body content."
        result = normalize_content(content)

        assert result.embeddable_body == content
        assert result.fts_body == content
        assert result.title == "My Title"
        assert result.is_title_only is False

    def test_no_heading_with_content(self):
        """Content without heading should not be title-only."""
        content = "Just some content without a heading."
        result = normalize_content(content)

        assert result.embeddable_body == content
        assert result.fts_body == content
        assert result.title is None
        assert result.is_title_only is False

    def test_bom_stripped(self):
        """BOM character should be stripped."""
        content = "\ufeff# Title with BOM"
        result = normalize_content(content)

        assert result.embeddable_body == ""
        assert result.fts_body == "Title with BOM"
        assert result.title == "Title with BOM"
        assert result.is_title_only is True

    def test_second_level_heading_not_title(self):
        """Second-level heading should not be treated as title."""
        content = "## Not a Title"
        result = normalize_content(content)

        assert result.embeddable_body == content
        assert result.fts_body == content
        assert result.title is None
        assert result.is_title_only is False

    def test_code_block_with_hash(self):
        """Hash in non-heading context should not be treated as title."""
        content = "```python\n# This is a comment\n```"
        result = normalize_content(content)

        # First line doesn't start with "# " (it starts with ```)
        assert result.embeddable_body == content
        assert result.fts_body == content
        assert result.title is None
        assert result.is_title_only is False

    def test_title_only_with_minimal_body(self):
        """Title with minimal body (just whitespace) should be title-only."""
        content = "# Heading\n   "
        result = normalize_content(content)

        assert result.embeddable_body == ""
        assert result.fts_body == "Heading"
        assert result.title == "Heading"
        assert result.is_title_only is True

    def test_title_with_single_word_body(self):
        """Title with even single word body should not be title-only."""
        content = "# Heading\n\nContent"
        result = normalize_content(content)

        assert result.embeddable_body == content
        assert result.fts_body == content
        assert result.title == "Heading"
        assert result.is_title_only is False

    def test_multiline_body(self):
        """Multi-line body should preserve all content."""
        content = "# Title\n\nLine 1\nLine 2\nLine 3"
        result = normalize_content(content)

        assert result.embeddable_body == content
        assert result.fts_body == content
        assert result.title == "Title"
        assert result.is_title_only is False
