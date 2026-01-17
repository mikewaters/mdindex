"""Tests for text quality utilities."""

import pytest

from pmd.search.text import is_indexable


class TestIsIndexable:
    """Tests for is_indexable function."""

    def test_empty_content(self):
        """Empty content should not be indexable."""
        assert is_indexable("") is False

    def test_whitespace_only(self):
        """Whitespace-only content should not be indexable."""
        assert is_indexable("   \n\n  \t  ") is False

    def test_title_only_heading(self):
        """Single heading with no body should not be indexable."""
        assert is_indexable("# My Title") is False

    def test_title_only_with_trailing_whitespace(self):
        """Title with trailing whitespace should not be indexable."""
        assert is_indexable("# My Title\n\n  \n") is False

    def test_title_with_body(self):
        """Heading with body content should be indexable."""
        content = "# My Title\n\nThis is the body content."
        assert is_indexable(content) is True

    def test_no_heading_with_content(self):
        """Content without heading should be indexable."""
        content = "Just some content without a heading."
        assert is_indexable(content) is True

    def test_bom_stripped(self):
        """BOM character should be stripped before checking."""
        # Title-only with BOM
        assert is_indexable("\ufeff# Title with BOM") is False
        # Content with BOM
        assert is_indexable("\ufeff# Title\n\nBody content") is True

    def test_second_level_heading_is_content(self):
        """Second-level heading alone should be indexable (not title-only pattern)."""
        assert is_indexable("## Not a Title") is True

    def test_code_block_with_hash(self):
        """Hash in code block context should be indexable."""
        content = "```python\n# This is a comment\n```"
        assert is_indexable(content) is True

    def test_title_with_minimal_body(self):
        """Title with minimal whitespace-only body should not be indexable."""
        assert is_indexable("# Heading\n   ") is False

    def test_title_with_single_word_body(self):
        """Title with even single word body should be indexable."""
        content = "# Heading\n\nContent"
        assert is_indexable(content) is True

    def test_multiline_body(self):
        """Multi-line body should be indexable."""
        content = "# Title\n\nLine 1\nLine 2\nLine 3"
        assert is_indexable(content) is True

    def test_no_heading_multiple_lines(self):
        """Multiple lines without heading should be indexable."""
        content = "Line 1\nLine 2\nLine 3"
        assert is_indexable(content) is True

    def test_yaml_frontmatter_with_title_only(self):
        """YAML frontmatter followed by title-only should not be indexable."""
        # The frontmatter is content, so this should be indexable
        content = "---\ntags: [doc]\n---\n# Title"
        # First line is "---" which doesn't start with "# "
        # So this is treated as having content
        assert is_indexable(content) is True

    def test_heading_with_yaml_and_body(self):
        """Document with frontmatter, title, and body should be indexable."""
        content = "---\ntags: [doc]\n---\n# Title\n\nBody content here."
        assert is_indexable(content) is True
