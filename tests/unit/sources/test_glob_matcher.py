"""Tests for MultiGlobMatcher."""

import pytest
from pathlib import Path

from pmd.sources.content.glob_matcher import (
    MultiGlobMatcher,
    parse_glob_patterns,
)


class TestMultiGlobMatcher:
    """Tests for MultiGlobMatcher."""

    def test_single_pattern_matches(self):
        """Single pattern matches correctly."""
        matcher = MultiGlobMatcher(["**/*.md"])

        assert matcher.matches("doc.md")
        assert matcher.matches("subdir/doc.md")
        assert matcher.matches("a/b/c/doc.md")
        assert not matcher.matches("doc.txt")
        assert not matcher.matches("readme.rst")

    def test_multiple_include_patterns_or_logic(self):
        """Multiple include patterns use OR logic."""
        matcher = MultiGlobMatcher(["**/*.md", "**/*.txt"])

        assert matcher.matches("doc.md")
        assert matcher.matches("doc.txt")
        assert matcher.matches("subdir/readme.md")
        assert matcher.matches("subdir/notes.txt")
        assert not matcher.matches("doc.rst")
        assert not matcher.matches("image.png")

    def test_exclude_pattern_filters_out_files(self):
        """Exclude patterns filter out matching files."""
        matcher = MultiGlobMatcher(["**/*.md", "!**/drafts/**"])

        assert matcher.matches("doc.md")
        assert matcher.matches("docs/readme.md")
        assert not matcher.matches("drafts/wip.md")
        assert not matcher.matches("docs/drafts/temp.md")

    def test_multiple_excludes(self):
        """Multiple exclude patterns all apply."""
        matcher = MultiGlobMatcher([
            "**/*.md",
            "!**/drafts/**",
            "!**/archive/**",
        ])

        assert matcher.matches("doc.md")
        assert not matcher.matches("drafts/wip.md")
        assert not matcher.matches("archive/old.md")
        assert not matcher.matches("docs/archive/legacy.md")

    def test_mixed_includes_and_excludes(self):
        """Mix of includes and excludes works correctly."""
        matcher = MultiGlobMatcher([
            "**/*.md",
            "**/*.txt",
            "!**/node_modules/**",
            "!**/test_*.md",
        ])

        assert matcher.matches("readme.md")
        assert matcher.matches("notes.txt")
        assert not matcher.matches("node_modules/pkg/readme.md")
        assert not matcher.matches("test_file.md")
        assert matcher.matches("file_test.md")  # Pattern is test_*, not *test*

    def test_exclude_specific_file(self):
        """Can exclude a specific file pattern."""
        matcher = MultiGlobMatcher(["**/*.md", "!README.md"])

        assert matcher.matches("docs/readme.md")  # Different case/path
        assert matcher.matches("guide.md")
        assert not matcher.matches("README.md")

    def test_no_include_patterns_raises(self):
        """Raises ValueError if no include patterns provided."""
        with pytest.raises(ValueError, match="At least one include pattern"):
            MultiGlobMatcher(["!**/exclude/**"])

    def test_empty_patterns_raises(self):
        """Raises ValueError for empty pattern list."""
        with pytest.raises(ValueError, match="At least one include pattern"):
            MultiGlobMatcher([])

    def test_path_normalization(self):
        """Handles different path separators."""
        matcher = MultiGlobMatcher(["**/*.md"])

        # Both forward and backslash should work
        assert matcher.matches("docs/readme.md")
        assert matcher.matches("docs\\readme.md")

    def test_list_matching_files(self, tmp_path: Path):
        """list_matching_files finds correct files."""
        # Create test structure
        (tmp_path / "doc.md").write_text("# Doc")
        (tmp_path / "notes.txt").write_text("Notes")
        (tmp_path / "drafts").mkdir()
        (tmp_path / "drafts" / "wip.md").write_text("# WIP")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "readme.md").write_text("# Readme")

        matcher = MultiGlobMatcher(["**/*.md", "!**/drafts/**"])

        files = list(matcher.list_matching_files(tmp_path))
        names = {f.name for f in files}

        assert "doc.md" in names
        assert "readme.md" in names
        assert "wip.md" not in names
        assert "notes.txt" not in names

    def test_list_matching_files_deduplicates(self, tmp_path: Path):
        """list_matching_files doesn't return duplicates."""
        (tmp_path / "readme.md").write_text("# Readme")

        # Both patterns match the same file
        matcher = MultiGlobMatcher(["**/*.md", "*.md"])

        files = list(matcher.list_matching_files(tmp_path))
        assert len(files) == 1

    def test_list_matching_files_multiple_includes(self, tmp_path: Path):
        """list_matching_files finds files from all include patterns."""
        (tmp_path / "doc.md").write_text("# Doc")
        (tmp_path / "notes.txt").write_text("Notes")
        (tmp_path / "data.json").write_text("{}")

        matcher = MultiGlobMatcher(["**/*.md", "**/*.txt"])

        files = list(matcher.list_matching_files(tmp_path))
        names = {f.name for f in files}

        assert "doc.md" in names
        assert "notes.txt" in names
        assert "data.json" not in names


class TestParseGlobPatterns:
    """Tests for parse_glob_patterns helper."""

    def test_none_returns_default(self):
        """None returns default pattern."""
        assert parse_glob_patterns(None) == ["**/*.md"]

    def test_string_returns_list(self):
        """Single string is wrapped in list."""
        assert parse_glob_patterns("**/*.txt") == ["**/*.txt"]

    def test_empty_list_returns_default(self):
        """Empty list returns default pattern."""
        assert parse_glob_patterns([]) == ["**/*.md"]

    def test_list_passed_through(self):
        """List is returned as-is."""
        patterns = ["**/*.md", "**/*.txt", "!**/drafts/**"]
        assert parse_glob_patterns(patterns) == patterns

    def test_list_is_copied(self):
        """Returns a copy of the input list."""
        original = ["**/*.md"]
        result = parse_glob_patterns(original)
        result.append("**/*.txt")
        assert original == ["**/*.md"]  # Original unchanged
