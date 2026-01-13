"""Multi-glob pattern matching utility.

This module provides a pattern matcher that supports multiple glob patterns
with include/exclude semantics, inspired by gitignore and ripgrep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from loguru import logger


class MultiGlobMatcher:
    """Match files against multiple glob patterns with include/exclude semantics.

    Supports multiple glob patterns where:
    - Patterns without prefix are "include" patterns (OR'd together)
    - Patterns with ! prefix are "exclude" patterns

    A file matches if it satisfies ANY include pattern AND does NOT match
    ANY exclude pattern.

    Example:
        matcher = MultiGlobMatcher(["**/*.md", "**/*.txt", "!**/drafts/**"])
        matcher.matches("docs/readme.md")  # True
        matcher.matches("drafts/wip.md")   # False (excluded)
    """

    def __init__(self, patterns: list[str]) -> None:
        """Initialize with pattern list.

        Args:
            patterns: List of glob patterns. Use ! prefix for exclusions.

        Raises:
            ValueError: If no include patterns are provided.
        """
        self.includes = [p for p in patterns if not p.startswith("!")]
        self.excludes = [p[1:] for p in patterns if p.startswith("!")]

        if not self.includes:
            raise ValueError(
                "At least one include pattern required (patterns without ! prefix)"
            )

        logger.debug(
            f"MultiGlobMatcher initialized: includes={self.includes}, excludes={self.excludes}"
        )

    def matches(self, path: str) -> bool:
        """Check if a relative path matches the pattern set.

        Args:
            path: Relative file path to check (forward slashes).

        Returns:
            True if path matches any include and no excludes.
        """
        # Normalize path separators
        normalized = path.replace("\\", "/")

        # Must match at least one include pattern
        if not any(self._glob_match(normalized, inc) for inc in self.includes):
            return False

        # Must not match any exclude pattern
        if any(self._glob_match(normalized, exc) for exc in self.excludes):
            return False

        return True

    def _glob_match(self, path: str, pattern: str) -> bool:
        """Match path against a single glob pattern.

        Uses pathlib.PurePath.match with special handling for ** patterns.

        Args:
            path: Normalized path string.
            pattern: Glob pattern to match.

        Returns:
            True if path matches the pattern.
        """
        p = Path(path)

        # Handle ** at the start specially
        # "**/*.md" should match both "doc.md" and "subdir/doc.md"
        if pattern.startswith("**/"):
            # Try without the **/ prefix for root-level matches
            suffix_pattern = pattern[3:]  # Remove "**/"

            # If suffix itself starts with **/, recurse
            if suffix_pattern.startswith("**/"):
                if self._glob_match(path, suffix_pattern):
                    return True

            # Try matching the suffix pattern
            if p.match(suffix_pattern):
                return True

            # Also try the full pattern for nested files
            if p.match(pattern):
                return True

            # For patterns like **/node_modules/**, check if path contains the middle part
            # e.g., "node_modules/pkg/file.md" should match "**/node_modules/**"
            if "/**" in pattern:
                # Pattern: **/X/** -> check if X is in path
                middle = pattern[3:]  # Remove leading **/
                if middle.endswith("/**"):
                    dir_part = middle[:-3]  # Remove trailing /**
                    # Check if this directory appears in the path
                    path_parts = path.replace("\\", "/").split("/")
                    if dir_part in path_parts:
                        return True

            return False

        return p.match(pattern)

    def list_matching_files(self, base_path: Path) -> Iterator[Path]:
        """List all files under base_path matching the pattern set.

        Iterates through each include pattern, deduplicates results,
        and filters out files matching exclude patterns.

        Args:
            base_path: Root directory to search.

        Yields:
            Path objects for matching files (deduplicated).
        """
        seen: set[Path] = set()

        for pattern in self.includes:
            logger.debug(f"Globbing pattern: {pattern}")
            try:
                for file_path in base_path.glob(pattern):
                    if not file_path.is_file():
                        continue

                    if file_path in seen:
                        continue

                    # Check against excludes
                    try:
                        rel_path = str(file_path.relative_to(base_path))
                    except ValueError:
                        # File is not under base_path (shouldn't happen)
                        continue

                    if any(self._glob_match(rel_path, exc) for exc in self.excludes):
                        logger.debug(f"Excluded by pattern: {rel_path}")
                        continue

                    seen.add(file_path)
                    yield file_path

            except OSError as e:
                logger.warning(f"Error globbing pattern {pattern}: {e}")


def parse_glob_patterns(patterns: list[str] | str | None) -> list[str]:
    """Normalize glob pattern input to a list.

    Args:
        patterns: Single pattern, list of patterns, or None.

    Returns:
        List of patterns, defaulting to ["**/*.md"] if None/empty.
    """
    if patterns is None:
        return ["**/*.md"]
    if isinstance(patterns, str):
        return [patterns]
    if not patterns:
        return ["**/*.md"]
    return list(patterns)
