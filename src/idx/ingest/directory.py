"""Local directory source reader.

Enumerates files from a directory, filtering by glob patterns,
and surfaces basic metadata for refresh semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Iterator, Any

from loguru import logger
from pydantic import BaseModel, Field
from llama_index.core import Document



class DirectorySource:
    """Source that enumerates files from a local directory.

    Scans a directory using glob patterns and yields SourceDocument
    instances for each matching file.

    Example:
        source = DirectorySource(
            Path("/path/to/docs"),
            patterns=["**/*.md", "**/*.txt"]
        )
        for doc in source.enumerate():
            print(f"{doc.relative_path}: {len(doc.content)} chars")
    """
    type_name = 'directory'

    def __init__(
        self,
        path: Path,
        patterns: list[str] | None = None,
        *,
        encoding: str = "utf-8",
        follow_symlinks: bool = False,
    ) -> None:
        """Initialize directory source.

        Args:
            path: Root directory to scan for documents.
            patterns: Glob patterns for matching files.
                Use ! prefix for exclusion patterns.
                Defaults to ["**/*.md"].
            encoding: File encoding to use. Defaults to utf-8.
            follow_symlinks: Whether to follow symbolic links. Defaults to False.

        Raises:
            ValueError: If no include patterns are provided.
        """
        raise NotImplementedError("This class needs to be adjusted to wrap a SimpleDirectoryReader")
        self._directory = path.resolve()
        self._patterns = self._parse_patterns(patterns)
        self._encoding = encoding
        self._follow_symlinks = follow_symlinks

        # Separate include and exclude patterns
        self._includes = [p for p in self._patterns if not p.startswith("!")]
        self._excludes = [p[1:] for p in self._patterns if p.startswith("!")]

        if not self._includes:
            raise ValueError(
                "At least one include pattern required (patterns without ! prefix)"
            )

        logger.debug(
            f"DirectorySource initialized: "
            f"directory={self._directory}, "
            f"includes={self._includes}, "
            f"excludes={self._excludes}"
        )

    @property
    def path(self) -> Path:
        """Get the resolved base directory."""
        return self._directory

    @property
    def patterns(self) -> list[str]:
        """Get all patterns (include and exclude)."""
        return self._patterns

    @cached_property
    def documents(self) -> list[Document]:
        """Load and return all documents from the directory as LlamaIndex Documents.

        This property enumerates all matching files and converts them to
        LlamaIndex Document instances suitable for ingestion pipelines.
        Results are cached for consistent document identity across accesses.

        Returns:
            List of LlamaIndex Document instances.
        """
        return [doc for doc in self.enumerate()]

    @staticmethod
    def validate(path: Path) -> None:
        """Validate that the given path is a valid directory."""
        if not path.exists():
            raise ValueError(f"Directory path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

    def enumerate(self) -> Iterator[Document]:
        """Enumerate all documents matching the glob patterns.

        Yields SourceDocument instances for each matching file,
        reading file content and gathering metadata.

        Yields:
            SourceDocument for each matching file.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
            PermissionError: If the directory cannot be read.
        """
        if not self._directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {self._directory}")

        if not self._directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self._directory}")

        logger.debug(f"Enumerating documents from: {self._directory}")

        seen: set[Path] = set()

        for pattern in self._includes:
            logger.debug(f"Globbing pattern: {pattern}")
            try:
                for file_path in self._directory.glob(pattern):
                    # Skip if not a file
                    if not file_path.is_file():
                        continue

                    # Skip symlinks if not following them
                    if file_path.is_symlink() and not self._follow_symlinks:
                        logger.debug(f"Skipping symlink: {file_path}")
                        continue

                    # Deduplicate (multiple patterns may match same file)
                    if file_path in seen:
                        continue

                    # Get relative path for exclusion checks
                    try:
                        rel_path = file_path.relative_to(self._directory)
                        rel_path_str = str(rel_path)
                    except ValueError:
                        # File is not under directory (shouldn't happen)
                        continue

                    # Check against exclusions
                    if self._matches_exclude(rel_path_str):
                        logger.debug(f"Excluded by pattern: {rel_path_str}")
                        continue

                    seen.add(file_path)

                    # Read file and gather metadata
                    doc = self._read_file(file_path, rel_path_str)
                    if doc is not None:
                        yield doc

            except PermissionError as e:
                logger.warning(f"Permission denied while globbing {pattern}: {e}")
            except OSError as e:
                logger.warning(f"Error globbing pattern {pattern}: {e}")

    def _read_file(self, file_path: Path, relative_path: str) -> Document | None:
        """Read a file and create a SourceDocument.

        Args:
            file_path: Absolute path to the file.
            relative_path: Path relative to source directory.

        Returns:
            SourceDocument or None if the file cannot be read.
        """
        # Get modification time
        last_modified: datetime | None = None
        try:
            stat = file_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        except OSError as e:
            logger.warning(f"Could not stat file {file_path}: {e}")

        # Read content
        try:
            content = file_path.read_text(encoding=self._encoding)
        except UnicodeDecodeError as e:
            logger.warning(
                f"Encoding error reading {file_path} with {self._encoding}: {e}"
            )
            return None
        except PermissionError:
            logger.warning(f"Permission denied reading {file_path}")
            return None
        except OSError as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None

        return Document(
            path=file_path,
            relative_path=relative_path,
            last_modified=last_modified,
            content=content,
            etag=None,
        )

    def _matches_exclude(self, path: str) -> bool:
        """Check if a relative path matches any exclusion pattern.

        Args:
            path: Relative file path to check.

        Returns:
            True if path matches any exclude pattern.
        """
        if not self._excludes:
            return False

        # Normalize path separators
        normalized = path.replace("\\", "/")
        p = Path(normalized)

        for pattern in self._excludes:
            if self._glob_match(p, pattern):
                return True

        return False

    @staticmethod
    def _glob_match(path: Path, pattern: str) -> bool:
        """Match path against a single glob pattern.

        Args:
            path: Path object to match.
            pattern: Glob pattern to match.

        Returns:
            True if path matches the pattern.
        """
        # Handle ** at the start specially
        # "**/*.md" should match both "doc.md" and "subdir/doc.md"
        if pattern.startswith("**/"):
            # Try without the **/ prefix for root-level matches
            suffix_pattern = pattern[3:]  # Remove "**/"

            # If suffix itself starts with **/, recurse
            if suffix_pattern.startswith("**/"):
                if DirectorySource._glob_match(path, suffix_pattern):
                    return True

            # Try matching the suffix pattern
            if path.match(suffix_pattern):
                return True

            # Also try the full pattern for nested files
            if path.match(pattern):
                return True

            # For patterns like **/node_modules/**, check if path contains the middle part
            if "/**" in pattern:
                middle = pattern[3:]  # Remove leading **/
                if middle.endswith("/**"):
                    dir_part = middle[:-3]  # Remove trailing /**
                    path_parts = str(path).replace("\\", "/").split("/")
                    if dir_part in path_parts:
                        return True

            return False

        return path.match(pattern)

    @staticmethod
    def _parse_patterns(patterns: list[str] | str | None) -> list[str]:
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



__all__ = ["DirectorySource"]
