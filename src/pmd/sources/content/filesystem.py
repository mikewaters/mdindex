"""Filesystem document source.

This module provides a document source that reads files from the local
filesystem using glob patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

from loguru import logger

from .base import (
    BaseDocumentSource,
    DocumentReference,
    FetchResult,
    SourceCapabilities,
    SourceConfig,
    SourceFetchError,
    SourceListError,
)
from .glob_matcher import MultiGlobMatcher, parse_glob_patterns
from pmd.metadata import (
    ExtractedMetadata,
    MetadataProfileRegistry,
    get_default_profile_registry,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FileSystemConfig:
    """Configuration for filesystem source.

    Attributes:
        base_path: Root directory to scan for documents.
        glob_patterns: Patterns for matching files (default: ['**/*.md']).
            Use ! prefix for exclusion patterns.
        encoding: File encoding to use (default: 'utf-8').
        follow_symlinks: Whether to follow symbolic links (default: False).
        metadata_profile: Optional profile name to force (default: auto-detect).
    """

    base_path: Path
    glob_patterns: list[str] = field(default_factory=lambda: ["**/*.md"])
    encoding: str = "utf-8"
    follow_symlinks: bool = False
    metadata_profile: str | None = None

    @classmethod
    def from_source_config(cls, config: SourceConfig) -> "FileSystemConfig":
        """Create FileSystemConfig from generic SourceConfig.

        Args:
            config: Generic source configuration.

        Returns:
            FileSystemConfig instance.
        """
        # Parse the URI to get the base path
        parsed = urlparse(config.uri)
        if parsed.scheme == "file":
            # file:///path/to/dir -> /path/to/dir
            base_path = Path(unquote(parsed.path))
        else:
            # Bare path
            base_path = Path(config.uri)

        # Handle both glob_patterns (new) and glob_pattern (legacy)
        patterns = config.get("glob_patterns")
        if patterns is None:
            # Fall back to legacy single pattern
            patterns = config.get("glob_pattern", "**/*.md")
        patterns = parse_glob_patterns(patterns)

        return cls(
            base_path=base_path,
            glob_patterns=patterns,
            encoding=config.get("encoding", "utf-8"),
            follow_symlinks=config.get("follow_symlinks", False),
            metadata_profile=config.get("metadata_profile"),
        )


# =============================================================================
# Source Implementation
# =============================================================================


class FileSystemSource(BaseDocumentSource):
    """Document source for local filesystem.

    Scans a directory using glob patterns and yields documents for each
    matching file.

    Example:
        config = SourceConfig(
            uri="file:///path/to/docs",
            extra={"glob_pattern": "**/*.md"}
        )
        source = FileSystemSource(config)

        for ref in source.list_documents():
            result = await source.fetch_content(ref)
            print(f"{ref.path}: {len(result.content)} chars")
    """

    def __init__(self, config: SourceConfig) -> None:
        """Initialize filesystem source.

        Args:
            config: Source configuration with URI and options.
        """
        self._config = FileSystemConfig.from_source_config(config)
        self._base_path = self._config.base_path.resolve()
        self._metadata_registry: MetadataProfileRegistry = (
            get_default_profile_registry()
        )

    @property
    def base_path(self) -> Path:
        """Get the resolved base path."""
        return self._base_path

    @property
    def glob_pattern(self) -> str:
        """Get the primary glob pattern (for display)."""
        return self._config.glob_patterns[0] if self._config.glob_patterns else "**/*.md"

    @property
    def glob_patterns(self) -> list[str]:
        """Get all glob patterns."""
        return self._config.glob_patterns

    def list_documents(self) -> Iterator[DocumentReference]:
        """Enumerate all documents matching the glob patterns.

        Supports multiple include patterns (OR'd together) and exclude
        patterns (prefixed with !).

        Yields:
            DocumentReference for each matching file.

        Raises:
            SourceListError: If the base path doesn't exist or can't be read.
        """
        if not self._base_path.exists():
            raise SourceListError(
                str(self._base_path),
                f"Directory does not exist: {self._base_path}",
            )

        if not self._base_path.is_dir():
            raise SourceListError(
                str(self._base_path),
                f"Path is not a directory: {self._base_path}",
            )

        logger.debug(
            f"Listing documents: base={self._base_path}, patterns={self._config.glob_patterns}"
        )

        try:
            matcher = MultiGlobMatcher(self._config.glob_patterns)

            for file_path in matcher.list_matching_files(self._base_path):
                # Skip symlinks if not following them
                if file_path.is_symlink() and not self._config.follow_symlinks:
                    logger.debug(f"Skipping symlink: {file_path}")
                    continue

                relative_path = str(file_path.relative_to(self._base_path))
                uri = file_path.as_uri()

                # Get file metadata
                try:
                    stat = file_path.stat()
                    metadata = {
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                except OSError as e:
                    logger.warning(f"Could not stat file {file_path}: {e}")
                    metadata = {}

                yield DocumentReference(
                    uri=uri,
                    path=relative_path,
                    title=None,  # Will be extracted from content
                    metadata=metadata,
                )

        except ValueError as e:
            raise SourceListError(str(self._base_path), str(e))
        except PermissionError as e:
            raise SourceListError(str(self._base_path), f"Permission denied: {e}")
        except OSError as e:
            raise SourceListError(str(self._base_path), str(e))

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        """Read file content.

        Args:
            ref: Reference to the document to fetch.

        Returns:
            FetchResult with file content.

        Raises:
            SourceFetchError: If the file can't be read.
        """
        # Reconstruct the file path
        file_path = self._base_path / ref.path

        if not file_path.exists():
            raise SourceFetchError(ref.uri, "File not found", retryable=False)

        if not file_path.is_file():
            raise SourceFetchError(ref.uri, "Not a file", retryable=False)

        try:
            content = file_path.read_text(encoding=self._config.encoding)
        except UnicodeDecodeError as e:
            raise SourceFetchError(
                ref.uri,
                f"Encoding error ({self._config.encoding}): {e}",
                retryable=False,
            )
        except PermissionError:
            raise SourceFetchError(ref.uri, "Permission denied", retryable=False)
        except OSError as e:
            raise SourceFetchError(ref.uri, str(e), retryable=False)

        # Determine content type from extension
        content_type = self._guess_content_type(file_path)

        # Get fresh metadata
        try:
            stat = file_path.stat()
            metadata = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            metadata = {}

        extracted_metadata = self._extract_metadata(content, ref.path)

        return FetchResult(
            content=content,
            content_type=content_type,
            encoding=self._config.encoding,
            metadata=metadata,
            extracted_metadata=extracted_metadata,
        )

    def capabilities(self) -> SourceCapabilities:
        """Return filesystem source capabilities.

        The filesystem source supports efficient incremental updates
        using modification time comparison.
        """
        return SourceCapabilities(
            supports_incremental=True,
            supports_etag=False,
            supports_last_modified=True,  # Uses mtime
            supports_streaming=False,
            is_readonly=True,
            provides_document_metadata=True,
        )

    async def check_modified(
        self,
        ref: DocumentReference,
        stored_metadata: dict[str, Any],
    ) -> bool:
        """Check if file has been modified since last fetch.

        Uses modification time (mtime_ns for nanosecond precision) to
        detect changes without reading the file content.

        Args:
            ref: Reference to the document.
            stored_metadata: Metadata from previous fetch.

        Returns:
            True if the file may have changed, False if definitely unchanged.
        """
        file_path = self._base_path / ref.path

        if not file_path.exists():
            # File was deleted, consider it "modified" (needs removal)
            return True

        try:
            stat = file_path.stat()
        except OSError:
            # Can't stat, assume modified
            return True

        # Compare nanosecond-precision mtime if available
        stored_mtime_ns = stored_metadata.get("mtime_ns")
        if stored_mtime_ns is not None:
            return stat.st_mtime_ns != stored_mtime_ns

        # Fall back to second-precision mtime
        stored_mtime = stored_metadata.get("mtime")
        if stored_mtime is not None:
            return stat.st_mtime != stored_mtime

        # No stored mtime, assume modified
        return True

    @staticmethod
    def _guess_content_type(path: Path) -> str:
        """Guess MIME type from file extension.

        Args:
            path: File path.

        Returns:
            MIME type string.
        """
        suffix = path.suffix.lower()
        content_types = {
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".txt": "text/plain",
            ".rst": "text/x-rst",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
            ".xml": "application/xml",
            ".csv": "text/csv",
        }
        return content_types.get(suffix, "text/plain")

    def _extract_metadata(self, content: str, path: str) -> ExtractedMetadata | None:
        """Extract metadata for the given document content.

        Args:
            content: Document content.
            path: Relative path for profile detection.

        Returns:
            Extracted metadata or None if extraction fails.
        """
        try:
            if self._config.metadata_profile:
                profile = self._metadata_registry.get(self._config.metadata_profile)
                if not profile:
                    logger.warning(
                        "Configured metadata profile %r not found; falling back to auto-detection",
                        self._config.metadata_profile,
                    )
                    profile = self._metadata_registry.detect_or_default(content, path)
            else:
                profile = self._metadata_registry.detect_or_default(content, path)
            logger.debug(f"Using metadata profile: {profile.name}")
            return profile.extract_metadata(content, path)
        except Exception as exc:
            logger.warning(f"Failed to extract metadata for {path}: {exc}")
            return None
