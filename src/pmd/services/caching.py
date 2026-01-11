"""Document caching service using fsspec.

This module provides caching functionality to store ingested documents
locally, converting remote or ephemeral content into concrete local files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import fsspec
from loguru import logger

if TYPE_CHECKING:
    from pmd.core.config import CacheConfig


class DocumentCacher:
    """Service for caching document content to a local filesystem.

    Caches ingested documents by collection, providing uniform file:// URIs
    for all content regardless of original source.

    Example:
        cacher = DocumentCacher(cache_config)
        uri = cacher.cache_document("my-collection", "path/to/doc.md", content)
        # Returns: file:///home/user/.cache/pmd/files/my-collection/path/to/doc.md
    """

    def __init__(self, config: "CacheConfig"):
        """Initialize DocumentCacher.

        Args:
            config: Cache configuration with base_path and enabled flag.
        """
        self._config = config
        self._base_path = Path(config.base_path).expanduser().resolve()
        self._fs = fsspec.filesystem("file")

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled."""
        return self._config.enabled

    @property
    def base_path(self) -> Path:
        """Base path for cached files."""
        return self._base_path

    def cache_document(
        self,
        collection_name: str,
        doc_path: str,
        content: str,
    ) -> str:
        """Cache document content and return the cached file URI.

        Args:
            collection_name: Name of the collection (used as subdirectory).
            doc_path: Document path within the collection.
            content: Document content to cache.

        Returns:
            file:// URI pointing to the cached file.

        Raises:
            OSError: If the file cannot be written.
        """
        # Build target path: <base>/<collection>/<doc_path>
        target_path = self._base_path / collection_name / doc_path

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content using fsspec
        with self._fs.open(str(target_path), "w", encoding="utf-8") as f:
            f.write(content)

        # Build file:// URI with proper encoding
        uri = target_path.as_uri()

        logger.debug(
            f"Cached document: collection={collection_name!r}, "
            f"path={doc_path!r}, size={len(content)}"
        )

        return uri

    def remove_document(self, collection_name: str, doc_path: str) -> bool:
        """Remove a cached document.

        Args:
            collection_name: Name of the collection.
            doc_path: Document path within the collection.

        Returns:
            True if file was removed, False if it didn't exist.
        """
        target_path = self._base_path / collection_name / doc_path

        if not target_path.exists():
            return False

        try:
            self._fs.rm(str(target_path))
            logger.debug(
                f"Removed cached document: collection={collection_name!r}, "
                f"path={doc_path!r}"
            )

            # Clean up empty parent directories
            self._cleanup_empty_dirs(target_path.parent, collection_name)

            return True
        except Exception as e:
            logger.warning(f"Failed to remove cached file {target_path}: {e}")
            return False

    def remove_collection(self, collection_name: str) -> int:
        """Remove all cached files for a collection.

        Args:
            collection_name: Name of the collection to remove.

        Returns:
            Number of files removed.
        """
        collection_path = self._base_path / collection_name

        if not collection_path.exists():
            return 0

        try:
            # Count files before removal
            count = sum(1 for _ in collection_path.rglob("*") if _.is_file())

            # Remove entire collection directory
            self._fs.rm(str(collection_path), recursive=True)

            logger.info(f"Removed cache for collection {collection_name!r}: {count} files")
            return count
        except Exception as e:
            logger.warning(f"Failed to remove collection cache {collection_path}: {e}")
            return 0

    def get_cached_path(self, collection_name: str, doc_path: str) -> Path | None:
        """Get the cached file path if it exists.

        Args:
            collection_name: Name of the collection.
            doc_path: Document path within the collection.

        Returns:
            Path to cached file if it exists, None otherwise.
        """
        target_path = self._base_path / collection_name / doc_path
        return target_path if target_path.exists() else None

    def _cleanup_empty_dirs(self, dir_path: Path, collection_name: str) -> None:
        """Remove empty directories up to the collection root.

        Args:
            dir_path: Directory to start cleanup from.
            collection_name: Collection name (stop cleanup at collection root).
        """
        collection_root = self._base_path / collection_name

        current = dir_path
        while current != collection_root and current.is_relative_to(collection_root):
            try:
                if current.is_dir() and not any(current.iterdir()):
                    current.rmdir()
                    current = current.parent
                else:
                    break
            except OSError:
                break
