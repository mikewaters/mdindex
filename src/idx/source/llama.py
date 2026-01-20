"""LlamaIndex readers for various source types.

Provides LlamaIndex-compatible readers that integrate with the
LlamaIndex ecosystem for document loading and indexing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from loguru import logger

from idx.source.obsidian import (
    ObsidianVaultSource,
    _extract_aliases,
    _extract_tags,
    _parse_frontmatter,
)


class ObsidianReader(BaseReader):
    """LlamaIndex reader for Obsidian vaults.

    Reads markdown files from an Obsidian vault, parsing YAML frontmatter
    and extracting metadata including tags and aliases. Returns LlamaIndex
    Document objects suitable for indexing.

    The reader uses relative file paths as document IDs for deterministic
    identification, making it suitable for incremental indexing workflows.

    Example:
        >>> reader = ObsidianReader("/path/to/vault")
        >>> documents = reader.load_data()
        >>> for doc in documents:
        ...     print(doc.doc_id, doc.metadata.get("tags"))

    Attributes:
        vault_path: Resolved absolute path to the Obsidian vault.
    """

    def __init__(self, vault_path: str | Path) -> None:
        """Initialize the Obsidian reader.

        Args:
            vault_path: Path to the Obsidian vault root directory.
                Must contain a .obsidian subdirectory.

        Raises:
            ValueError: If the path is not a valid Obsidian vault.
        """
        self.vault_path = Path(vault_path).resolve()
        # Validate vault by creating ObsidianVaultSource (will raise if invalid)
        self._source = ObsidianVaultSource(self.vault_path)
        logger.info(f"Initialized ObsidianReader for vault: {self.vault_path}")

    def load_data(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[Document]:
        """Load all documents from the Obsidian vault.

        Enumerates all markdown files in the vault, parses frontmatter,
        and returns LlamaIndex Document objects with extracted metadata.

        Args:
            *args: Ignored (for interface compatibility).
            **kwargs: Ignored (for interface compatibility).

        Returns:
            List of LlamaIndex Document objects, each containing:
                - text: Markdown content without frontmatter
                - doc_id: Relative file path (deterministic ID)
                - metadata: Dict with path, tags, aliases, frontmatter
        """
        documents: list[Document] = []

        for obsidian_doc in self._source.enumerate():
            doc = Document(
                text=obsidian_doc.body,
                doc_id=obsidian_doc.relative_path,
                metadata=self._build_metadata(obsidian_doc),
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from vault")
        return documents

    def _build_metadata(self, obsidian_doc: Any) -> dict[str, Any]:
        """Build metadata dictionary from an ObsidianDocument.

        Args:
            obsidian_doc: ObsidianDocument instance from the vault source.

        Returns:
            Metadata dictionary containing:
                - file_path: Absolute path to the file
                - file_name: Name of the file
                - relative_path: Path relative to vault root
                - tags: List of tags from frontmatter
                - aliases: List of aliases from frontmatter
                - frontmatter: Full parsed frontmatter dict (if present)
                - last_modified: File modification time (if available)
        """
        metadata: dict[str, Any] = {
            "file_path": str(obsidian_doc.path),
            "file_name": obsidian_doc.path.name,
            "relative_path": obsidian_doc.relative_path,
            "tags": obsidian_doc.tags,
            "aliases": obsidian_doc.aliases,
        }

        if obsidian_doc.frontmatter is not None:
            metadata["frontmatter"] = obsidian_doc.frontmatter

        if obsidian_doc.last_modified is not None:
            metadata["last_modified"] = obsidian_doc.last_modified.isoformat()

        return metadata


class ObsidianFileReader(BaseReader):
    """LlamaIndex reader for individual Obsidian markdown files.

    Reads a single markdown file with Obsidian-style frontmatter parsing.
    Can be used with SimpleDirectoryReader's file_extractor parameter
    for custom markdown handling.

    Example:
        >>> from llama_index.core import SimpleDirectoryReader
        >>> reader = SimpleDirectoryReader(
        ...     input_dir="./docs",
        ...     file_extractor={".md": ObsidianFileReader()}
        ... )
        >>> documents = reader.load_data()
    """

    def __init__(self, base_path: str | Path | None = None) -> None:
        """Initialize the file reader.

        Args:
            base_path: Optional base path for computing relative paths.
                If provided, document IDs will be relative to this path.
        """
        self._base_path = Path(base_path).resolve() if base_path else None

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Load a single markdown file with frontmatter parsing.

        Args:
            file: Path to the markdown file to load.
            extra_info: Optional additional metadata to merge.

        Returns:
            List containing a single Document with parsed content.
        """
        file_path = Path(file).resolve()

        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return []

        frontmatter, body = _parse_frontmatter(content)
        tags = _extract_tags(frontmatter)
        aliases = _extract_aliases(frontmatter)

        # Compute document ID from relative path if base_path is set
        if self._base_path:
            try:
                relative_path = str(file_path.relative_to(self._base_path))
                doc_id = relative_path
            except ValueError:
                doc_id = str(file_path)
        else:
            doc_id = str(file_path)

        metadata: dict[str, Any] = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "tags": tags,
            "aliases": aliases,
        }

        if frontmatter is not None:
            metadata["frontmatter"] = frontmatter

        # Merge any extra info provided by SimpleDirectoryReader
        if extra_info:
            metadata.update(extra_info)

        doc = Document(
            text=body,
            doc_id=doc_id,
            metadata=metadata,
        )

        return [doc]


__all__ = ["ObsidianFileReader", "ObsidianReader"]
