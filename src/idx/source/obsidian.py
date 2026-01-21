"""Obsidian vault source reader.

Provides source reading capabilities for Obsidian vaults,
extracting markdown files with YAML frontmatter parsing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import Field
from llama_index.core import Document as LlamaDocument

from idx.source.directory import DirectorySource, SourceDocument


class ObsidianDocument(SourceDocument):
    """Document from an Obsidian vault with parsed frontmatter.

    Extends SourceDocument with Obsidian-specific metadata
    extracted from YAML frontmatter.
    """

    frontmatter: dict[str, Any] | None = None
    """Parsed YAML frontmatter dictionary, or None if not present."""

    tags: list[str] = Field(default_factory=list)
    """Tags extracted from frontmatter tags field."""

    aliases: list[str] = Field(default_factory=list)
    """Aliases extracted from frontmatter aliases field."""

    body: str = ""
    """Document content without frontmatter."""


# Regex to match YAML frontmatter block at start of document
# Matches: ---\n<yaml content>\n---\n
_FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n?---\s*\n?",
    re.DOTALL,
)


def _parse_frontmatter(content: str) -> tuple[dict[str, Any] | None, str]:
    """Parse YAML frontmatter from markdown content.

    Extracts the YAML block between --- delimiters at the start
    of the document, if present.

    Args:
        content: Full markdown document content.

    Returns:
        Tuple of (frontmatter_dict or None, remaining_content).
    """
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return None, content

    yaml_text = match.group(1)
    remaining_content = content[match.end() :]

    try:
        data = yaml.safe_load(yaml_text)
        if data is None:
            return {}, remaining_content
        if not isinstance(data, dict):
            # YAML could parse to a scalar or list - wrap it
            return {"_raw": data}, remaining_content
        return data, remaining_content
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        return None, content


def _extract_tags(frontmatter: dict[str, Any] | None) -> list[str]:
    """Extract tags from frontmatter tags field.

    Handles various YAML formats:
    - List: ["tag1", "tag2"]
    - String: "tag1, tag2" or "tag1 tag2"
    - Single value: "tag1"

    Args:
        frontmatter: Parsed frontmatter dictionary.

    Returns:
        List of extracted tag strings.
    """
    if frontmatter is None:
        return []

    tags_value = frontmatter.get("tags")
    if tags_value is None:
        return []

    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t is not None]

    if isinstance(tags_value, str):
        # Split by comma if multiple
        if "," in tags_value:
            return [t.strip() for t in tags_value.split(",") if t.strip()]
        return [tags_value.strip()] if tags_value.strip() else []

    # Fallback for other types
    return [str(tags_value).strip()] if tags_value else []


def _extract_aliases(frontmatter: dict[str, Any] | None) -> list[str]:
    """Extract aliases from frontmatter aliases field.

    Args:
        frontmatter: Parsed frontmatter dictionary.

    Returns:
        List of extracted alias strings.
    """
    if frontmatter is None:
        return []

    aliases_value = frontmatter.get("aliases")
    if aliases_value is None:
        return []

    if isinstance(aliases_value, list):
        return [str(a).strip() for a in aliases_value if a is not None]

    if isinstance(aliases_value, str):
        # Could be comma-separated
        if "," in aliases_value:
            return [a.strip() for a in aliases_value.split(",") if a.strip()]
        return [aliases_value.strip()] if aliases_value.strip() else []

    return [str(aliases_value).strip()] if aliases_value else []


class ObsidianVaultSource:
    """Source reader for Obsidian vaults.

    Enumerates markdown files from an Obsidian vault directory,
    extracting YAML frontmatter and yielding ObsidianDocument instances.

    Uses DirectorySource internally with glob pattern ["**/*.md"].

    Example:
        >>> source = ObsidianVaultSource("/path/to/vault")
        >>> for doc in source:
        ...     print(doc.path, doc.tags)
    """
    type_name = 'obsidian'

    @staticmethod
    def validate(path: Path) -> None:
        """Validate that the given path is a valid Obsidian vault."""
        if not path.exists():
            raise ValueError(f"Vault path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Vault path is not a directory: {path}")
        obsidian_dir = path / ".obsidian"
        if not obsidian_dir.is_dir():
            raise ValueError(
                f"Not a valid Obsidian vault (missing .obsidian directory): {path}"
            )

    def __init__(self, path: str | Path) -> None:
        """Initialize Obsidian vault source.

        Args:
            path: Path to the Obsidian vault root directory.
                Must contain a .obsidian subdirectory.

        Raises:
            ValueError: If the path is not a valid Obsidian vault.
        """
        self.path = Path(path).resolve()

        self.validate(self.path)

        # Create DirectorySource for markdown files, excluding .obsidian directory
        self._directory_source = DirectorySource(
            path=self.path,
            patterns=["**/*.md", "!.obsidian/**"],
        )

        logger.info(f"Initialized ObsidianVaultSource for vault: {self.path}")

    def __iter__(self) -> Iterator[ObsidianDocument]:
        """Iterate over all markdown documents in the vault.

        Yields:
            ObsidianDocument instances for each .md file found.
            Non-text documents are skipped with a log message.
        """
        return self.enumerate()

    def enumerate(self) -> Iterator[ObsidianDocument]:
        """Enumerate all markdown documents in the vault.

        Uses DirectorySource internally to find all .md files,
        then parses frontmatter and yields ObsidianDocument instances.

        Yields:
            ObsidianDocument instances for each .md file found.

        Note:
            Files in .obsidian directory are skipped.
            Non-text documents are logged and skipped.
        """
        for source_doc in self._directory_source.enumerate():
            try:
                obsidian_doc = self._convert_document(source_doc)
                yield obsidian_doc
            except Exception as e:
                logger.warning(f"Error processing document {source_doc.path}: {e}")
                continue

    def _convert_document(self, source_doc: SourceDocument) -> ObsidianDocument:
        """Convert a SourceDocument to an ObsidianDocument with parsed frontmatter.

        Args:
            source_doc: Source document from DirectorySource.

        Returns:
            ObsidianDocument with parsed frontmatter, tags, and aliases.
        """
        frontmatter, body = _parse_frontmatter(source_doc.content)
        tags = _extract_tags(frontmatter)
        aliases = _extract_aliases(frontmatter)

        return ObsidianDocument(
            path=source_doc.path,
            relative_path=source_doc.relative_path,
            last_modified=source_doc.last_modified,
            content=source_doc.content,
            etag=source_doc.etag,
            frontmatter=frontmatter,
            tags=tags,
            aliases=aliases,
            body=body,
        )

    #TODO: convert `to_llama_doc` to a classmethod
    @staticmethod
    def to_llama_doc(doc: ObsidianDocument) -> LlamaDocument:
        """Convert an ObsidianDocument to a LlamaIndex Document.

        Args:
            doc: The Obsidian document to convert.

        Returns:
            LlamaIndex Document with text and metadata.
        """
        metadata: dict[str, Any] = {
            "file_path": str(doc.path),
            "relative_path": doc.relative_path,
        }

        if doc.last_modified is not None:
            metadata["last_modified"] = doc.last_modified.isoformat()

        if doc.etag is not None:
            metadata["etag"] = doc.etag

        if doc.tags:
            metadata["tags"] = doc.tags

        if doc.aliases:
            metadata["aliases"] = doc.aliases

        if doc.frontmatter:
            metadata["frontmatter"] = doc.frontmatter

        # Use body (content without frontmatter) for text
        return LlamaDocument(
            text=doc.body,
            doc_id=doc.relative_path,
            metadata=metadata,
        )

__all__ = ["ObsidianDocument", "ObsidianVaultSource"]
