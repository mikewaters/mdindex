"""Concrete MetadataProfile implementations.

Provides app-specific metadata extraction profiles for different
document sources.
"""

import re

from .parsers import (
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)
from .profiles import ExtractedMetadata


class GenericProfile:
    """Fallback profile with minimal normalization.

    Extracts:
    - YAML frontmatter tags field
    - Inline #tags from content
    - Common frontmatter attributes (title, author, date, etc.)

    Normalization:
    - Lowercase tags
    - Strip whitespace
    - Remove leading # if present
    """

    name = "generic"

    def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
        """Extract metadata using generic markdown conventions.

        Args:
            content: Full document content.
            path: Document path (unused in generic profile).

        Returns:
            ExtractedMetadata with tags and common attributes.
        """
        result = parse_frontmatter(content)

        tags: set[str] = set()
        source_tags: list[str] = []
        attributes: dict[str, str | list[str]] = {}

        # Extract tags from frontmatter
        if result.has_frontmatter:
            # Look for tags in common field names
            for field in ("tags", "tag", "keywords", "categories", "category"):
                if field in result.data:
                    field_tags = extract_tags_from_field(result.data[field])
                    source_tags.extend(field_tags)
                    for tag in field_tags:
                        tags.add(self.normalize_tag(tag))

            # Extract common attributes
            for attr in ("title", "author", "date", "created", "modified", "description"):
                if attr in result.data and result.data[attr]:
                    value = result.data[attr]
                    if isinstance(value, (str, int, float)):
                        attributes[attr] = str(value)
                    elif isinstance(value, list):
                        attributes[attr] = [str(v) for v in value]

        # Extract inline tags from content (after frontmatter)
        inline_tags = extract_inline_tags(result.content)
        source_tags.extend([f"#{t}" for t in inline_tags])
        for tag in inline_tags:
            tags.add(self.normalize_tag(tag))

        return ExtractedMetadata(
            tags=tags,
            source_tags=source_tags,
            attributes=attributes,
            extraction_source=self.name,
        )

    def normalize_tag(self, tag: str) -> str:
        """Normalize a tag to lowercase form.

        Args:
            tag: Raw tag string.

        Returns:
            Lowercase, trimmed tag without leading #.
        """
        return tag.lstrip("#").lower().strip()


class ObsidianProfile:
    """Profile for Obsidian vault documents.

    Handles Obsidian-specific features:
    - YAML frontmatter with tags array
    - Inline #tags anywhere in content
    - Nested tags with / hierarchy (e.g., #project/active)
    - aliases and cssclass fields

    Normalization:
    - Nested tags expanded: #parent/child -> both "parent" and "parent/child"
    - Lowercase with hierarchy preserved
    """

    name = "obsidian"

    def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
        """Extract metadata from Obsidian markdown document.

        Args:
            content: Full document content.
            path: Document path (may be used for vault-relative lookups).

        Returns:
            ExtractedMetadata with Obsidian-style tags and attributes.
        """
        result = parse_frontmatter(content)

        tags: set[str] = set()
        source_tags: list[str] = []
        attributes: dict[str, str | list[str]] = {}

        # Extract from frontmatter
        if result.has_frontmatter:
            # Tags field (can be array or inline-style)
            if "tags" in result.data:
                field_tags = extract_tags_from_field(result.data["tags"])
                source_tags.extend(field_tags)
                for tag in field_tags:
                    tags.update(self._expand_nested_tag(tag))

            # Obsidian-specific fields
            if "aliases" in result.data:
                aliases = extract_tags_from_field(result.data["aliases"])
                if aliases:
                    attributes["aliases"] = aliases

            if "cssclass" in result.data:
                css = result.data["cssclass"]
                if isinstance(css, str):
                    attributes["cssclass"] = css
                elif isinstance(css, list):
                    attributes["cssclass"] = [str(c) for c in css]

            # Common metadata
            for attr in ("title", "author", "date", "created", "modified"):
                if attr in result.data and result.data[attr]:
                    attributes[attr] = str(result.data[attr])

        # Extract inline tags from content
        inline_tags = extract_inline_tags(result.content)
        source_tags.extend([f"#{t}" for t in inline_tags])
        for tag in inline_tags:
            tags.update(self._expand_nested_tag(tag))

        return ExtractedMetadata(
            tags=tags,
            source_tags=source_tags,
            attributes=attributes,
            extraction_source=self.name,
        )

    def normalize_tag(self, tag: str) -> str:
        """Normalize an Obsidian tag.

        Preserves hierarchy but lowercases.

        Args:
            tag: Raw tag string (may include /).

        Returns:
            Lowercase tag with hierarchy intact.
        """
        return tag.lstrip("#").lower().strip()

    def _expand_nested_tag(self, tag: str) -> set[str]:
        """Expand a nested tag to include parent tags.

        For example: "project/active" expands to {"project", "project/active"}

        Args:
            tag: Tag string, possibly with / hierarchy.

        Returns:
            Set of normalized tags including parent tags.
        """
        normalized = self.normalize_tag(tag)
        if "/" not in normalized:
            return {normalized}

        result = set()
        parts = normalized.split("/")
        current = ""
        for part in parts:
            if current:
                current = f"{current}/{part}"
            else:
                current = part
            result.add(current)
        return result


class DraftsProfile:
    """Profile for Drafts app exports.

    Handles Drafts-specific features:
    - Tags stored in accompanying JSON metadata files
    - Hyphen-based hierarchy convention (e.g., "project-active")
    - Draft metadata (uuid, created_at, modified_at, flagged)

    Normalization:
    - Hyphen hierarchy can optionally expand: "parent-child" -> both
    - Lowercase
    """

    name = "drafts"

    def __init__(self, expand_hyphen_hierarchy: bool = False):
        """Initialize DraftsProfile.

        Args:
            expand_hyphen_hierarchy: If True, expand "parent-child" to
                include "parent" as well (default: False).
        """
        self.expand_hyphen_hierarchy = expand_hyphen_hierarchy

    def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
        """Extract metadata from Drafts export.

        Note: Drafts stores metadata in a separate .json file alongside
        the .md file. This profile expects the content to be the markdown
        and looks for tags in frontmatter as a fallback.

        For full Drafts support, the indexing pipeline should load the
        companion JSON file and pass its tags via frontmatter injection.

        Args:
            content: Document content (markdown).
            path: Document path (may help locate companion .json).

        Returns:
            ExtractedMetadata with Drafts-style tags.
        """
        result = parse_frontmatter(content)

        tags: set[str] = set()
        source_tags: list[str] = []
        attributes: dict[str, str | list[str]] = {}

        if result.has_frontmatter:
            # Drafts metadata fields
            for attr in ("uuid", "flagged", "created_at", "modified_at", "created_latitude", "created_longitude"):
                if attr in result.data and result.data[attr] is not None:
                    attributes[attr] = str(result.data[attr])

            # Tags
            if "tags" in result.data:
                field_tags = extract_tags_from_field(result.data["tags"])
                source_tags.extend(field_tags)
                for tag in field_tags:
                    if self.expand_hyphen_hierarchy:
                        tags.update(self._expand_hyphen_hierarchy(tag))
                    else:
                        tags.add(self.normalize_tag(tag))

        # Also check inline tags
        inline_tags = extract_inline_tags(result.content)
        source_tags.extend([f"#{t}" for t in inline_tags])
        for tag in inline_tags:
            if self.expand_hyphen_hierarchy:
                tags.update(self._expand_hyphen_hierarchy(tag))
            else:
                tags.add(self.normalize_tag(tag))

        return ExtractedMetadata(
            tags=tags,
            source_tags=source_tags,
            attributes=attributes,
            extraction_source=self.name,
        )

    def normalize_tag(self, tag: str) -> str:
        """Normalize a Drafts tag.

        Args:
            tag: Raw tag string.

        Returns:
            Lowercase, trimmed tag.
        """
        return tag.lstrip("#").lower().strip()

    def _expand_hyphen_hierarchy(self, tag: str) -> set[str]:
        """Expand hyphen-separated tag to include prefixes.

        For example: "project-active-sprint1" expands to
        {"project", "project-active", "project-active-sprint1"}

        Args:
            tag: Tag string with potential hyphen hierarchy.

        Returns:
            Set of normalized tags including parent prefixes.
        """
        normalized = self.normalize_tag(tag)
        if "-" not in normalized:
            return {normalized}

        result = set()
        parts = normalized.split("-")
        current = ""
        for part in parts:
            if current:
                current = f"{current}-{part}"
            else:
                current = part
            result.add(current)
        return result
