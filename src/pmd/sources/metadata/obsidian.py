from __future__ import annotations

import re

from pmd.sources.metadata.types import (
    ExtractedMetadata
)
from pmd.sources.metadata.parsing import (
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)


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


def _detect_obsidian_content(content: str, path: str) -> bool:
    """Detect Obsidian documents by content."""
    # Check for wikilinks or embeds
    if re.search(r"\[\[[^\]]+\]\]", content):
        return True

    # Check for Obsidian-style comments
    if "%%" in content:
        return True

    # Check for nested tags in frontmatter or inline
    if re.search(r"#[a-zA-Z]+/[a-zA-Z]+", content):
        return True

    return False
