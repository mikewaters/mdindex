"""Drafts app metadata profile.

Provides metadata extraction for Drafts app exports.
"""

from __future__ import annotations

import re

from pmd.metadata.model import ExtractedMetadata
from pmd.metadata.extraction.parsing import (
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)


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


def detect_drafts_content(content: str, path: str) -> bool:
    """Detect Drafts app documents by content.

    Args:
        content: Document content.
        path: Document path.

    Returns:
        True if content appears to be from Drafts app.
    """
    # Check for Drafts UUID pattern in frontmatter
    if re.search(r"uuid:\s*[a-f0-9-]{36}", content, re.IGNORECASE):
        return True

    # Check for created_latitude (Drafts geolocation)
    if "created_latitude:" in content:
        return True

    return False
