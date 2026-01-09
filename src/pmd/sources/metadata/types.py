"""Core metadata types and protocols.

Defines the foundational types for metadata handling across both
source extraction and search inference.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ExtractedMetadata:
    """Normalized metadata extracted from a document.

    Attributes:
        tags: Normalized tag set (lowercase, trimmed, deduplicated).
        source_tags: Original tags as found in the document (preserves case/format).
        attributes: Key-value metadata from frontmatter or other sources.
        extraction_source: Which profile performed the extraction.
    """

    tags: set[str] = field(default_factory=set)
    source_tags: list[str] = field(default_factory=list)
    attributes: dict[str, str | list[str]] = field(default_factory=dict)
    extraction_source: str = ""

    def merge(self, other: "ExtractedMetadata") -> "ExtractedMetadata":
        """Merge another ExtractedMetadata into this one.

        Useful when combining metadata from multiple sources
        (e.g., frontmatter + inline tags).

        Args:
            other: Another ExtractedMetadata to merge.

        Returns:
            New ExtractedMetadata with combined values.
        """
        merged_attrs = {**self.attributes}
        for key, value in other.attributes.items():
            if key in merged_attrs:
                # Combine values into list if not already
                existing = merged_attrs[key]
                if isinstance(existing, list):
                    if isinstance(value, list):
                        merged_attrs[key] = existing + value
                    else:
                        merged_attrs[key] = existing + [value]
                else:
                    if isinstance(value, list):
                        merged_attrs[key] = [existing] + value
                    else:
                        merged_attrs[key] = [existing, value]
            else:
                merged_attrs[key] = value

        return ExtractedMetadata(
            tags=self.tags | other.tags,
            source_tags=self.source_tags + other.source_tags,
            attributes=merged_attrs,
            extraction_source=self.extraction_source or other.extraction_source,
        )


@runtime_checkable
class MetadataProfile(Protocol):
    """Protocol for app-aware metadata extraction.

    Implementations handle extracting and normalizing metadata
    from documents originating from different applications
    (e.g., Obsidian, Drafts, generic markdown).

    Example:
        class ObsidianProfile:
            name = "obsidian"

            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                # Extract YAML frontmatter and inline #tags
                ...

            def normalize_tag(self, tag: str) -> str:
                # Handle nested tags like "parent/child"
                return tag.lower().replace("/", "-")
    """

    name: str
    """Unique identifier for this profile."""

    def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
        """Extract metadata from document content.

        Args:
            content: Full document content.
            path: Document path (may influence extraction, e.g., for detection).

        Returns:
            ExtractedMetadata with tags, attributes, and source info.
        """
        ...

    def normalize_tag(self, tag: str) -> str:
        """Normalize a raw tag to canonical form.

        Handles app-specific tag formats (e.g., Obsidian's "parent/child"
        hierarchy, Drafts' "prefix-suffix" convention).

        Args:
            tag: Raw tag string from document.

        Returns:
            Normalized tag string (typically lowercase, trimmed).
        """
        ...
