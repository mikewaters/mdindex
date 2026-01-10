"""Generic metadata profile.

Provides a fallback profile with minimal normalization for
standard markdown documents.
"""

from pmd.metadata.model import ExtractedMetadata
from pmd.metadata.extraction.parsing import (
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)


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
                if attr in result.data and result.data[attr] is not None:
                    value = result.data[attr]
                    if isinstance(value, list):
                        attributes[attr] = [str(v) for v in value]
                    elif not isinstance(value, dict):
                        # Convert scalars (str, int, float, datetime, etc.) to string
                        attributes[attr] = str(value)

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
