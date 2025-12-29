"""Text normalization utilities for document processing."""

from dataclasses import dataclass


@dataclass
class NormalizedContent:
    """Result of normalizing document content for indexing."""

    embeddable_body: str
    """Content suitable for embedding. Empty if title-only."""

    fts_body: str
    """Content for FTS indexing. Uses title if body is empty."""

    title: str | None
    """Extracted title from heading, if present."""

    is_title_only: bool
    """True if document contains only a title heading."""


def normalize_content(content: str) -> NormalizedContent:
    """Normalize document content for embedding and FTS indexing.

    Detects title-only documents (single `# Heading` with no meaningful body)
    and returns appropriate content for each indexing path:
    - embeddable_body: Empty for title-only docs (skip embedding)
    - fts_body: Title text for title-only docs (remain searchable via BM25)

    Args:
        content: Raw document content.

    Returns:
        NormalizedContent with separated embedding and FTS content.
    """
    # Strip BOM and normalize whitespace
    text = content.lstrip("\ufeff").strip()

    # Empty content is title-only
    if not text:
        return NormalizedContent(
            embeddable_body="",
            fts_body="",
            title=None,
            is_title_only=True,
        )

    lines = text.split("\n")
    first_line = lines[0].strip()

    # Check if first line is a markdown heading
    if first_line.startswith("# "):
        title = first_line[2:].strip()
        # Get remaining content after the heading
        remaining = "\n".join(lines[1:]).strip()

        if not remaining:
            # Title-only: no content after heading
            return NormalizedContent(
                embeddable_body="",
                fts_body=title,
                title=title,
                is_title_only=True,
            )
        else:
            # Has body content
            return NormalizedContent(
                embeddable_body=content,
                fts_body=content,
                title=title,
                is_title_only=False,
            )
    else:
        # No heading - treat as regular content
        return NormalizedContent(
            embeddable_body=content,
            fts_body=content,
            title=None,
            is_title_only=False,
        )
