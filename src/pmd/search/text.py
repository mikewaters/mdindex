"""Text quality utilities for document processing."""


def is_indexable(content: str) -> bool:
    """Check if document content has sufficient quality for indexing.

    Documents that are empty or contain only a title heading (no body content)
    are considered low-quality and should be skipped during indexing.
    This prevents polluting search results with placeholder documents.

    Args:
        content: Raw document content.

    Returns:
        True if the document should be indexed, False to skip.
    """
    # Strip BOM and normalize whitespace
    text = content.lstrip("\ufeff").strip()

    # Empty content is not indexable
    if not text:
        return False

    lines = text.split("\n")
    first_line = lines[0].strip()

    # Check if first line is a markdown heading
    if first_line.startswith("# "):
        # Get remaining content after the heading
        remaining = "\n".join(lines[1:]).strip()
        # Title-only (no body content) is not indexable
        if not remaining:
            return False

    # Has meaningful content
    return True
