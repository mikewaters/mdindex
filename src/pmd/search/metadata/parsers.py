"""Backward-compatible shim for metadata parsing helpers."""

from pmd.sources.metadata.parsers import (
    FrontmatterResult,
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)

__all__ = [
    "FrontmatterResult",
    "extract_inline_tags",
    "extract_tags_from_field",
    "parse_frontmatter",
]
