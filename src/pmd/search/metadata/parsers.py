"""Parsing utilities for metadata extraction.

Provides functions to parse YAML frontmatter and extract inline tags
from markdown content.
"""

import re
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class FrontmatterResult:
    """Result from parsing YAML frontmatter.

    Attributes:
        data: Parsed YAML data as dictionary (empty if no frontmatter).
        content: Document content after frontmatter is removed.
        has_frontmatter: Whether frontmatter was found.
    """

    data: dict[str, Any]
    content: str
    has_frontmatter: bool


# Regex to match YAML frontmatter block at start of document
# Matches: ---\n<yaml content>\n---\n
_FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n?",
    re.DOTALL,
)

# Regex to match inline hashtags
# Matches: #tag, #tag/subtag, #tag-with-dashes, #tag_with_underscores
# Does NOT match: # heading, #123 (pure numbers), # (bare hash)
_INLINE_TAG_PATTERN = re.compile(
    r"(?<![^\s([\"{])#([a-zA-Z][a-zA-Z0-9_/-]*)",
)


def parse_frontmatter(content: str) -> FrontmatterResult:
    """Parse YAML frontmatter from markdown content.

    Extracts the YAML block between --- delimiters at the start
    of the document, if present.

    Args:
        content: Full markdown document content.

    Returns:
        FrontmatterResult with parsed data and remaining content.

    Example:
        >>> result = parse_frontmatter('''---
        ... title: My Doc
        ... tags: [python, code]
        ... ---
        ... # Hello World
        ... ''')
        >>> result.data
        {'title': 'My Doc', 'tags': ['python', 'code']}
        >>> result.has_frontmatter
        True
    """
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return FrontmatterResult(data={}, content=content, has_frontmatter=False)

    yaml_text = match.group(1)
    remaining_content = content[match.end() :]

    try:
        data = yaml.safe_load(yaml_text)
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            # YAML could parse to a scalar or list - wrap or reject
            data = {"_raw": data}
    except yaml.YAMLError:
        # Invalid YAML - treat as no frontmatter
        return FrontmatterResult(data={}, content=content, has_frontmatter=False)

    return FrontmatterResult(data=data, content=remaining_content, has_frontmatter=True)


def extract_inline_tags(content: str) -> list[str]:
    """Extract inline hashtags from content.

    Finds all #tag patterns in the content. Supports:
    - Simple tags: #python
    - Nested tags: #parent/child
    - Tags with dashes/underscores: #my-tag, #my_tag

    Ignores:
    - Markdown headings: # Heading
    - Pure numbers: #123
    - Tags in code blocks (basic heuristic)

    Args:
        content: Document content to search.

    Returns:
        List of tags found (without the # prefix), in order of appearance.
        Duplicates are preserved to show frequency.

    Example:
        >>> extract_inline_tags("Check #python and #rust/async code")
        ['python', 'rust/async']
    """
    # Remove code blocks to avoid matching tags inside them
    # This is a simple heuristic - fenced blocks and inline code
    content_no_code = _remove_code_blocks(content)

    matches = _INLINE_TAG_PATTERN.findall(content_no_code)
    return matches


def _remove_code_blocks(content: str) -> str:
    """Remove fenced and inline code from content.

    Args:
        content: Document content.

    Returns:
        Content with code blocks replaced by spaces (preserves positions).
    """
    # Remove fenced code blocks (``` ... ```)
    result = re.sub(r"```.*?```", lambda m: " " * len(m.group()), content, flags=re.DOTALL)
    # Remove inline code (` ... `)
    result = re.sub(r"`[^`]+`", lambda m: " " * len(m.group()), result)
    return result


def extract_tags_from_field(value: Any) -> list[str]:
    """Extract tags from a frontmatter field value.

    Handles various YAML formats for tags:
    - String: "tag1, tag2" or "tag1 tag2"
    - List: ["tag1", "tag2"]
    - Single value: "tag1"

    Args:
        value: Field value from YAML frontmatter.

    Returns:
        List of extracted tag strings.
    """
    if value is None:
        return []

    if isinstance(value, list):
        # Flatten and convert to strings
        tags = []
        for item in value:
            if isinstance(item, str):
                tags.append(item.strip())
            elif item is not None:
                tags.append(str(item).strip())
        return [t for t in tags if t]

    if isinstance(value, str):
        # Split by comma or whitespace if multiple
        if "," in value:
            return [t.strip() for t in value.split(",") if t.strip()]
        # Check for space-separated (but be careful with multi-word tags)
        # Only split if it looks like a list (multiple #-prefixed items)
        if re.search(r"#\w+\s+#\w+", value):
            return [t.lstrip("#").strip() for t in re.findall(r"#?\w+", value) if t.strip()]
        # Single tag
        return [value.strip()] if value.strip() else []

    # Fallback for other types
    return [str(value).strip()] if value else []
