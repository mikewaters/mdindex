"""idx.transform.normalize - Mime detection, text policy, and normalization.

Provides utilities for:
- Detecting file mime types
- Determining if a file should be processed as text
- Normalizing text content for storage, FTS, and chunking

Example usage:
    from idx.transform.normalize import is_text_file, TextNormalizer

    if is_text_file(path):
        normalizer = TextNormalizer()
        normalized = normalizer.normalize(content)
"""

import mimetypes
import re
from pathlib import Path

__all__ = [
    "MimeDetector",
    "TextPolicy",
    "TextNormalizer",
    "is_text_file",
    "is_text_mime",
    "detect_mime",
]

# Additional text mime types not always recognized by mimetypes module
_EXTRA_TEXT_TYPES: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/x-rst",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/x-toml",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".ini": "text/x-ini",
    ".cfg": "text/x-ini",
    ".conf": "text/x-ini",
    ".sh": "text/x-shellscript",
    ".bash": "text/x-shellscript",
    ".zsh": "text/x-shellscript",
    ".fish": "text/x-shellscript",
    ".py": "text/x-python",
    ".pyi": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".jsx": "text/javascript",
    ".vue": "text/x-vue",
    ".svelte": "text/x-svelte",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".java": "text/x-java",
    ".kt": "text/x-kotlin",
    ".scala": "text/x-scala",
    ".c": "text/x-c",
    ".h": "text/x-c",
    ".cpp": "text/x-c++",
    ".hpp": "text/x-c++",
    ".cs": "text/x-csharp",
    ".swift": "text/x-swift",
    ".sql": "text/x-sql",
    ".graphql": "text/x-graphql",
    ".gql": "text/x-graphql",
    ".proto": "text/x-protobuf",
    ".tex": "text/x-tex",
    ".latex": "text/x-latex",
    ".bib": "text/x-bibtex",
    ".env": "text/plain",
    ".gitignore": "text/plain",
    ".dockerignore": "text/plain",
    ".editorconfig": "text/plain",
}

# Text mime type prefixes that indicate text content
_TEXT_MIME_PREFIXES: tuple[str, ...] = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-sh",
    "application/x-shellscript",
    "application/x-python",
    "application/x-ndjson",
    "application/toml",
    "application/yaml",
)


class MimeDetector:
    """Detects mime types for files.

    Uses Python's mimetypes module with additional mappings for
    common text file types used in development and documentation.
    """

    def __init__(self) -> None:
        """Initialize the mime detector with extended type mappings."""
        # Register additional types
        for ext, mime_type in _EXTRA_TEXT_TYPES.items():
            mimetypes.add_type(mime_type, ext)

    def detect(self, path: Path) -> str | None:
        """Detect the mime type for a file path.

        Args:
            path: Path to the file (doesn't need to exist).

        Returns:
            The detected mime type, or None if unknown.
        """
        # Check our extended types first
        suffix = path.suffix.lower()
        if suffix in _EXTRA_TEXT_TYPES:
            return _EXTRA_TEXT_TYPES[suffix]

        # Handle files without extension by name
        name = path.name.lower()
        if name in (".gitignore", ".dockerignore", ".editorconfig", "makefile", "dockerfile"):
            return "text/plain"

        # Fall back to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type

    def detect_from_content(self, content: bytes, path: Path | None = None) -> str | None:
        """Detect mime type from content with optional path hint.

        Uses simple heuristics - checks if content is valid UTF-8 text.

        Args:
            content: File content as bytes.
            path: Optional path hint for extension-based detection.

        Returns:
            The detected mime type, or None if unknown.
        """
        # If we have a path, try extension-based detection first
        if path is not None:
            mime_type = self.detect(path)
            if mime_type is not None:
                return mime_type

        # Try to detect if it's text by attempting UTF-8 decode
        try:
            content.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            return "application/octet-stream"


class TextPolicy:
    """Policy for determining if files should be processed as text.

    Defines rules for which files are considered text and should be
    indexed, versus binary files that should be skipped.
    """

    def __init__(self, detector: MimeDetector | None = None) -> None:
        """Initialize the text policy.

        Args:
            detector: Optional MimeDetector instance. Creates one if not provided.
        """
        self._detector = detector or MimeDetector()

    def is_text(self, path: Path) -> bool:
        """Check if a file should be treated as text.

        Args:
            path: Path to the file.

        Returns:
            True if the file should be processed as text.
        """
        mime_type = self._detector.detect(path)
        if mime_type is None:
            return False
        return self.is_text_mime(mime_type)

    def is_text_mime(self, mime_type: str) -> bool:
        """Check if a mime type represents text content.

        Args:
            mime_type: The mime type string.

        Returns:
            True if the mime type is considered text.
        """
        mime_lower = mime_type.lower()
        return mime_lower.startswith(_TEXT_MIME_PREFIXES)

    def is_text_content(self, content: bytes) -> bool:
        """Check if content appears to be valid text.

        Attempts to decode as UTF-8 and checks for common binary indicators.

        Args:
            content: File content as bytes.

        Returns:
            True if content appears to be text.
        """
        # Empty content is considered text
        if not content:
            return True

        # Check for common binary file signatures
        binary_signatures = [
            b"\x89PNG",  # PNG
            b"\xff\xd8\xff",  # JPEG
            b"GIF8",  # GIF
            b"PK\x03\x04",  # ZIP/Office docs
            b"%PDF",  # PDF
            b"\x7fELF",  # ELF binary
            b"MZ",  # Windows executable
        ]
        for sig in binary_signatures:
            if content.startswith(sig):
                return False

        # Try to decode as UTF-8
        try:
            text = content.decode("utf-8")
            # Check for excessive null bytes (binary indicator)
            if "\x00" in text[:1000]:
                return False
            return True
        except UnicodeDecodeError:
            return False


class TextNormalizer:
    """Normalizes text content for storage, FTS, and chunking.

    Performs operations like:
    - Stripping BOM (Byte Order Mark)
    - Normalizing line endings to \\n
    - Collapsing excessive whitespace
    - Stripping leading/trailing whitespace
    """

    def __init__(
        self,
        *,
        strip_bom: bool = True,
        normalize_line_endings: bool = True,
        collapse_blank_lines: bool = True,
        max_consecutive_blank_lines: int = 2,
        strip_trailing_whitespace: bool = True,
    ) -> None:
        """Initialize the normalizer with configuration.

        Args:
            strip_bom: Remove UTF-8 BOM if present.
            normalize_line_endings: Convert \\r\\n and \\r to \\n.
            collapse_blank_lines: Limit consecutive blank lines.
            max_consecutive_blank_lines: Maximum blank lines to allow.
            strip_trailing_whitespace: Remove trailing whitespace from lines.
        """
        self.strip_bom = strip_bom
        self.normalize_line_endings = normalize_line_endings
        self.collapse_blank_lines = collapse_blank_lines
        self.max_consecutive_blank_lines = max_consecutive_blank_lines
        self.strip_trailing_whitespace = strip_trailing_whitespace

    def normalize(self, text: str) -> str:
        """Normalize text content.

        Args:
            text: The text to normalize.

        Returns:
            Normalized text.
        """
        result = text

        # Strip BOM
        if self.strip_bom and result.startswith("\ufeff"):
            result = result[1:]

        # Normalize line endings
        if self.normalize_line_endings:
            result = result.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing whitespace from lines
        if self.strip_trailing_whitespace:
            result = "\n".join(line.rstrip() for line in result.split("\n"))

        # Collapse blank lines
        if self.collapse_blank_lines:
            # Pattern to match more than max consecutive blank lines
            pattern = r"\n{" + str(self.max_consecutive_blank_lines + 2) + r",}"
            replacement = "\n" * (self.max_consecutive_blank_lines + 1)
            result = re.sub(pattern, replacement, result)

        # Strip leading/trailing whitespace from the entire text
        result = result.strip()

        return result

    def normalize_bytes(self, content: bytes, encoding: str = "utf-8") -> str:
        """Normalize content from bytes.

        Args:
            content: Content as bytes.
            encoding: Character encoding to use.

        Returns:
            Normalized text string.

        Raises:
            UnicodeDecodeError: If content cannot be decoded.
        """
        text = content.decode(encoding)
        return self.normalize(text)


# Convenience functions
_default_detector = MimeDetector()
_default_policy = TextPolicy(_default_detector)
_default_normalizer = TextNormalizer()


def detect_mime(path: Path) -> str | None:
    """Detect mime type for a file path.

    Args:
        path: Path to the file.

    Returns:
        Detected mime type or None.
    """
    return _default_detector.detect(path)


def is_text_file(path: Path) -> bool:
    """Check if a file should be treated as text.

    Args:
        path: Path to the file.

    Returns:
        True if the file is a text file.
    """
    return _default_policy.is_text(path)


def is_text_mime(mime_type: str) -> bool:
    """Check if a mime type represents text.

    Args:
        mime_type: The mime type string.

    Returns:
        True if mime type is text.
    """
    return _default_policy.is_text_mime(mime_type)
