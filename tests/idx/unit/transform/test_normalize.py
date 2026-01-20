"""Tests for idx.transform.normalize module."""

from pathlib import Path

import pytest

from idx.transform.normalize import (
    MimeDetector,
    TextNormalizer,
    TextPolicy,
    detect_mime,
    is_text_file,
    is_text_mime,
)


class TestMimeDetector:
    """Tests for MimeDetector class."""

    @pytest.fixture
    def detector(self) -> MimeDetector:
        """Create a MimeDetector instance."""
        return MimeDetector()

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("README.md", "text/markdown"),
            ("notes.markdown", "text/markdown"),
            ("config.yaml", "text/yaml"),
            ("config.yml", "text/yaml"),
            ("data.json", "application/json"),
            ("settings.toml", "text/x-toml"),
            ("script.py", "text/x-python"),
            ("app.js", "text/javascript"),
            ("component.tsx", "text/typescript"),
            ("main.go", "text/x-go"),
            ("lib.rs", "text/x-rust"),
            ("doc.txt", "text/plain"),
            ("page.html", "text/html"),
            ("style.css", "text/css"),
            ("query.sql", "text/x-sql"),
        ],
    )
    def test_detect_common_text_types(
        self, detector: MimeDetector, filename: str, expected: str
    ) -> None:
        """Detector correctly identifies common text file types."""
        path = Path(filename)
        result = detector.detect(path)
        assert result == expected

    @pytest.mark.parametrize(
        "filename",
        [
            ".gitignore",
            ".dockerignore",
            ".editorconfig",
            "Makefile",
            "Dockerfile",
        ],
    )
    def test_detect_dotfiles_and_special_names(
        self, detector: MimeDetector, filename: str
    ) -> None:
        """Detector handles dotfiles and special filenames."""
        path = Path(filename)
        result = detector.detect(path)
        assert result == "text/plain"

    @pytest.mark.parametrize(
        "filename",
        [
            "image.png",
            "photo.jpg",
            "image.jpeg",
            "animation.gif",
            "document.pdf",
        ],
    )
    def test_detect_binary_types(self, detector: MimeDetector, filename: str) -> None:
        """Detector correctly identifies binary file types."""
        path = Path(filename)
        result = detector.detect(path)
        assert result is not None
        assert not result.startswith("text/")

    def test_detect_unknown_extension(self, detector: MimeDetector) -> None:
        """Detector returns None for unknown extensions."""
        path = Path("file.xyz123unknown")
        result = detector.detect(path)
        assert result is None

    def test_detect_from_content_text(self, detector: MimeDetector) -> None:
        """Detector identifies text from content."""
        content = b"Hello, world!\nThis is text."
        result = detector.detect_from_content(content)
        assert result == "text/plain"

    def test_detect_from_content_with_path_hint(self, detector: MimeDetector) -> None:
        """Detector uses path hint when available."""
        content = b"# Heading\nSome content"
        path = Path("readme.md")
        result = detector.detect_from_content(content, path)
        assert result == "text/markdown"

    def test_detect_from_content_binary(self, detector: MimeDetector) -> None:
        """Detector identifies binary content."""
        # PNG header
        content = b"\x89PNG\r\n\x1a\n\x00\x00"
        result = detector.detect_from_content(content)
        assert result == "application/octet-stream"


class TestTextPolicy:
    """Tests for TextPolicy class."""

    @pytest.fixture
    def policy(self) -> TextPolicy:
        """Create a TextPolicy instance."""
        return TextPolicy()

    @pytest.mark.parametrize(
        "filename",
        [
            "readme.md",
            "script.py",
            "data.json",
            "config.yaml",
            "notes.txt",
            "style.css",
            "app.js",
        ],
    )
    def test_is_text_for_text_files(self, policy: TextPolicy, filename: str) -> None:
        """Policy identifies text files correctly."""
        path = Path(filename)
        assert policy.is_text(path) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "image.png",
            "photo.jpg",
            "document.pdf",
            "archive.zip",
            "music.mp3",
        ],
    )
    def test_is_text_for_binary_files(self, policy: TextPolicy, filename: str) -> None:
        """Policy identifies binary files correctly."""
        path = Path(filename)
        assert policy.is_text(path) is False

    @pytest.mark.parametrize(
        "mime_type",
        [
            "text/plain",
            "text/html",
            "text/markdown",
            "text/css",
            "text/javascript",
            "application/json",
            "application/xml",
            "application/javascript",
        ],
    )
    def test_is_text_mime_for_text_types(self, policy: TextPolicy, mime_type: str) -> None:
        """Policy identifies text mime types."""
        assert policy.is_text_mime(mime_type) is True

    @pytest.mark.parametrize(
        "mime_type",
        [
            "image/png",
            "image/jpeg",
            "application/pdf",
            "application/zip",
            "audio/mpeg",
            "video/mp4",
        ],
    )
    def test_is_text_mime_for_binary_types(self, policy: TextPolicy, mime_type: str) -> None:
        """Policy identifies binary mime types."""
        assert policy.is_text_mime(mime_type) is False

    def test_is_text_content_for_utf8(self, policy: TextPolicy) -> None:
        """Policy identifies UTF-8 text content."""
        content = "Hello, world!\nLine 2".encode("utf-8")
        assert policy.is_text_content(content) is True

    def test_is_text_content_for_empty(self, policy: TextPolicy) -> None:
        """Policy treats empty content as text."""
        assert policy.is_text_content(b"") is True

    def test_is_text_content_for_binary_signatures(self, policy: TextPolicy) -> None:
        """Policy detects common binary file signatures."""
        # PNG
        assert policy.is_text_content(b"\x89PNG\r\n\x1a\n") is False
        # JPEG
        assert policy.is_text_content(b"\xff\xd8\xff\xe0") is False
        # PDF
        assert policy.is_text_content(b"%PDF-1.4") is False
        # ZIP
        assert policy.is_text_content(b"PK\x03\x04") is False

    def test_is_text_content_with_null_bytes(self, policy: TextPolicy) -> None:
        """Policy rejects content with null bytes."""
        content = b"Hello\x00World"
        assert policy.is_text_content(content) is False


class TestTextNormalizer:
    """Tests for TextNormalizer class."""

    @pytest.fixture
    def normalizer(self) -> TextNormalizer:
        """Create a TextNormalizer instance."""
        return TextNormalizer()

    def test_strip_bom(self, normalizer: TextNormalizer) -> None:
        """Normalizer strips UTF-8 BOM."""
        text = "\ufeffHello"
        result = normalizer.normalize(text)
        assert result == "Hello"

    def test_normalize_crlf_to_lf(self, normalizer: TextNormalizer) -> None:
        """Normalizer converts CRLF to LF."""
        text = "Line 1\r\nLine 2\r\nLine 3"
        result = normalizer.normalize(text)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalize_cr_to_lf(self, normalizer: TextNormalizer) -> None:
        """Normalizer converts CR to LF."""
        text = "Line 1\rLine 2\rLine 3"
        result = normalizer.normalize(text)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_strip_trailing_whitespace(self, normalizer: TextNormalizer) -> None:
        """Normalizer strips trailing whitespace from lines."""
        text = "Line 1   \nLine 2\t\nLine 3"
        result = normalizer.normalize(text)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_collapse_blank_lines(self, normalizer: TextNormalizer) -> None:
        """Normalizer collapses excessive blank lines."""
        text = "Line 1\n\n\n\n\nLine 2"
        result = normalizer.normalize(text)
        # Default allows 2 consecutive blank lines
        assert result == "Line 1\n\n\nLine 2"

    def test_strip_leading_trailing_whitespace(self, normalizer: TextNormalizer) -> None:
        """Normalizer strips leading/trailing whitespace from text."""
        text = "\n\n  Hello  \n\n"
        result = normalizer.normalize(text)
        assert result == "Hello"

    def test_custom_max_blank_lines(self) -> None:
        """Custom max_consecutive_blank_lines is respected."""
        normalizer = TextNormalizer(max_consecutive_blank_lines=0)
        text = "Line 1\n\n\nLine 2"
        result = normalizer.normalize(text)
        assert result == "Line 1\nLine 2"

    def test_disable_bom_stripping(self) -> None:
        """BOM stripping can be disabled."""
        normalizer = TextNormalizer(strip_bom=False)
        text = "\ufeffHello"
        result = normalizer.normalize(text)
        assert result == "\ufeffHello"

    def test_disable_line_ending_normalization(self) -> None:
        """Line ending normalization can be disabled."""
        # Also disable trailing whitespace stripping since it uses split("\n")
        # which would inadvertently strip \r as trailing whitespace
        normalizer = TextNormalizer(
            normalize_line_endings=False, strip_trailing_whitespace=False
        )
        text = "Line 1\r\nLine 2"
        result = normalizer.normalize(text)
        assert "\r\n" in result

    def test_disable_trailing_whitespace_stripping(self) -> None:
        """Trailing whitespace stripping can be disabled."""
        normalizer = TextNormalizer(strip_trailing_whitespace=False)
        text = "Hello   "
        result = normalizer.normalize(text)
        # Only leading/trailing text whitespace is stripped, but line trailing is kept
        assert result == "Hello"  # Full text strip still happens

    def test_normalize_bytes(self, normalizer: TextNormalizer) -> None:
        """Normalizer handles bytes input."""
        content = "Hello\r\nWorld".encode("utf-8")
        result = normalizer.normalize_bytes(content)
        assert result == "Hello\nWorld"

    def test_normalize_bytes_with_bom(self, normalizer: TextNormalizer) -> None:
        """Normalizer handles bytes with BOM."""
        content = b"\xef\xbb\xbfHello"  # UTF-8 BOM
        result = normalizer.normalize_bytes(content)
        assert result == "Hello"

    def test_normalize_bytes_invalid_encoding(self, normalizer: TextNormalizer) -> None:
        """Normalizer raises on invalid encoding."""
        content = b"\xff\xfe"  # Invalid UTF-8
        with pytest.raises(UnicodeDecodeError):
            normalizer.normalize_bytes(content)

    def test_preserves_content_structure(self, normalizer: TextNormalizer) -> None:
        """Normalizer preserves meaningful content structure."""
        text = """# Heading

Some paragraph text.

- List item 1
- List item 2

Another paragraph."""
        result = normalizer.normalize(text)
        assert "# Heading" in result
        assert "- List item 1" in result
        assert "\n\n" in result  # Blank lines between sections preserved


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_mime(self) -> None:
        """detect_mime function works."""
        result = detect_mime(Path("test.md"))
        assert result == "text/markdown"

    def test_is_text_file(self) -> None:
        """is_text_file function works."""
        assert is_text_file(Path("readme.md")) is True
        assert is_text_file(Path("image.png")) is False

    def test_is_text_mime(self) -> None:
        """is_text_mime function works."""
        assert is_text_mime("text/plain") is True
        assert is_text_mime("image/png") is False
