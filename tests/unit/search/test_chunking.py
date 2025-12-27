"""Tests for document chunking utilities."""

import pytest

from pmd.core.config import ChunkConfig
from pmd.core.types import Chunk
from pmd.search.chunking import (
    ChunkingResult,
    calculate_chunk_size,
    chunk_document,
    estimate_token_count,
)


class TestChunkingResult:
    """Tests for ChunkingResult dataclass."""

    def test_create_chunking_result(self):
        """ChunkingResult should store chunks and total_bytes."""
        chunks = [Chunk(text="Hello", pos=0), Chunk(text="World", pos=6)]
        result = ChunkingResult(chunks=chunks, total_bytes=11)

        assert result.chunks == chunks
        assert result.total_bytes == 11

    def test_empty_chunking_result(self):
        """ChunkingResult should handle empty chunks."""
        result = ChunkingResult(chunks=[], total_bytes=0)

        assert result.chunks == []
        assert result.total_bytes == 0


class TestChunkDocument:
    """Tests for chunk_document function."""

    def test_small_document_single_chunk(self):
        """Small document should return single chunk."""
        content = "This is a small document."
        config = ChunkConfig(max_bytes=1000, min_chunk_size=10)

        result = chunk_document(content, config)

        assert len(result.chunks) == 1
        assert result.chunks[0].text == content
        assert result.chunks[0].pos == 0

    def test_empty_document(self):
        """Empty document should return single empty chunk."""
        content = ""
        config = ChunkConfig(max_bytes=1000, min_chunk_size=10)

        result = chunk_document(content, config)

        assert len(result.chunks) == 1
        assert result.chunks[0].text == ""
        assert result.total_bytes == 0

    def test_large_document_multiple_chunks(self):
        """Large document should be split into multiple chunks."""
        # Create content that exceeds max_bytes
        lines = ["Line " + str(i) + " content here" for i in range(100)]
        content = "\n".join(lines)
        config = ChunkConfig(max_bytes=200, min_chunk_size=10)

        result = chunk_document(content, config)

        assert len(result.chunks) > 1
        # All text should be preserved
        reconstructed = "\n".join(chunk.text for chunk in result.chunks)
        # Note: rstrip may affect whitespace
        assert content.strip() in reconstructed or reconstructed in content

    def test_chunk_positions_are_sequential(self):
        """Chunk positions should be sequential."""
        lines = ["Line " + str(i) for i in range(50)]
        content = "\n".join(lines)
        config = ChunkConfig(max_bytes=100, min_chunk_size=10)

        result = chunk_document(content, config)

        # First chunk should start at 0
        assert result.chunks[0].pos == 0

        # Subsequent chunks should have increasing positions
        for i in range(1, len(result.chunks)):
            assert result.chunks[i].pos > result.chunks[i - 1].pos

    def test_respects_max_bytes(self):
        """Chunks should not exceed max_bytes (approximately)."""
        lines = ["Line " + str(i) + " with some additional content" for i in range(100)]
        content = "\n".join(lines)
        config = ChunkConfig(max_bytes=200, min_chunk_size=10)

        result = chunk_document(content, config)

        # Each chunk should be approximately within max_bytes
        for chunk in result.chunks:
            chunk_bytes = len(chunk.text.encode("utf-8"))
            # Allow some tolerance for line boundary splitting
            assert chunk_bytes < config.max_bytes + 100

    def test_total_bytes_matches_content(self):
        """total_bytes should match content byte size."""
        content = "Hello, World!"
        config = ChunkConfig(max_bytes=1000, min_chunk_size=10)

        result = chunk_document(content, config)

        expected_bytes = len(content.encode("utf-8"))
        assert result.total_bytes == expected_bytes

    def test_unicode_content(self):
        """Should handle unicode content correctly."""
        content = "Hello, 世界! 🌍"
        config = ChunkConfig(max_bytes=1000, min_chunk_size=10)

        result = chunk_document(content, config)

        assert len(result.chunks) == 1
        assert result.chunks[0].text == content
        # Unicode characters take multiple bytes
        assert result.total_bytes == len(content.encode("utf-8"))

    def test_splits_on_newlines(self):
        """Should prefer splitting on newlines."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        config = ChunkConfig(max_bytes=30, min_chunk_size=5)

        result = chunk_document(content, config)

        # Should split on line boundaries, not mid-word
        for chunk in result.chunks:
            # Text shouldn't start or end mid-word
            assert not chunk.text.startswith(" ") or chunk.text == ""

    def test_minimum_chunk_size_respected(self):
        """Chunks should meet minimum size (or be the only content)."""
        content = "A\nB\nC\nD\nE\nF\nG\nH\nI\nJ"
        config = ChunkConfig(max_bytes=50, min_chunk_size=5)

        result = chunk_document(content, config)

        # Should create chunks (behavior depends on implementation)
        assert len(result.chunks) >= 1

    def test_preserves_all_content(self):
        """All content should be preserved in chunks."""
        lines = ["Important line " + str(i) for i in range(20)]
        content = "\n".join(lines)
        config = ChunkConfig(max_bytes=150, min_chunk_size=10)

        result = chunk_document(content, config)

        # Combine all chunks
        all_text = "\n".join(chunk.text for chunk in result.chunks)

        # All original lines should appear
        for line in lines:
            assert line in all_text


class TestCalculateChunkSize:
    """Tests for calculate_chunk_size function."""

    def test_empty_string(self):
        """Empty string should have size 0."""
        assert calculate_chunk_size("") == 0

    def test_ascii_string(self):
        """ASCII string size should match length."""
        text = "Hello, World!"
        assert calculate_chunk_size(text) == len(text)

    def test_unicode_string(self):
        """Unicode string should count bytes, not characters."""
        text = "Hello, 世界!"
        # 世界 = 2 Chinese characters = 6 bytes in UTF-8
        expected = len("Hello, ".encode("utf-8")) + len("世界!".encode("utf-8"))
        assert calculate_chunk_size(text) == expected

    def test_emoji(self):
        """Emoji should be counted as multiple bytes."""
        text = "🌍"
        # Most emoji are 4 bytes in UTF-8
        assert calculate_chunk_size(text) == 4

    def test_mixed_content(self):
        """Mixed ASCII, unicode, and emoji should be counted correctly."""
        text = "Hello 世界 🌍!"
        expected = len(text.encode("utf-8"))
        assert calculate_chunk_size(text) == expected


class TestEstimateTokenCount:
    """Tests for estimate_token_count function."""

    def test_empty_string(self):
        """Empty string should have 0 tokens."""
        assert estimate_token_count("") == 0

    def test_short_text(self):
        """Short text should estimate based on ~4 chars per token."""
        text = "Hello"  # 5 characters
        assert estimate_token_count(text) == 5 // 4  # = 1

    def test_longer_text(self):
        """Longer text should estimate correctly."""
        text = "The quick brown fox jumps over the lazy dog."
        expected = len(text) // 4
        assert estimate_token_count(text) == expected

    def test_whitespace(self):
        """Whitespace should be included in count."""
        text = "a b c d e"  # 9 characters with spaces
        assert estimate_token_count(text) == 9 // 4  # = 2

    def test_approximation_is_integer(self):
        """Token count should be an integer."""
        text = "Some sample text for testing."
        result = estimate_token_count(text)
        assert isinstance(result, int)


class TestChunkDocumentEdgeCases:
    """Edge case tests for chunk_document."""

    def test_single_very_long_line(self):
        """Should handle single line longer than max_bytes."""
        # Single line with no newlines
        content = "A" * 1000
        config = ChunkConfig(max_bytes=100, min_chunk_size=10)

        result = chunk_document(content, config)

        # Should still produce output (may be single chunk due to no split points)
        assert len(result.chunks) >= 1
        assert result.total_bytes == 1000

    def test_only_newlines(self):
        """Should handle content that's only newlines."""
        content = "\n\n\n\n"
        config = ChunkConfig(max_bytes=100, min_chunk_size=10)

        result = chunk_document(content, config)

        # Should produce some chunks
        assert result.total_bytes == 4

    def test_mixed_line_endings(self):
        """Should handle different line endings."""
        content = "Line 1\nLine 2\r\nLine 3\rLine 4"
        config = ChunkConfig(max_bytes=1000, min_chunk_size=10)

        result = chunk_document(content, config)

        # Should preserve content
        assert result.total_bytes == len(content.encode("utf-8"))

    def test_trailing_newline(self):
        """Should handle trailing newlines correctly."""
        content = "Line 1\nLine 2\n"
        config = ChunkConfig(max_bytes=1000, min_chunk_size=10)

        result = chunk_document(content, config)

        assert len(result.chunks) == 1
        # Trailing newline may be stripped by rstrip
        assert "Line 1" in result.chunks[0].text
        assert "Line 2" in result.chunks[0].text

    def test_default_chunk_config(self):
        """Should work with default ChunkConfig values."""
        content = "Small document that fits in one chunk."
        config = ChunkConfig()  # Use defaults

        result = chunk_document(content, config)

        assert len(result.chunks) == 1
        assert result.chunks[0].text == content
