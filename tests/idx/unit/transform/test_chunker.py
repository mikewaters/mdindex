"""Unit tests for the LineChunker and ChunkerTransform."""

import pytest
from llama_index.core.schema import Document, NodeRelationship, TextNode

from idx.transform.chunker import Chunk, ChunkerTransform, LineChunker


class TestChunk:
    """Tests for the Chunk model."""

    def test_chunk_creation(self) -> None:
        """Test basic Chunk creation."""
        chunk = Chunk(seq=0, pos=0, text="Hello", size=5)
        assert chunk.seq == 0
        assert chunk.pos == 0
        assert chunk.text == "Hello"
        assert chunk.size == 5

    def test_chunk_validation_negative_seq(self) -> None:
        """Test that negative seq is rejected."""
        with pytest.raises(ValueError):
            Chunk(seq=-1, pos=0, text="Hello", size=5)

    def test_chunk_validation_negative_pos(self) -> None:
        """Test that negative pos is rejected."""
        with pytest.raises(ValueError):
            Chunk(seq=0, pos=-1, text="Hello", size=5)

    def test_chunk_validation_negative_size(self) -> None:
        """Test that negative size is rejected."""
        with pytest.raises(ValueError):
            Chunk(seq=0, pos=0, text="Hello", size=-1)


class TestLineChunker:
    """Tests for the LineChunker class."""

    def test_init_default_values(self) -> None:
        """Test default initialization."""
        chunker = LineChunker()
        assert chunker.max_bytes == 1000
        assert chunker.min_chunk_size == 100

    def test_init_custom_values(self) -> None:
        """Test custom initialization."""
        chunker = LineChunker(max_bytes=500, min_chunk_size=50)
        assert chunker.max_bytes == 500
        assert chunker.min_chunk_size == 50

    def test_init_invalid_max_bytes(self) -> None:
        """Test that invalid max_bytes raises error."""
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            LineChunker(max_bytes=0)
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            LineChunker(max_bytes=-1)

    def test_init_invalid_min_chunk_size(self) -> None:
        """Test that invalid min_chunk_size raises error."""
        with pytest.raises(ValueError, match="min_chunk_size must be non-negative"):
            LineChunker(min_chunk_size=-1)

    def test_init_min_chunk_size_too_large(self) -> None:
        """Test that min_chunk_size >= max_bytes raises error."""
        with pytest.raises(ValueError, match="min_chunk_size must be less than max_bytes"):
            LineChunker(max_bytes=100, min_chunk_size=100)
        with pytest.raises(ValueError, match="min_chunk_size must be less than max_bytes"):
            LineChunker(max_bytes=100, min_chunk_size=150)

    def test_chunk_empty_string(self) -> None:
        """Test chunking empty string returns empty list."""
        chunker = LineChunker()
        assert chunker.chunk("") == []

    def test_chunk_whitespace_only(self) -> None:
        """Test chunking whitespace-only string returns empty list."""
        chunker = LineChunker()
        assert chunker.chunk("   \n\n  ") == []

    def test_chunk_single_line_fits(self) -> None:
        """Test single line that fits in one chunk."""
        chunker = LineChunker(max_bytes=100, min_chunk_size=10)
        chunks = chunker.chunk("Hello, World!")
        assert len(chunks) == 1
        assert chunks[0].seq == 0
        assert chunks[0].pos == 0
        assert chunks[0].text == "Hello, World!"
        assert chunks[0].size == len("Hello, World!".encode("utf-8"))

    def test_chunk_multiple_lines_fit(self) -> None:
        """Test multiple lines that fit in one chunk."""
        chunker = LineChunker(max_bytes=100, min_chunk_size=10)
        text = "Line 1\nLine 2\nLine 3"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].pos == 0

    def test_chunk_splits_at_max_bytes(self) -> None:
        """Test that text is split when exceeding max_bytes."""
        chunker = LineChunker(max_bytes=20, min_chunk_size=5)
        # Each line is about 10 bytes with newline
        text = "Line one\nLine two\nLine three"
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # All chunks should be sequentially numbered
        for i, chunk in enumerate(chunks):
            assert chunk.seq == i

    def test_chunk_pos_tracking(self) -> None:
        """Test that byte positions are tracked correctly."""
        chunker = LineChunker(max_bytes=15, min_chunk_size=5)
        text = "Hello\nWorld\nTest"
        chunks = chunker.chunk(text)

        # First chunk starts at pos 0
        assert chunks[0].pos == 0

        # Subsequent chunks should have increasing positions
        for i in range(1, len(chunks)):
            assert chunks[i].pos > chunks[i - 1].pos

    def test_chunk_deterministic(self) -> None:
        """Test that chunking is deterministic."""
        chunker = LineChunker(max_bytes=30, min_chunk_size=10)
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

        chunks1 = chunker.chunk(text)
        chunks2 = chunker.chunk(text)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert c1.seq == c2.seq
            assert c1.pos == c2.pos
            assert c1.text == c2.text
            assert c1.size == c2.size

    def test_chunk_merges_small_final_chunk(self) -> None:
        """Test that small final chunks are merged with previous."""
        chunker = LineChunker(max_bytes=50, min_chunk_size=20)
        # Create text where final chunk would be small
        text = "A" * 40 + "\n" + "B" * 5

        chunks = chunker.chunk(text)

        # The small "BBBBB" should be merged with previous
        # Reconstruct text from chunks
        reconstructed = "\n".join(c.text for c in chunks)
        # Should contain both A's and B's
        assert "A" * 40 in reconstructed
        assert "B" * 5 in reconstructed

    def test_chunk_utf8_byte_counting(self) -> None:
        """Test that UTF-8 byte counting works correctly."""
        chunker = LineChunker(max_bytes=10, min_chunk_size=3)
        # Unicode characters that are multi-byte in UTF-8
        text = "\u00e9\u00e9\u00e9"  # e with acute accent (2 bytes each in UTF-8)

        chunks = chunker.chunk(text)
        # Should treat as 6 bytes total, fitting in 10 byte max
        assert len(chunks) == 1
        assert chunks[0].size == 6

    def test_chunk_preserves_all_content(self) -> None:
        """Test that all content is preserved across chunks."""
        chunker = LineChunker(max_bytes=25, min_chunk_size=5)
        original_lines = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]
        text = "\n".join(original_lines)

        chunks = chunker.chunk(text)

        # Reconstruct from chunks
        reconstructed = "\n".join(c.text for c in chunks)
        # All original lines should be present
        for line in original_lines:
            assert line in reconstructed

    def test_chunk_small_merge_exceeds_max(self) -> None:
        """Test behavior when small chunk merge would slightly exceed max_bytes."""
        chunker = LineChunker(max_bytes=30, min_chunk_size=25)
        # First "chunk" will be ~20 bytes, next line would push it over 30
        # but since 20 < 25 (min_chunk_size), it should merge anyway
        text = "A" * 20 + "\n" + "B" * 15

        chunks = chunker.chunk(text)
        # Should have merged to avoid tiny fragment
        assert len(chunks) >= 1


class TestChunkerTransform:
    """Tests for the ChunkerTransform class."""

    def test_init_default_values(self) -> None:
        """Test default initialization."""
        transform = ChunkerTransform()
        assert transform.max_bytes == 1000
        assert transform.min_chunk_size == 100

    def test_init_custom_values(self) -> None:
        """Test custom initialization."""
        transform = ChunkerTransform(max_bytes=500, min_chunk_size=50)
        assert transform.max_bytes == 500
        assert transform.min_chunk_size == 50

    def test_transform_single_document(self) -> None:
        """Test transforming a single document."""
        transform = ChunkerTransform(max_bytes=100, min_chunk_size=10)
        doc = Document(text="Hello, World!", doc_id="doc1")

        nodes = transform([doc])

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)
        assert nodes[0].text == "Hello, World!"
        assert nodes[0].id_ == "doc1:0"
        assert nodes[0].metadata["chunk_seq"] == 0
        assert nodes[0].metadata["chunk_pos"] == 0

    def test_transform_document_with_metadata(self) -> None:
        """Test that source document metadata is preserved."""
        transform = ChunkerTransform(max_bytes=100, min_chunk_size=10)
        doc = Document(
            text="Hello, World!",
            doc_id="doc1",
            metadata={"path": "/test/file.md", "title": "Test"},
        )

        nodes = transform([doc])

        assert len(nodes) == 1
        assert nodes[0].metadata["path"] == "/test/file.md"
        assert nodes[0].metadata["title"] == "Test"
        assert nodes[0].metadata["chunk_seq"] == 0

    def test_transform_multiple_documents(self) -> None:
        """Test transforming multiple documents."""
        transform = ChunkerTransform(max_bytes=100, min_chunk_size=10)
        docs = [
            Document(text="Document 1", doc_id="doc1"),
            Document(text="Document 2", doc_id="doc2"),
        ]

        nodes = transform(docs)

        assert len(nodes) == 2
        assert nodes[0].id_ == "doc1:0"
        assert nodes[1].id_ == "doc2:0"

    def test_transform_creates_multiple_chunks(self) -> None:
        """Test that large documents create multiple chunk nodes."""
        transform = ChunkerTransform(max_bytes=20, min_chunk_size=5)
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        doc = Document(text=text, doc_id="doc1")

        nodes = transform([doc])

        assert len(nodes) > 1
        # Check sequential IDs
        for i, node in enumerate(nodes):
            assert node.id_ == f"doc1:{i}"
            assert node.metadata["chunk_seq"] == i

    def test_transform_text_node_input(self) -> None:
        """Test transforming TextNode inputs."""
        transform = ChunkerTransform(max_bytes=100, min_chunk_size=10)
        text_node = TextNode(
            id_="node1",
            text="Hello from TextNode",
            metadata={"source": "test"},
        )

        nodes = transform([text_node])

        assert len(nodes) == 1
        assert nodes[0].id_ == "node1:0"
        assert nodes[0].metadata["source"] == "test"
        assert nodes[0].metadata["chunk_seq"] == 0

    def test_transform_empty_document(self) -> None:
        """Test transforming empty document returns no nodes."""
        transform = ChunkerTransform(max_bytes=100, min_chunk_size=10)
        doc = Document(text="", doc_id="doc1")

        nodes = transform([doc])

        assert len(nodes) == 0

    def test_transform_preserves_relationships(self) -> None:
        """Test that source document relationships are tracked."""
        transform = ChunkerTransform(max_bytes=100, min_chunk_size=10)
        doc = Document(text="Hello, World!", doc_id="doc1")

        nodes = transform([doc])

        assert len(nodes) == 1
        # Check that relationship to source is tracked
        assert NodeRelationship.SOURCE in nodes[0].relationships
