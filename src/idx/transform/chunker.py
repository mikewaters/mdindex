"""Deterministic line-oriented chunker for document processing.

This module provides a deterministic, line-oriented chunking strategy
for splitting documents into smaller pieces suitable for embedding.
The chunker preserves byte offsets for provenance tracking.
"""

from typing import Any

from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeRelationship,
    TextNode,
    TransformComponent,
)
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of text from a document.

    Attributes:
        seq: Sequence number (0-indexed) within the document.
        pos: Byte offset of this chunk's start in the original document.
        text: The chunk content.
        size: Byte size of the chunk in UTF-8 encoding.
    """

    seq: int = Field(ge=0, description="0-indexed sequence number")
    pos: int = Field(ge=0, description="Byte offset in original document")
    text: str = Field(description="Chunk content")
    size: int = Field(ge=0, description="Byte size of chunk")


class LineChunker:
    """Deterministic line-oriented document chunker.

    Splits documents into chunks by accumulating lines until a byte budget
    is reached. Avoids tiny fragments by merging small final chunks.

    Attributes:
        max_bytes: Maximum byte size for a chunk (default: 1000).
        min_chunk_size: Minimum byte size before merging with previous (default: 100).
    """

    def __init__(self, max_bytes: int = 1000, min_chunk_size: int = 100) -> None:
        """Initialize the chunker.

        Args:
            max_bytes: Maximum byte size for a chunk.
            min_chunk_size: Minimum byte size; smaller final chunks are merged.
        """
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size must be non-negative")
        if min_chunk_size >= max_bytes:
            raise ValueError("min_chunk_size must be less than max_bytes")

        self.max_bytes = max_bytes
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks.

        Algorithm:
        1. Split text into lines (preserving line endings)
        2. Accumulate lines until adding the next line would exceed max_bytes
        3. When max_bytes is reached, emit chunk and start new accumulator
        4. After all lines processed, if final chunk < min_chunk_size, merge with previous
        5. Track pos as cumulative byte offset at chunk start

        Args:
            text: The text to chunk.

        Returns:
            List of Chunk objects with sequence numbers and byte offsets.
        """
        # Handle empty content
        if not text.strip():
            return []

        content_bytes = text.encode("utf-8")

        # If content fits in one chunk, return it as-is
        if len(content_bytes) <= self.max_bytes:
            return [
                Chunk(
                    seq=0,
                    pos=0,
                    text=text,
                    size=len(content_bytes),
                )
            ]

        chunks: list[Chunk] = []
        current_chunk = ""
        current_pos = 0
        current_bytes = 0

        # Split preserving line structure (split on \n, then re-add)
        for line in text.split("\n"):
            line_with_newline = line + "\n"
            line_with_newline_bytes = line_with_newline.encode("utf-8")

            # Check if adding this line would exceed max_bytes
            if (
                current_bytes > 0
                and current_bytes + len(line_with_newline_bytes) > self.max_bytes
            ):
                # If the current chunk is too small, merge it with this line even
                # if it slightly exceeds max_bytes to avoid tiny fragments.
                if current_bytes < self.min_chunk_size:
                    current_chunk += line_with_newline
                    current_bytes += len(line_with_newline_bytes)
                    chunk_text = current_chunk.rstrip()
                    chunks.append(
                        Chunk(
                            seq=len(chunks),
                            pos=current_pos,
                            text=chunk_text,
                            size=len(chunk_text.encode("utf-8")),
                        )
                    )
                    current_pos += current_bytes
                    current_chunk = ""
                    current_bytes = 0
                    continue

                # Finalize current chunk if it meets minimum size
                chunk_text = current_chunk.rstrip()
                chunks.append(
                    Chunk(
                        seq=len(chunks),
                        pos=current_pos,
                        text=chunk_text,
                        size=len(chunk_text.encode("utf-8")),
                    )
                )

                # Start new chunk with this line
                current_chunk = line_with_newline
                current_pos += current_bytes
                current_bytes = len(line_with_newline_bytes)
            else:
                # Add line to current chunk
                current_chunk += line_with_newline
                current_bytes += len(line_with_newline_bytes)

        # Handle final chunk
        if current_chunk.strip():
            chunk_text = current_chunk.rstrip()
            final_size = len(chunk_text.encode("utf-8"))

            # Merge with previous chunk if too small
            if final_size < self.min_chunk_size and chunks:
                prev_chunk = chunks.pop()
                merged_text = prev_chunk.text + "\n" + chunk_text
                chunks.append(
                    Chunk(
                        seq=prev_chunk.seq,
                        pos=prev_chunk.pos,
                        text=merged_text,
                        size=len(merged_text.encode("utf-8")),
                    )
                )
            else:
                chunks.append(
                    Chunk(
                        seq=len(chunks),
                        pos=current_pos,
                        text=chunk_text,
                        size=final_size,
                    )
                )

        return chunks


class ChunkerTransform(TransformComponent):
    """LlamaIndex TransformComponent wrapper for LineChunker.

    This transform takes a list of Documents and returns TextNodes
    with chunk metadata for use in LlamaIndex pipelines.

    Attributes:
        max_bytes: Maximum byte size for a chunk (default: 1000).
        min_chunk_size: Minimum byte size before merging (default: 100).
    """

    max_bytes: int = 1000
    min_chunk_size: int = 100

    def __init__(
        self,
        max_bytes: int = 1000,
        min_chunk_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize the transform.

        Args:
            max_bytes: Maximum byte size for a chunk.
            min_chunk_size: Minimum byte size for a chunk.
            **kwargs: Additional arguments passed to TransformComponent.
        """
        super().__init__(**kwargs)
        self.max_bytes = max_bytes
        self.min_chunk_size = min_chunk_size

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Transform documents into chunked nodes.

        Args:
            nodes: List of Document or BaseNode objects to chunk.
            **kwargs: Additional arguments (unused).

        Returns:
            List of TextNode objects with chunk metadata.
        """
        chunker = LineChunker(
            max_bytes=self.max_bytes,
            min_chunk_size=self.min_chunk_size,
        )

        result_nodes: list[BaseNode] = []

        for node in nodes:
            # Get text content from node
            if isinstance(node, Document):
                text = node.text
                doc_id = node.doc_id
                source_metadata = node.metadata.copy() if node.metadata else {}
            else:
                text = node.get_content()
                doc_id = node.node_id
                source_metadata = node.metadata.copy() if node.metadata else {}

            # Chunk the text
            chunks = chunker.chunk(text)

            # Create nodes for each chunk
            for chunk in chunks:
                # Build node ID from doc_id and sequence
                node_id = f"{doc_id}:{chunk.seq}"

                # Merge source metadata with chunk metadata
                chunk_metadata = {
                    **source_metadata,
                    "chunk_seq": chunk.seq,
                    "chunk_pos": chunk.pos,
                    "chunk_size": chunk.size,
                }

                text_node = TextNode(
                    id_=node_id,
                    text=chunk.text,
                    metadata=chunk_metadata,
                )

                # Track relationship to source document
                if isinstance(node, Document):
                    text_node.relationships = {
                        NodeRelationship.SOURCE: node.as_related_node_info(),
                    }

                result_nodes.append(text_node)

        return result_nodes
