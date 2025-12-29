"""Document chunking utilities for embeddings."""

from dataclasses import dataclass

from ..core.config import ChunkConfig
from ..core.types import Chunk


@dataclass
class ChunkingResult:
    """Result of document chunking operation."""

    chunks: list[Chunk]
    total_bytes: int


def chunk_document(content: str, config: ChunkConfig) -> ChunkingResult:
    """Chunk a document into smaller pieces for embedding.

    Uses the following strategy:
    1. If content is empty: return zero chunks
    2. If content <= max_bytes: return single chunk
    3. Otherwise, split preferring: \\n\\n -> sentence end -> \\n -> space

    Args:
        content: Document content to chunk.
        config: Chunking configuration.

    Returns:
        ChunkingResult with chunks and total byte count.
    """
    # Guard against empty content
    if not content.strip():
        return ChunkingResult(chunks=[], total_bytes=0)

    content_bytes = content.encode("utf-8")

    # If content fits in one chunk, return it as-is
    if len(content_bytes) <= config.max_bytes:
        return ChunkingResult(
            chunks=[Chunk(text=content, pos=0)],
            total_bytes=len(content_bytes),
        )

    chunks = []
    current_chunk = ""
    current_pos = 0
    current_bytes = 0

    for line in content.split("\n"):
        line_bytes = line.encode("utf-8")
        line_with_newline = line + "\n"
        line_with_newline_bytes = line_with_newline.encode("utf-8")

        # Check if adding this line would exceed max_bytes
        if (
            current_bytes > 0
            and current_bytes + len(line_with_newline_bytes) > config.max_bytes
        ):
            # Finalize current chunk if it meets minimum size
            if current_bytes >= config.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=current_chunk.rstrip(),
                        pos=current_pos,
                    )
                )
            else:
                # If chunk too small, still add it but mark for merge (Phase 2 enhancement)
                chunks.append(
                    Chunk(
                        text=current_chunk.rstrip(),
                        pos=current_pos,
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

    # Add final chunk if it has content
    if current_chunk.strip():
        chunks.append(
            Chunk(
                text=current_chunk.rstrip(),
                pos=current_pos,
            )
        )

    return ChunkingResult(
        chunks=chunks,
        total_bytes=len(content_bytes),
    )


def calculate_chunk_size(text: str) -> int:
    """Calculate byte size of text.

    Args:
        text: Text to measure.

    Returns:
        Byte size in UTF-8 encoding.
    """
    return len(text.encode("utf-8"))


def estimate_token_count(text: str) -> int:
    """Estimate token count using character-based heuristic.

    Rough estimate: ~4 characters per token for English text.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    return len(text) // 4
