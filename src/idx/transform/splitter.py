"""idx.transform.splitter - Size-aware chunk splitting transform.

Provides a LlamaIndex TransformComponent that conditionally splits oversized
nodes using SentenceSplitter as a fallback mechanism.

Example usage:
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import MarkdownNodeParser
    from idx.transform.splitter import SizeAwareChunkSplitter

    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SizeAwareChunkSplitter(max_chars=2000),
        ]
    )
    nodes = pipeline.run(documents=documents)
"""

from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent

from idx.core.logging import get_logger

__all__ = ["SizeAwareChunkSplitter"]

logger = get_logger(__name__)


class SizeAwareChunkSplitter(TransformComponent):
    """LlamaIndex TransformComponent that splits oversized nodes.

    This transform conditionally splits nodes that exceed a character threshold
    using LlamaIndex's SentenceSplitter. Nodes within the size limit pass
    through unchanged, preserving their original structure.

    Useful as a post-processing step after MarkdownNodeParser to handle
    sections that are too large for embedding models while preserving
    the semantic structure of smaller chunks.

    Attributes:
        max_chars: Maximum character count before triggering split.
        fallback_chunk_size: Target chunk size for SentenceSplitter.
        fallback_chunk_overlap: Overlap between chunks for SentenceSplitter.
    """

    max_chars: int = 2000
    fallback_chunk_size: int = 512
    fallback_chunk_overlap: int = 50

    def __init__(
        self,
        *,
        max_chars: int = 2000,
        fallback_chunk_size: int = 512,
        fallback_chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> None:
        """Initialize the size-aware chunk splitter.

        Args:
            max_chars: Maximum character count before triggering fallback split.
                Nodes with content exceeding this threshold will be split.
            fallback_chunk_size: Target chunk size in characters for
                SentenceSplitter when splitting oversized nodes.
            fallback_chunk_overlap: Number of overlapping characters between
                chunks when using SentenceSplitter.
            **kwargs: Additional arguments passed to TransformComponent.
        """
        super().__init__(**kwargs)
        self.max_chars = max_chars
        self.fallback_chunk_size = fallback_chunk_size
        self.fallback_chunk_overlap = fallback_chunk_overlap

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Process nodes, splitting those that exceed the size threshold.

        Nodes within the max_chars limit pass through unchanged. Oversized
        nodes are split using SentenceSplitter with the configured parameters.
        Metadata from the original node is preserved on all resulting chunks.

        Args:
            nodes: List of nodes to process.
            **kwargs: Additional arguments (unused).

        Returns:
            List of nodes with oversized ones split into smaller chunks.
        """
        logger.info(f"SizeAwareChunkSplitter: processing {len(nodes)} nodes")

        result: list[BaseNode] = []
        split_count = 0
        pass_through_count = 0

        # Create splitter instance for oversized nodes
        splitter = SentenceSplitter(
            chunk_size=self.fallback_chunk_size,
            chunk_overlap=self.fallback_chunk_overlap,
        )

        for node in nodes:
            content = node.get_content()
            content_len = len(content)

            if content_len <= self.max_chars:
                # Node is within size limit - pass through unchanged
                result.append(node)
                pass_through_count += 1
            else:
                # Node exceeds threshold - split using SentenceSplitter
                logger.debug(
                    f"Splitting oversized node ({content_len} chars > {self.max_chars}): "
                    f"{node.node_id[:50]}..."
                )

                # SentenceSplitter.get_nodes_from_documents expects nodes
                split_nodes = splitter.get_nodes_from_documents([node])

                # Preserve original metadata on all split chunks
                original_metadata = node.metadata.copy() if node.metadata else {}
                for i, split_node in enumerate(split_nodes):
                    # Merge original metadata (split_node may have its own metadata)
                    merged_metadata = original_metadata.copy()
                    if split_node.metadata:
                        merged_metadata.update(split_node.metadata)
                    split_node.metadata = merged_metadata

                    # Track that this came from a split
                    split_node.metadata["_split_from"] = node.node_id
                    split_node.metadata["_split_index"] = i
                    split_node.metadata["_split_total"] = len(split_nodes)

                result.extend(split_nodes)
                split_count += 1

                logger.debug(
                    f"Split node into {len(split_nodes)} chunks"
                )

        if split_count > 0:
            logger.info(
                f"SizeAwareChunkSplitter complete: {pass_through_count} passed through, "
                f"{split_count} split -> {len(result)} total nodes"
            )
        else:
            logger.info(
                f"SizeAwareChunkSplitter complete: {len(result)} nodes (no splitting needed)"
            )

        return result
