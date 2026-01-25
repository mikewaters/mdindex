"""idx.search.fts_chunk - LlamaIndex retriever for FTS5 chunk search.

Provides a LlamaIndex-compatible retriever that queries the chunks_fts
virtual table via FTSChunkManager.

Example usage:
    from idx.search.fts_chunk import FTSChunkRetriever
    from idx.store.database import get_session
    from idx.store.session_context import use_session

    with get_session() as session:
        with use_session(session):
            retriever = FTSChunkRetriever(similarity_top_k=10, dataset_name="obsidian")
            nodes = retriever.retrieve("machine learning concepts")
"""

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from idx.core.logging import get_logger
from idx.store.fts_chunk import FTSChunkManager

__all__ = [
    "FTSChunkRetriever",
]

logger = get_logger(__name__)


class FTSChunkRetriever(BaseRetriever):
    """LlamaIndex retriever using FTS5 full-text search on document chunks.

    Queries the chunks_fts virtual table via FTSChunkManager and converts
    results to LlamaIndex NodeWithScore objects.

    Uses ambient session via contextvars. The session must be set
    via `use_session()` before calling retrieve methods.

    Attributes:
        fts_manager: The FTSChunkManager instance for database operations.
        similarity_top_k: Maximum number of results to return.
        dataset_name: Optional dataset filter (e.g., "obsidian").

    Example:
        with get_session() as session:
            with use_session(session):
                retriever = FTSChunkRetriever(
                    similarity_top_k=10,
                    dataset_name="obsidian"
                )
                nodes = retriever.retrieve("python async patterns")
    """

    def __init__(
        self,
        fts_manager: FTSChunkManager | None = None,
        similarity_top_k: int = 10,
        dataset_name: str | None = None,
    ) -> None:
        """Initialize the FTS chunk retriever.

        Args:
            fts_manager: Optional FTSChunkManager instance. If None, creates
                a new instance using the ambient session.
            similarity_top_k: Maximum number of results to return. Defaults to 10.
            dataset_name: Optional dataset name filter. If provided, only chunks
                from documents in this dataset will be returned. The filter is
                applied via source_doc_id prefix matching (e.g., "obsidian:").
        """
        super().__init__()
        self._fts_manager = fts_manager if fts_manager is not None else FTSChunkManager()
        self._similarity_top_k = similarity_top_k
        self._dataset_name = dataset_name

    @property
    def similarity_top_k(self) -> int:
        """Get the maximum number of results to return."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, value: int) -> None:
        """Set the maximum number of results to return."""
        self._similarity_top_k = value

    @property
    def dataset_name(self) -> str | None:
        """Get the dataset name filter."""
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value: str | None) -> None:
        """Set the dataset name filter."""
        self._dataset_name = value

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes matching the query using FTS5 search.

        Args:
            query_bundle: LlamaIndex query bundle containing the search query.

        Returns:
            List of NodeWithScore objects sorted by relevance (highest score first).
        """
        query_str = query_bundle.query_str

        # Build source_doc_id prefix from dataset_name
        source_doc_id_prefix: str | None = None
        if self._dataset_name:
            source_doc_id_prefix = f"{self._dataset_name}:"

        # Execute FTS search
        fts_results = self._fts_manager.search_with_scores(
            query=query_str,
            limit=self._similarity_top_k,
            source_doc_id_prefix=source_doc_id_prefix,
        )

        logger.debug(
            f"FTS chunk search '{query_str}' returned {len(fts_results)} results"
        )

        # Convert FTSChunkResult to NodeWithScore
        nodes_with_scores: list[NodeWithScore] = []
        for result in fts_results:
            node = TextNode(
                id_=result.node_id,
                text=result.text,
                metadata={"source_doc_id": result.source_doc_id},
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=result.score))

        return nodes_with_scores
