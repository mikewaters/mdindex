"""Embedding generation and management for PMD."""

from ..core.config import Config
from ..core.types import Chunk
from ..search.chunking import chunk_document
from ..store.database import Database
from ..store.embeddings import EmbeddingRepository
from .base import LLMProvider


class EmbeddingGenerator:
    """Generates and manages document embeddings."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_repo: EmbeddingRepository,
        config: Config,
    ):
        """Initialize embedding generator.

        Args:
            llm_provider: LLM provider for generating embeddings.
            embedding_repo: Repository for storing embeddings.
            config: Application configuration.
        """
        self.llm = llm_provider
        self.embedding_repo = embedding_repo
        self.config = config
        self.model = llm_provider.get_default_embedding_model()

    async def embed_document(
        self,
        hash_value: str,
        content: str,
        force: bool = False,
    ) -> int:
        """Generate and store embeddings for a document.

        Args:
            hash_value: SHA256 hash of content.
            content: Document content to embed.
            force: Force regenation even if embeddings exist.

        Returns:
            Number of chunks embedded.
        """
        # Check if already embedded
        if not force and self.embedding_repo.has_embeddings(hash_value, self.model):
            return 0

        # Clear existing embeddings for this hash
        self.embedding_repo.delete_embeddings(hash_value)

        # Chunk the document
        chunking_result = chunk_document(content, self.config.chunk)

        embedded_count = 0
        for seq, chunk in enumerate(chunking_result.chunks):
            # Generate embedding for chunk
            embedding_result = await self.llm.embed(
                chunk.text,
                model=self.model,
                is_query=False,
            )

            if embedding_result:
                # Store embedding
                self.embedding_repo.store_embedding(
                    hash_value=hash_value,
                    seq=seq,
                    pos=chunk.pos,
                    embedding=embedding_result.embedding,
                    model=self.model,
                )
                embedded_count += 1

        return embedded_count

    async def embed_query(self, query: str) -> list[float] | None:
        """Generate embedding for a search query.

        Args:
            query: Search query string.

        Returns:
            Embedding vector or None on failure.
        """
        embedding_result = await self.llm.embed(
            query,
            model=self.model,
            is_query=True,
        )

        return embedding_result.embedding if embedding_result else None

    def get_embeddings_for_content(self, hash_value: str) -> list[tuple[int, int]]:
        """Get stored embeddings for content.

        Args:
            hash_value: SHA256 hash of content.

        Returns:
            List of (seq, pos) tuples for chunks with embeddings.
        """
        embeddings = self.embedding_repo.get_embeddings_for_content(hash_value)
        return [(seq, pos) for seq, pos, _ in embeddings]

    async def clear_embeddings_by_model(self, model: str) -> int:
        """Clear all embeddings for a specific model.

        Useful for switching embedding models.

        Args:
            model: Model name.

        Returns:
            Number of embeddings deleted.
        """
        return self.embedding_repo.clear_embeddings_by_model(model)

    def get_embedding_count(self) -> int:
        """Get total number of stored embeddings.

        Returns:
            Count of embedded chunks.
        """
        return self.embedding_repo.count_embeddings()

    def get_embedding_count_by_model(self, model: str) -> int:
        """Get count of embeddings for a specific model.

        Args:
            model: Model name.

        Returns:
            Count of embeddings.
        """
        return self.embedding_repo.count_embeddings(model=model)
