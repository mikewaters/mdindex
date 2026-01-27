"""idx.embedding.mlx - MLX-based embeddings for Apple Silicon.

Provides local embedding generation using mlx-embeddings for efficient
inference on Apple Silicon devices. Implements the LlamaIndex BaseEmbedding
interface for seamless integration with ingestion pipelines.

Example usage:
    from idx.embedding.mlx import MLXEmbedding

    embed_model = MLXEmbedding()
    embedding = embed_model.get_text_embedding("Hello world")

    # Batch processing
    embeddings = embed_model.get_text_embedding_batch(["Hello", "World"])
"""

from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from idx.core.logging import get_logger

__all__ = ["MLXEmbedding"]

logger = get_logger(__name__)

# Default model for MLX embeddings
DEFAULT_MODEL_NAME = "mlx-community/all-MiniLM-L6-v2-bf16"


class MLXEmbedding(BaseEmbedding):
    """MLX-based embedding model for Apple Silicon.

    Implements LlamaIndex's BaseEmbedding interface using mlx-embeddings
    for efficient local inference on Apple Silicon devices. Supports lazy
    model loading to minimize startup time.

    The model and tokenizer are loaded on first embedding request, not
    during initialization. This allows configuration without immediate
    resource allocation.

    Attributes:
        model_name: Name or path of the MLX embedding model to use.
            Defaults to "mlx-community/all-MiniLM-L6-v2-bf16".

    Example:
        embed_model = MLXEmbedding(model_name="mlx-community/all-MiniLM-L6-v2-4bit")
        embeddings = embed_model.get_text_embedding_batch(texts)
    """

    model_name: str = DEFAULT_MODEL_NAME

    # Private attributes for lazy-loaded model and tokenizer
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        embed_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLX embedding model.

        Args:
            model_name: Name or path of the MLX embedding model.
                Defaults to "mlx-community/all-MiniLM-L6-v2-bf16".
            embed_batch_size: Batch size for embedding generation.
                Defaults to 32.
            **kwargs: Additional arguments passed to BaseEmbedding.
        """
        logger.info(f"Initializing MLXEmbedding with model: {model_name}")
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for serialization."""
        return "MLXEmbedding"

    def _load_model(self) -> None:
        """Load the model and tokenizer if not already loaded.

        Performs lazy initialization of the MLX model and tokenizer.
        Called automatically before first embedding generation.
        """
        if self._model is None or self._tokenizer is None:
            from mlx_embeddings.utils import load

            logger.info(f"Loading MLX embedding model: {self.model_name}")
            self._model, self._tokenizer = load(self.model_name)
            logger.info(f"MLX embedding model loaded: {self.model_name}")

    def _get_text_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the text embedding.
        """
        self._load_model()

        # Tokenize and generate embedding
        inputs = self._tokenizer.batch_encode_plus(
            [text],
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = self._model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Return mean pooled and normalized embeddings
        embedding = outputs.text_embeds[0]
        return embedding.tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Processes texts in a single batch for efficiency.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one per input text.
        """
        if not texts:
            return []

        self._load_model()

        #logger.info(f"Generating embeddings for {len(texts)} texts")

        # Batch tokenize and generate embeddings
        inputs = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = self._model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Return mean pooled and normalized embeddings
        embeddings = outputs.text_embeds
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return [emb.tolist() for emb in embeddings]

    def _get_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query.

        For E5 models, queries should ideally be prefixed with "query: "
        but this implementation uses the same method as text embedding
        for simplicity. Override in subclass if query-specific handling
        is needed.

        Args:
            query: The query text to embed.

        Returns:
            List of floats representing the query embedding.
        """
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async wrapper for query embedding generation.

        MLX operations are synchronous, so this wraps the sync method.

        Args:
            query: The query text to embed.

        Returns:
            List of floats representing the query embedding.
        """
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async wrapper for single text embedding generation.

        MLX operations are synchronous, so this wraps the sync method.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the text embedding.
        """
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper for batch text embedding generation.

        MLX operations are synchronous, so this wraps the sync method.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one per input text.
        """
        return self._get_text_embeddings(texts)
