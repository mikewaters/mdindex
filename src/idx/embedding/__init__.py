"""idx.embedding - Embedding models for vector generation.

Provides embedding model implementations for generating vector representations
of text. Currently supports MLX-based embeddings for efficient inference on
Apple Silicon devices.

Example usage:
    from idx.embedding import MLXEmbedding

    embed_model = MLXEmbedding()
    embedding = embed_model.get_text_embedding("Hello world")
"""

from idx.embedding.mlx import MLXEmbedding

__all__ = ["MLXEmbedding"]
