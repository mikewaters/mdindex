"""idx.llm - LLM provider abstractions.

Normalizes LLM providers (MLX via mlx-lm).
Provides LLM-as-judge reranking with position-aware score blending.

Example:
    from idx.llm import Reranker, MLXProvider

    reranker = Reranker()
    results = await reranker.rerank(query, search_results)
"""

from idx.llm.prompts import RERANK_PROMPT, RERANK_SYSTEM, format_rerank_prompt
from idx.llm.provider import LLMProviderError, MLXProvider
from idx.llm.reranker import (
    RerankConfig,
    RerankScore,
    Reranker,
    blend_scores,
    get_position_weight,
)

__all__ = [
    # Provider
    "LLMProviderError",
    "MLXProvider",
    # Reranker
    "RerankConfig",
    "RerankScore",
    "Reranker",
    "blend_scores",
    "get_position_weight",
    # Prompts
    "RERANK_PROMPT",
    "RERANK_SYSTEM",
    "format_rerank_prompt",
]
