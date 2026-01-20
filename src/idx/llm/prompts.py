"""idx.llm.prompts - Centrally-managed prompts for LLM operations.

All prompts for LLM-as-judge reranking and other LLM operations.
Prompts are designed to work with small local models (MLX).
"""

from __future__ import annotations

__all__ = [
    "RERANK_PROMPT",
    "RERANK_SYSTEM",
    "format_rerank_prompt",
]


# System prompt for reranking (used with chat models)
RERANK_SYSTEM = """You are a relevance judge. Your task is to determine if a document is relevant to a search query.
Respond with ONLY 'Yes' if the document answers or relates to the query, or 'No' if it does not.
Do not explain your reasoning. Answer with a single word: Yes or No."""


# Template for rerank user message
RERANK_PROMPT = """Query: {query}

Document:
{document}

Is this document relevant to the query? Answer Yes or No:"""


def format_rerank_prompt(
    query: str,
    document_text: str,
    max_doc_chars: int = 2000,
) -> str:
    """Format the rerank prompt with query and document.

    Args:
        query: The search query.
        document_text: The document text to evaluate (full doc or chunk).
        max_doc_chars: Maximum characters of document to include.
            Longer documents are truncated to avoid context limits.

    Returns:
        Formatted prompt string ready for LLM.
    """
    # Truncate document if needed
    if len(document_text) > max_doc_chars:
        document_text = document_text[:max_doc_chars] + "..."

    return RERANK_PROMPT.format(
        query=query,
        document=document_text,
    )
