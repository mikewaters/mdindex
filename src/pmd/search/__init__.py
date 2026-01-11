"""Search orchestration for PMD."""

from .dspy_modules import (
    DSPyQueryExpander,
    DSPyReranker,
    DocumentRerankerAdapter,
    QueryExpanderAdapter,
    create_query_expander,
    create_reranker,
)
from .dspy_retriever import (
    PMDRetriever,
    PMDRetrieverAsync,
    RetrievedPassage,
    create_retriever,
)
from .dspy_rag import (
    ChatAgent,
    ChatMessage,
    ChatResponse,
    RAGModule,
    create_chat_agent,
    create_rag_module,
)

__all__ = [
    # DSPy modules
    "DSPyQueryExpander",
    "DSPyReranker",
    "DocumentRerankerAdapter",
    "QueryExpanderAdapter",
    "create_query_expander",
    "create_reranker",
    # DSPy retriever
    "PMDRetriever",
    "PMDRetrieverAsync",
    "RetrievedPassage",
    "create_retriever",
    # DSPy RAG
    "ChatAgent",
    "ChatMessage",
    "ChatResponse",
    "RAGModule",
    "create_chat_agent",
    "create_rag_module",
]
