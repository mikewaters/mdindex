"""DSPy RAG (Retrieval-Augmented Generation) module for chat.

This module provides a DSPy-based RAG implementation that uses PMD's
hybrid search for context retrieval and generates responses using
the configured LLM provider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dspy
from loguru import logger

if TYPE_CHECKING:
    from pmd.search.pipeline import HybridSearchPipeline
    from pmd.llm import LLMProvider


# =============================================================================
# Signatures
# =============================================================================


class RAGSignature(dspy.Signature):
    """Generate an answer based on context and question.

    Given relevant context passages from a knowledge base and a user question,
    generate a helpful, accurate answer based on the provided context.
    """

    context: str = dspy.InputField(
        desc="Relevant passages from the knowledge base"
    )
    question: str = dspy.InputField(
        desc="The user's question to answer"
    )
    answer: str = dspy.OutputField(
        desc="A helpful answer based on the context"
    )


class ConversationalRAGSignature(dspy.Signature):
    """Generate a conversational answer with history context.

    Given conversation history, relevant passages, and a new question,
    generate a contextually appropriate response.
    """

    history: str = dspy.InputField(
        desc="Previous conversation turns (user/assistant exchanges)"
    )
    context: str = dspy.InputField(
        desc="Relevant passages from the knowledge base"
    )
    question: str = dspy.InputField(
        desc="The user's current question"
    )
    answer: str = dspy.OutputField(
        desc="A helpful, conversational response"
    )


# =============================================================================
# RAG Module
# =============================================================================


class RAGModule(dspy.Module):
    """DSPy RAG module for question answering.

    Retrieves relevant context from PMD's search pipeline and generates
    answers using chain-of-thought reasoning.

    Example:
        from pmd.search.dspy_rag import RAGModule
        from pmd.search.pipeline import HybridSearchPipeline

        pipeline = HybridSearchPipeline(...)
        rag = RAGModule(pipeline, k=5)

        result = rag("What is the purpose of this project?")
        print(result.answer)
    """

    def __init__(
        self,
        pipeline: "HybridSearchPipeline",
        k: int = 5,
        use_vector: bool = True,
    ):
        """Initialize RAG module.

        Args:
            pipeline: PMD search pipeline for retrieval.
            k: Number of passages to retrieve (default: 5).
            use_vector: Whether to use vector search (default: True).
        """
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self._use_vector = use_vector
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question: str) -> dspy.Prediction:
        """Answer a question using RAG.

        Args:
            question: The user's question.

        Returns:
            Prediction with 'answer' and 'context' fields.
        """
        # Retrieve relevant context
        context = self._retrieve(question)

        # Generate answer
        result = self.generate(context=context, question=question)

        return dspy.Prediction(
            answer=result.answer,
            context=context,
        )

    def _retrieve(self, query: str) -> str:
        """Retrieve context passages synchronously.

        Args:
            query: Search query.

        Returns:
            Concatenated context string.
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._async_retrieve(query),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self._async_retrieve(query))

    async def _async_retrieve(self, query: str) -> str:
        """Retrieve context passages asynchronously.

        Args:
            query: Search query.

        Returns:
            Concatenated context string.
        """
        try:
            results = await self._pipeline.search(
                query=query,
                limit=self._k,
                use_expansion=False,
                use_vector=self._use_vector,
                use_rerank=False,
            )

            if not results:
                return "No relevant documents found."

            # Format context with source attribution
            passages = []
            for i, r in enumerate(results, 1):
                source = r.title or r.file
                passages.append(f"[{i}] {source}:\n{r.body}")

            return "\n\n".join(passages)

        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return "Unable to retrieve relevant context."


# =============================================================================
# Chat Agent
# =============================================================================


@dataclass
class ChatMessage:
    """A message in the chat history."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from the chat agent."""

    answer: str
    context: str
    sources: list[str]


class ChatAgent:
    """Conversational RAG agent for interactive chat.

    Maintains conversation history and uses RAG for context-aware responses.

    Example:
        from pmd.search.dspy_rag import ChatAgent
        from pmd.llm import create_llm_provider, create_dspy_client
        from pmd.search.pipeline import HybridSearchPipeline

        provider = create_llm_provider(config)
        lm = create_dspy_client(provider)
        pipeline = HybridSearchPipeline(...)

        agent = ChatAgent(pipeline, lm)

        response = agent.chat("What is PMD?")
        print(response.answer)

        response = agent.chat("How do I install it?")
        print(response.answer)
    """

    def __init__(
        self,
        pipeline: "HybridSearchPipeline",
        lm: dspy.LM,
        k: int = 5,
        max_history: int = 10,
    ):
        """Initialize chat agent.

        Args:
            pipeline: PMD search pipeline for retrieval.
            lm: DSPy language model.
            k: Number of passages to retrieve (default: 5).
            max_history: Maximum conversation turns to keep (default: 10).
        """
        self._pipeline = pipeline
        self._lm = lm
        self._k = k
        self._max_history = max_history
        self._history: list[ChatMessage] = []
        self._rag = RAGModule(pipeline, k=k)

    @property
    def history(self) -> list[ChatMessage]:
        """Get conversation history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def chat(self, message: str) -> ChatResponse:
        """Send a message and get a response.

        Args:
            message: User message.

        Returns:
            ChatResponse with answer, context, and sources.
        """
        # Add user message to history
        self._history.append(ChatMessage(role="user", content=message))

        # Generate response using RAG
        with dspy.context(lm=self._lm):
            result = self._rag(message)

        answer = result.answer if hasattr(result, "answer") else str(result)
        context = result.context if hasattr(result, "context") else ""

        # Extract sources from context
        sources = self._extract_sources(context)

        # Add assistant response to history
        self._history.append(ChatMessage(role="assistant", content=answer))

        # Trim history if needed
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

        return ChatResponse(
            answer=answer,
            context=context,
            sources=sources,
        )

    async def achat(self, message: str) -> ChatResponse:
        """Async version of chat.

        Args:
            message: User message.

        Returns:
            ChatResponse with answer, context, and sources.
        """
        # For now, run sync version in executor
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                self.chat,
                message,
            )

    def _extract_sources(self, context: str) -> list[str]:
        """Extract source references from context.

        Args:
            context: Retrieved context string.

        Returns:
            List of source file/title references.
        """
        sources = []
        for line in context.split("\n"):
            if line.startswith("[") and "]" in line:
                # Extract source name after bracket
                bracket_end = line.index("]")
                if ":" in line[bracket_end:]:
                    source = line[bracket_end + 1:line.index(":", bracket_end)].strip()
                    if source and source not in sources:
                        sources.append(source)
        return sources

    def format_history(self) -> str:
        """Format conversation history as a string.

        Returns:
            Formatted history string.
        """
        lines = []
        for msg in self._history:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            lines.append(f"{prefix} {msg.content}")
        return "\n".join(lines)


# =============================================================================
# Factory Functions
# =============================================================================


def create_rag_module(
    pipeline: "HybridSearchPipeline",
    k: int = 5,
    use_vector: bool = True,
) -> RAGModule:
    """Create a RAG module.

    Args:
        pipeline: PMD search pipeline.
        k: Number of passages to retrieve.
        use_vector: Whether to use vector search.

    Returns:
        Configured RAGModule.
    """
    return RAGModule(pipeline, k=k, use_vector=use_vector)


def create_chat_agent(
    pipeline: "HybridSearchPipeline",
    provider: "LLMProvider",
    k: int = 5,
    max_history: int = 10,
) -> ChatAgent:
    """Create a chat agent.

    Args:
        pipeline: PMD search pipeline.
        provider: LLM provider for generation.
        k: Number of passages to retrieve.
        max_history: Maximum conversation turns to keep.

    Returns:
        Configured ChatAgent.
    """
    from pmd.llm import create_dspy_client

    lm = create_dspy_client(provider)
    return ChatAgent(pipeline, lm, k=k, max_history=max_history)
