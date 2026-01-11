# Hybrid Architecture Plan: PMD + DSPy + LlamaIndex

Based on the [Migration Analysis](./LLAMAINDEX_MIGRATION_ANALYSIS.md) and the decision to prioritize the local MLX experience, this document outlines the roadmap.

**Core Philosophy:**
*   **MLX (Local):** The primary, first-class engine for local inference on Apple Silicon.
*   **LiteLLM (Remote):** The universal adapter for all API-based providers (OpenRouter, OpenAI, etc.), replacing specific API clients.
*   **DSPy:** The "intelligent" logic layer (Query Expansion, Reranking, Agents), capable of running on *either* MLX or LiteLLM.
*   **LlamaIndex:** The "plumbing" for Data Loading and Ingestion.
*   **PMD:** The custom storage and retrieval engine.

---

## Proposal 1: Standardized LLM Client (Coexistence)

**Goal:** Introduce `LiteLLMProvider` to replace `OpenRouterProvider` and `LMStudioProvider`, while preserving the optimized `MLXProvider`.

### Architecture Change
1.  **Add `LiteLLMProvider`:** A new implementation of `LLMProvider` in `src/pmd/llm/litellm_provider.py`.
2.  **Update Factory:** `src/pmd/llm/factory.py` will return `MLXProvider` for "mlx" config, and `LiteLLMProvider` for everything else (configured via LiteLLM's standard env vars or config).

#### LiteLLM Provider (Concept)
```python
import litellm
from pmd.llm.base import LLMProvider, EmbeddingResult

class LiteLLMProvider(LLMProvider):
    def __init__(self, config):
        self.model = config.model
        self.api_base = config.api_base
        # LiteLLM handles the rest

    async def generate(self, prompt: str, ...) -> str:
        resp = await litellm.acompletion(model=self.model, messages=[...], api_base=self.api_base)
        return resp.choices[0].message.content
```

---

## Proposal 2: DSPy for Search Logic

**Goal:** Replace manual prompt strings in `QueryExpander` and `DocumentReranker` with DSPy Modules.

### Key Requirement: MLX Support
We must create a `DSPyMLXClient` that wraps our existing `MLXProvider` so DSPy can drive the local model.

#### 1. DSPy LM Wrapper
File: `src/pmd/llm/dspy_client.py`
```python
import dspy

class PMDClient(dspy.LM):
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        
    def __call__(self, prompt, **kwargs):
        # Delegate to the active PMD provider (MLX or LiteLLM)
        return self.provider.generate(prompt, **kwargs)
```

#### 2. Query Expansion Module
File: `src/pmd/search/dspy_modules.py`
```python
import dspy

class QueryExpansionSignature(dspy.Signature):
    """Generate diverse search query variations."""
    query = dspy.InputField()
    variations = dspy.OutputField(format=list)

class DSPyQueryExpander:
    def __init__(self, provider: LLMProvider):
        # Configure DSPy to use our provider
        lm = PMDClient(provider)
        dspy.settings.configure(lm=lm)
        self.predict = dspy.ChainOfThought(QueryExpansionSignature)
```

### Benefits
*   **Unified Logic:** The same DSPy module (prompt/reasoning logic) works on both local MLX and remote APIs.
*   **Optimization:** We can optimize the DSPy modules specifically for the small local models (e.g. Llama 3 8B).

---

## Proposal 3: PMD as a Retrieval Backend for DSPy

**Goal:** Connect `HybridSearchPipeline` to DSPy.

#### 1. DSPy Retriever
```python
# src/pmd/search/dspy_retriever.py
class PMDRetriever(dspy.Retrieve):
    def __init__(self, pipeline: HybridSearchPipeline, k=5):
        self.pipeline = pipeline
        self.k = k

    def forward(self, query: str, k: int = None):
        results = await self.pipeline.search(query, limit=k or self.k)
        return [res.body for res in results]
```

---

## Proposal 4: Universal Data Source (LlamaIndex)

**Goal:** Use LlamaIndex Loaders to ingest data into PMD SQLite.

### Architecture Change
Enhance `src/pmd/sources/content/llamaindex.py` to allow configuration of any LlamaHub loader, which feeds into the standard PMD ingestion pipeline.

---

## Proposal 5: DSPy Chat Agent

**Goal:** "Chat" command using DSPy RAG module.

```python
class RAGModule(dspy.Module):
    def __init__(self, pipeline):
        self.retrieve = PMDRetriever(pipeline)
        self.generate = dspy.ChainOfThought(RAGSignature)
```

---

## Summary

1.  **LLM Layer:** **MLX** (Local) || **LiteLLM** (Remote).
2.  **Logic Layer:** **DSPy** (configured with the active LLM Layer).
3.  **Storage Layer:** **PMD SQLite**.
4.  **Ingest Layer:** **LlamaIndex** -> PMD SQLite.

This plan respects the local-first design while adding the power of DSPy and universal API support.