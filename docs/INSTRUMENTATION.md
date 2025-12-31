# Instrumentation and Monitoring

This document provides technical documentation on PMD's dynamic instrumentation implementation using OpenTelemetry and Arize Phoenix for LLM observability.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Configuration](#3-configuration)
4. [Core Abstractions](#4-core-abstractions)
5. [Traced Operations](#5-traced-operations)
6. [Integration Points](#6-integration-points)
7. [Span Attributes](#7-span-attributes)
8. [Usage Examples](#8-usage-examples)

---

## 1. Overview

PMD implements optional, configuration-driven instrumentation for MLX local inference operations. Traces are exported via OpenTelemetry Protocol (OTLP) to Arize Phoenix, providing:

- **LLM call visibility** — View prompts, completions, and generation parameters
- **Embedding instrumentation** — Track embedding generation with model and pooling details
- **Latency monitoring** — Measure operation timing for performance analysis
- **Error tracking** — Capture exceptions with full context

**Key Design Decisions:**

| Decision | Rationale |
|----------|-----------|
| Optional tracing | No runtime overhead when disabled |
| Lazy imports | OpenTelemetry packages only loaded when tracing enabled |
| Global tracer | Single tracer instance shared across modules |
| Context managers | Clean integration with existing code, automatic cleanup |
| Phoenix semantic conventions | Native integration with Phoenix LLM observability UI |

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
├─────────────────────────────────────────────────────────────────┤
│  CLI (main.py)                                                   │
│    └─ configure_phoenix_tracing(config.tracing)                 │
├─────────────────────────────────────────────────────────────────┤
│  MLX Provider (mlx_provider.py)                                 │
│    ├─ traced_mlx_embed() → Embedding operations                 │
│    └─ traced_mlx_generate() → Text generation                   │
├─────────────────────────────────────────────────────────────────┤
│  Instrumentation Module (core/instrumentation.py)               │
│    ├─ TracingConfig           Configuration dataclass           │
│    ├─ configure_phoenix_tracing()  Setup TracerProvider         │
│    ├─ traced_mlx_generate()   Context manager for generation    │
│    ├─ traced_mlx_embed()      Context manager for embeddings    │
│    ├─ traced_request()        High-level operation spans        │
│    └─ traced_streaming_generate()  Iterator wrapper             │
├─────────────────────────────────────────────────────────────────┤
│  OpenTelemetry SDK                                              │
│    ├─ TracerProvider          Creates and manages tracers       │
│    ├─ BatchSpanProcessor      Efficient span export             │
│    └─ OTLPSpanExporter        OTLP HTTP export to Phoenix       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Arize Phoenix  │
                    │   (localhost:   │
                    │     6006)       │
                    └─────────────────┘
```

### 2.2 Data Flow

```
1. CLI parses --phoenix-tracing flag
2. config.tracing.enabled = True
3. configure_phoenix_tracing() creates TracerProvider
4. Global _tracer set for module access
5. MLX operations use traced_* context managers
6. Spans created with input/output attributes
7. BatchSpanProcessor queues spans
8. OTLPSpanExporter sends to Phoenix
```

---

## 3. Configuration

### 3.1 TracingConfig Dataclass

**File:** `src/pmd/core/instrumentation.py:40-57`

```python
@dataclass
class TracingConfig:
    """Phoenix/OpenTelemetry tracing configuration.

    Attributes:
        enabled: Whether tracing is enabled (default: False).
        phoenix_endpoint: OTLP endpoint for Phoenix (default: local Phoenix).
        service_name: Service name for traces (default: pmd).
        service_version: Service version for traces.
        sample_rate: Sampling rate 0.0-1.0 (default: 1.0 = all traces).
        batch_export: Use BatchSpanProcessor vs SimpleSpanProcessor.
    """

    enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    service_name: str = "pmd"
    service_version: str = "1.0.0"
    sample_rate: float = 1.0
    batch_export: bool = True
```

### 3.2 Configuration Methods

**CLI Flags:**
```bash
pmd --phoenix-tracing search "query"
pmd --phoenix-tracing --phoenix-endpoint http://remote:6006/v1/traces search "query"
```

**TOML Configuration:**
```toml
[tracing]
enabled = true
phoenix_endpoint = "http://localhost:6006/v1/traces"
service_name = "pmd"
service_version = "1.0.0"
sample_rate = 1.0
batch_export = true
```

**Environment Variables:**
```bash
export PHOENIX_TRACING=1
export PHOENIX_ENDPOINT="http://localhost:6006/v1/traces"
export PHOENIX_SAMPLE_RATE=0.5
```

### 3.3 Precedence

1. CLI flags (`--phoenix-tracing`, `--phoenix-endpoint`)
2. Environment variables
3. TOML configuration file
4. Built-in defaults

---

## 4. Core Abstractions

### 4.1 Global Tracer State

**File:** `src/pmd/core/instrumentation.py:64-70`

```python
_tracer: "Tracer | None" = None
_warning_logged: bool = False

def get_tracer() -> "Tracer | None":
    """Get the configured tracer, or None if tracing is disabled."""
    return _tracer
```

The module maintains global state for the tracer instance. This allows traced operations throughout the codebase to access the tracer without parameter passing.

### 4.2 Tracer Configuration

**File:** `src/pmd/core/instrumentation.py:78-173`

```python
def configure_phoenix_tracing(config: TracingConfig) -> "Tracer | None":
    """Configure OpenTelemetry tracing for Phoenix.

    Sets up the TracerProvider with OTLP HTTP exporter targeting Phoenix.
    If tracing is disabled or setup fails, returns None.
    """
    global _tracer, _warning_logged

    if not config.enabled:
        logger.debug("Phoenix tracing is disabled")
        _tracer = None
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.sampling import (
            TraceIdRatioBased,
            ALWAYS_ON,
        )

        # Create resource with service info
        resource = Resource.create({
            "service.name": config.service_name,
            "service.version": config.service_version,
        })

        # Configure sampler
        sampler = ALWAYS_ON if config.sample_rate >= 1.0 else TraceIdRatioBased(config.sample_rate)

        # Create provider and exporter
        provider = TracerProvider(resource=resource, sampler=sampler)
        exporter = OTLPSpanExporter(endpoint=config.phoenix_endpoint)

        # Use batch processor for performance (long-running apps)
        # Use simple processor for scripts (ensures all spans sent)
        processor = BatchSpanProcessor(exporter) if config.batch_export else SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(config.service_name, config.service_version)

        logger.info(f"Phoenix tracing enabled: endpoint={config.phoenix_endpoint}")
        return _tracer

    except ImportError as e:
        if not _warning_logged:
            logger.warning(f"OpenTelemetry packages not installed: {e}")
            _warning_logged = True
        return None
```

**Design Pattern:** Lazy Import with Graceful Degradation

- OpenTelemetry packages only imported when tracing is enabled
- Missing packages logged once, then silently ignored
- Application continues without instrumentation if setup fails

### 4.3 Shutdown

**File:** `src/pmd/core/instrumentation.py:176-193`

```python
def shutdown_tracing() -> None:
    """Shutdown tracing and flush any pending spans."""
    global _tracer

    if _tracer is None:
        return

    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        logger.debug("Phoenix tracing shutdown complete")
    except Exception as e:
        logger.warning(f"Error shutting down tracing: {e}")

    _tracer = None
```

---

## 5. Traced Operations

### 5.1 traced_mlx_generate

Context manager for tracing MLX text generation.

**File:** `src/pmd/core/instrumentation.py:202-303`

```python
@contextmanager
def traced_mlx_generate(
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    tracer: "Tracer | None" = None,
    top_p: float | None = None,
    seed: int | None = None,
    stop_strings: list[str] | None = None,
    streaming: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for tracing mlx_lm.generate calls."""
```

**Span Name:** `mlx_lm.generate`

**Input Attributes:**
- `mlx.model_id` — Model identifier
- `mlx.device` — Always "mps" (Metal Performance Shaders)
- `gen_ai.prompt.length` — Prompt character count
- `gen_ai.request.max_tokens` — Maximum tokens requested
- `gen_ai.request.temperature` — Sampling temperature
- `gen_ai.streaming` — Whether streaming mode
- `input.value` — Full prompt text (Phoenix convention)
- `llm.model_name` — Model name (Phoenix convention)

**Output Attributes:**
- `mlx.latency_ms` — Operation duration in milliseconds
- `gen_ai.response.length` — Response character count
- `gen_ai.response.token_count` — Token count (if available)
- `output.value` — Generated text (Phoenix convention)

### 5.2 traced_mlx_embed

Context manager for tracing MLX embedding generation.

**File:** `src/pmd/core/instrumentation.py:312-387`

```python
@contextmanager
def traced_mlx_embed(
    model_id: str,
    input_text: str,
    *,
    tracer: "Tracer | None" = None,
    is_query: bool = False,
    batch_size: int = 1,
    pooling_strategy: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for tracing mlx_embeddings.generate calls."""
```

**Span Name:** `mlx_embedding.embed`

**Input Attributes:**
- `mlx.model_id` — Embedding model identifier
- `mlx.device` — Always "mps"
- `embedding.input_length` — Input text character count
- `embedding.is_query` — Query vs document embedding
- `embedding.batch_size` — Batch size
- `input.value` — Input text (Phoenix convention)

**Output Attributes:**
- `mlx.latency_ms` — Operation duration
- `embedding.dimension` — Vector dimension (e.g., 768)
- `embedding.pooling_used` — Pooling strategy (text_embeds, pooler_output, mean_last_hidden_state)

### 5.3 traced_request

High-level operation span for grouping related calls.

**File:** `src/pmd/core/instrumentation.py:478-522`

```python
@contextmanager
def traced_request(
    operation: str,
    *,
    tracer: "Tracer | None" = None,
    attributes: dict[str, Any] | None = None,
) -> Generator["Span | None", None, None]:
    """Create a parent span for a high-level operation.

    Nested mlx_lm and embedding spans will appear under this parent.
    """
```

**Span Name:** `pmd.{operation}` (e.g., `pmd.hybrid_search`, `pmd.index`)

**Use Case:** Grouping multiple MLX calls under a single logical operation for hierarchical trace visualization.

### 5.4 traced_streaming_generate

Iterator wrapper for tracing streaming generation.

**File:** `src/pmd/core/instrumentation.py:397-469`

```python
def traced_streaming_generate(
    generator: Iterator[T],
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    tracer: "Tracer | None" = None,
) -> Iterator[T]:
    """Wrap a streaming generator to trace the full generation."""
```

Measures time from first to last token and accumulates output for span attributes.

---

## 6. Integration Points

### 6.1 CLI Initialization

**File:** `src/pmd/cli/main.py:169-178`

```python
# Apply CLI tracing overrides (CLI takes precedence over config/env)
if args.phoenix_tracing:
    config.tracing.enabled = True
if args.phoenix_endpoint:
    config.tracing.phoenix_endpoint = args.phoenix_endpoint

# Initialize Phoenix tracing if enabled
if config.tracing.enabled:
    from ..core.instrumentation import configure_phoenix_tracing
    configure_phoenix_tracing(config.tracing)
```

Tracing is configured once at CLI startup before any operations execute.

### 6.2 MLX Provider - Embeddings

**File:** `src/pmd/llm/mlx_provider.py:152-236`

```python
async def embed(self, text: str, model: str | None = None, is_query: bool = False) -> EmbeddingResult | None:
    from ..core.instrumentation import traced_mlx_embed

    with traced_mlx_embed(
        model_id=self.config.embedding_model,
        input_text=text,
        is_query=is_query,
        batch_size=1,
    ) as trace_meta:
        try:
            # ... embedding generation ...

            # Record trace metadata
            trace_meta["success"] = True
            trace_meta["embedding_dim"] = len(embedding)
            trace_meta["pooling_used"] = pooling_used

            return EmbeddingResult(embedding=embedding, model=self.config.embedding_model)

        except Exception as e:
            trace_meta["success"] = False
            trace_meta["error"] = str(e)
            return None
```

### 6.3 MLX Provider - Generation

**File:** `src/pmd/llm/mlx_provider.py:256-318`

```python
async def generate(self, prompt: str, model: str | None = None, max_tokens: int = 256, temperature: float = 0.7) -> str | None:
    from ..core.instrumentation import traced_mlx_generate

    with traced_mlx_generate(
        model_id=self.config.model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=False,
    ) as trace_meta:
        try:
            # ... text generation ...

            # Record trace metadata
            trace_meta["success"] = result is not None
            if result:
                trace_meta["output_length"] = len(result)
                trace_meta["output"] = result

            return result

        except Exception as e:
            trace_meta["success"] = False
            trace_meta["error"] = str(e)
            return None
```

---

## 7. Span Attributes

### 7.1 MLX Generation Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mlx.model_id` | string | Model identifier (e.g., "mlx-community/Qwen2.5-1.5B-Instruct-4bit") |
| `mlx.device` | string | Always "mps" (Metal Performance Shaders) |
| `mlx.latency_ms` | float | Operation duration in milliseconds |
| `gen_ai.prompt.length` | int | Input prompt character count |
| `gen_ai.request.max_tokens` | int | Maximum tokens requested |
| `gen_ai.request.temperature` | float | Sampling temperature |
| `gen_ai.request.top_p` | float | Top-p sampling (if set) |
| `gen_ai.seed` | int | Random seed (if set) |
| `gen_ai.streaming` | bool | Streaming mode flag |
| `gen_ai.response.length` | int | Response character count |
| `gen_ai.response.token_count` | int | Actual token count |
| `gen_ai.response.estimated_tokens` | int | Estimated tokens (chars/4) |
| `input.value` | string | Full prompt (Phoenix UI) |
| `output.value` | string | Full response (Phoenix UI) |
| `llm.model_name` | string | Model name (Phoenix UI) |

### 7.2 MLX Embedding Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mlx.model_id` | string | Embedding model identifier |
| `mlx.device` | string | Always "mps" |
| `mlx.latency_ms` | float | Operation duration |
| `embedding.input_length` | int | Input text character count |
| `embedding.is_query` | bool | Query vs document embedding |
| `embedding.batch_size` | int | Batch size |
| `embedding.dimension` | int | Vector dimension (e.g., 768) |
| `embedding.pooling_strategy` | string | Configured pooling |
| `embedding.pooling_used` | string | Actual pooling used |
| `input.value` | string | Input text (Phoenix UI) |

### 7.3 Phoenix Semantic Conventions

The instrumentation follows Phoenix's semantic conventions for LLM observability:

- `input.value` — Makes prompts visible in Phoenix trace viewer
- `output.value` — Makes completions visible in Phoenix trace viewer
- `llm.model_name` — Enables model filtering in Phoenix UI

---

## 8. Usage Examples

### 8.1 Basic CLI Usage

```bash
# Start Phoenix server
phoenix serve

# Run search with tracing
pmd --phoenix-tracing search "machine learning"

# View traces at http://localhost:6006
```

### 8.2 Custom Endpoint

```bash
# Export to remote Phoenix instance
pmd --phoenix-tracing --phoenix-endpoint http://phoenix.example.com:6006/v1/traces search "query"
```

### 8.3 Programmatic Usage

```python
from pmd.core.instrumentation import (
    configure_phoenix_tracing,
    traced_mlx_generate,
    traced_mlx_embed,
    traced_request,
    shutdown_tracing,
    TracingConfig,
)

# Configure tracing
config = TracingConfig(
    enabled=True,
    phoenix_endpoint="http://localhost:6006/v1/traces",
    sample_rate=1.0,
)
tracer = configure_phoenix_tracing(config)

# Trace a high-level operation with nested calls
with traced_request("hybrid_search", attributes={"query": "test"}) as span:
    # Trace embedding generation
    with traced_mlx_embed("model-id", "search text", is_query=True) as embed_meta:
        embedding = generate_embedding(...)
        embed_meta["embedding_dim"] = len(embedding)

    # Trace text generation
    with traced_mlx_generate("model-id", prompt, 256, 0.7) as gen_meta:
        output = generate_text(...)
        gen_meta["output"] = output
        gen_meta["output_length"] = len(output)

# Cleanup
shutdown_tracing()
```

### 8.4 Sampling Configuration

```python
# Sample 50% of traces (for high-volume production)
config = TracingConfig(
    enabled=True,
    sample_rate=0.5,  # 50% sampling
)
```

### 8.5 Simple Export for Scripts

```python
# Use SimpleSpanProcessor for short-lived scripts
# (ensures all spans are sent before exit)
config = TracingConfig(
    enabled=True,
    batch_export=False,  # Use SimpleSpanProcessor
)
```

---

## File Reference

| Component | File | Lines |
|-----------|------|-------|
| TracingConfig | `src/pmd/core/instrumentation.py` | 40-57 |
| configure_phoenix_tracing | `src/pmd/core/instrumentation.py` | 78-173 |
| shutdown_tracing | `src/pmd/core/instrumentation.py` | 176-193 |
| traced_mlx_generate | `src/pmd/core/instrumentation.py` | 202-303 |
| traced_mlx_embed | `src/pmd/core/instrumentation.py` | 312-387 |
| traced_streaming_generate | `src/pmd/core/instrumentation.py` | 397-469 |
| traced_request | `src/pmd/core/instrumentation.py` | 478-522 |
| CLI integration | `src/pmd/cli/main.py` | 169-178 |
| MLX embed integration | `src/pmd/llm/mlx_provider.py` | 152-236 |
| MLX generate integration | `src/pmd/llm/mlx_provider.py` | 256-318 |
