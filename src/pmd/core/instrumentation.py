"""OpenTelemetry instrumentation for MLX local inference.

This module provides optional, configuration-driven instrumentation for
mlx-lm generation and mlx-embeddings calls, exporting traces compatible
with Arize Phoenix.

Usage:
    # In config or CLI, enable tracing:
    config.tracing.enabled = True
    config.tracing.phoenix_endpoint = "http://localhost:6006/v1/traces"

    # Initialize tracer early in application startup:
    tracer = configure_phoenix_tracing(config.tracing)

    # Use wrappers around MLX calls:
    with traced_mlx_generate(tracer, model_id, prompt, max_tokens, ...):
        result = mlx_lm.generate(...)
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterator, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


# =============================================================================
# Configuration
# =============================================================================


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


# =============================================================================
# Global State
# =============================================================================

_tracer: "Tracer | None" = None
_warning_logged: bool = False


def get_tracer() -> "Tracer | None":
    """Get the configured tracer, or None if tracing is disabled."""
    return _tracer


# =============================================================================
# Tracer Configuration
# =============================================================================


def configure_phoenix_tracing(config: TracingConfig) -> "Tracer | None":
    """Configure OpenTelemetry tracing for Phoenix.

    Sets up the TracerProvider with OTLP HTTP exporter targeting Phoenix.
    If tracing is disabled or setup fails, returns None.

    Args:
        config: TracingConfig with endpoint and settings.

    Returns:
        Configured Tracer instance, or None if disabled/failed.
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
        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": config.service_version,
            }
        )

        # Configure sampler
        if config.sample_rate >= 1.0:
            sampler = ALWAYS_ON
        else:
            sampler = TraceIdRatioBased(config.sample_rate)

        # Create provider
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Create OTLP exporter targeting Phoenix
        exporter = OTLPSpanExporter(endpoint=config.phoenix_endpoint)

        # Use batch processor for long-running apps (better performance)
        # Use simple processor for short-lived scripts (ensures all spans sent)
        if config.batch_export:
            processor = BatchSpanProcessor(exporter)
        else:
            processor = SimpleSpanProcessor(exporter)

        provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(provider)

        # Get tracer
        _tracer = trace.get_tracer(
            config.service_name,
            config.service_version,
        )

        logger.info(
            f"Phoenix tracing enabled: endpoint={config.phoenix_endpoint}, "
            f"sample_rate={config.sample_rate}"
        )
        _warning_logged = False
        return _tracer

    except ImportError as e:
        if not _warning_logged:
            logger.warning(
                f"OpenTelemetry packages not installed, tracing disabled: {e}. "
                "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
            )
            _warning_logged = True
        _tracer = None
        return None

    except Exception as e:
        if not _warning_logged:
            logger.warning(f"Failed to configure Phoenix tracing: {e}")
            _warning_logged = True
        _tracer = None
        return None


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


# =============================================================================
# MLX Generation Instrumentation
# =============================================================================


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
    """Context manager for tracing mlx_lm.generate calls.

    Creates a span with generation parameters and captures outcome.

    Args:
        model_id: Model identifier (e.g., "mlx-community/Qwen2.5-1.5B-Instruct-4bit").
        prompt: Input prompt text.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        tracer: Optional tracer (uses global if not provided).
        top_p: Top-p sampling parameter.
        seed: Random seed.
        stop_strings: Stop sequences.
        streaming: Whether streaming generation.

    Yields:
        Dict to store result metadata (output_length, token_count, etc.).
        Caller should populate this dict after generation completes.

    Example:
        with traced_mlx_generate("model", prompt, 256, 0.7) as result_meta:
            output = mlx_lm.generate(...)
            result_meta["output"] = output
            result_meta["output_length"] = len(output)
    """
    active_tracer = tracer or _tracer
    result_meta: dict[str, Any] = {}

    if active_tracer is None:
        # No-op when tracing disabled
        yield result_meta
        return

    from opentelemetry.trace import Status, StatusCode

    start_time = time.perf_counter()

    with active_tracer.start_as_current_span("mlx_lm.generate") as span:
        # Set input attributes
        span.set_attribute("mlx.model_id", model_id)
        span.set_attribute("mlx.device", "mps")  # MLX always uses Metal
        span.set_attribute("gen_ai.prompt.length", len(prompt))
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)
        span.set_attribute("gen_ai.request.temperature", temperature)
        span.set_attribute("gen_ai.streaming", streaming)
        # Phoenix semantic conventions for prompt/completion visibility
        span.set_attribute("input.value", prompt)
        span.set_attribute("llm.model_name", model_id)

        if top_p is not None:
            span.set_attribute("gen_ai.request.top_p", top_p)
        if seed is not None:
            span.set_attribute("gen_ai.seed", seed)
        if stop_strings:
            span.set_attribute("gen_ai.stop_sequences_count", len(stop_strings))

        try:
            yield result_meta

            # Record success and output metadata
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("mlx.latency_ms", elapsed_ms)

            if "output_length" in result_meta:
                span.set_attribute("gen_ai.response.length", result_meta["output_length"])
            if "output" in result_meta:
                span.set_attribute("output.value", result_meta["output"])

            if "token_count" in result_meta:
                span.set_attribute("gen_ai.response.token_count", result_meta["token_count"])
            else:
                span.set_attribute("gen_ai.token_count_unknown", True)
                # Estimate tokens from chars (rough: 1 token ~ 4 chars)
                if "output_length" in result_meta:
                    span.set_attribute(
                        "gen_ai.response.estimated_tokens",
                        result_meta["output_length"] // 4,
                    )

            if result_meta.get("success", True):
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, result_meta.get("error", "Unknown error")))

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("mlx.latency_ms", elapsed_ms)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


# =============================================================================
# MLX Embedding Instrumentation
# =============================================================================


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
    """Context manager for tracing mlx_embeddings.generate calls.

    Creates a span with embedding parameters and captures outcome.

    Args:
        model_id: Embedding model identifier.
        input_text: Text to embed.
        tracer: Optional tracer (uses global if not provided).
        is_query: Whether this is a query embedding (vs document).
        batch_size: Number of texts being embedded.
        pooling_strategy: Pooling strategy used (mean, cls, etc.).

    Yields:
        Dict to store result metadata (embedding_dim, etc.).

    Example:
        with traced_mlx_embed("model", text, is_query=True) as result_meta:
            embedding = mlx_embeddings.generate(...)
            result_meta["embedding_dim"] = len(embedding)
    """
    active_tracer = tracer or _tracer
    result_meta: dict[str, Any] = {}

    if active_tracer is None:
        yield result_meta
        return

    from opentelemetry.trace import Status, StatusCode

    start_time = time.perf_counter()

    with active_tracer.start_as_current_span("mlx_embedding.embed") as span:
        # Set input attributes
        span.set_attribute("mlx.model_id", model_id)
        span.set_attribute("mlx.device", "mps")
        span.set_attribute("embedding.input_length", len(input_text))
        span.set_attribute("embedding.is_query", is_query)
        span.set_attribute("embedding.batch_size", batch_size)
        # Phoenix semantic conventions for input visibility
        span.set_attribute("input.value", input_text)

        if pooling_strategy:
            span.set_attribute("embedding.pooling_strategy", pooling_strategy)

        try:
            yield result_meta

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("mlx.latency_ms", elapsed_ms)

            if "embedding_dim" in result_meta:
                span.set_attribute("embedding.dimension", result_meta["embedding_dim"])

            if "pooling_used" in result_meta:
                span.set_attribute("embedding.pooling_used", result_meta["pooling_used"])

            if result_meta.get("success", True):
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, result_meta.get("error", "Unknown error")))

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("mlx.latency_ms", elapsed_ms)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


# =============================================================================
# Streaming Generator Wrapper
# =============================================================================

T = TypeVar("T")


def traced_streaming_generate(
    generator: Iterator[T],
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    tracer: "Tracer | None" = None,
) -> Iterator[T]:
    """Wrap a streaming generator to trace the full generation.

    Measures time from first to last token and counts total output.

    Args:
        generator: Token/string iterator from mlx_lm streaming.
        model_id: Model identifier.
        prompt: Input prompt.
        max_tokens: Max tokens requested.
        temperature: Temperature used.
        tracer: Optional tracer.

    Yields:
        Items from the generator unchanged.
    """
    active_tracer = tracer or _tracer

    if active_tracer is None:
        yield from generator
        return

    from opentelemetry.trace import Status, StatusCode

    start_time = time.perf_counter()
    total_output = []
    token_count = 0

    span = active_tracer.start_span("mlx_lm.generate")
    span.set_attribute("mlx.model_id", model_id)
    span.set_attribute("mlx.device", "mps")
    span.set_attribute("gen_ai.prompt.length", len(prompt))
    span.set_attribute("gen_ai.request.max_tokens", max_tokens)
    span.set_attribute("gen_ai.request.temperature", temperature)
    span.set_attribute("gen_ai.streaming", True)
    # Phoenix semantic conventions
    span.set_attribute("input.value", prompt)
    span.set_attribute("llm.model_name", model_id)

    try:
        for item in generator:
            token_count += 1
            if isinstance(item, str):
                total_output.append(item)
            yield item

        # Generation complete
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        output_str = "".join(total_output) if total_output else ""

        span.set_attribute("mlx.latency_ms", elapsed_ms)
        span.set_attribute("gen_ai.response.length", len(output_str))
        span.set_attribute("gen_ai.response.token_count", token_count)
        span.set_attribute("output.value", output_str)
        span.set_status(Status(StatusCode.OK))

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        span.set_attribute("mlx.latency_ms", elapsed_ms)
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise

    finally:
        span.end()


# =============================================================================
# High-level Request Tracing
# =============================================================================


@contextmanager
def traced_request(
    operation: str,
    *,
    tracer: "Tracer | None" = None,
    attributes: dict[str, Any] | None = None,
) -> Generator["Span | None", None, None]:
    """Create a parent span for a high-level operation.

    Nested mlx_lm and embedding spans will appear under this parent.

    Args:
        operation: Operation name (e.g., "search", "index", "rerank").
        tracer: Optional tracer.
        attributes: Additional span attributes.

    Yields:
        The span (or None if tracing disabled).

    Example:
        with traced_request("hybrid_search", attributes={"query": query}) as span:
            # MLX calls here will be nested under this span
            embeddings = await provider.embed(query)
            results = search(embeddings)
    """
    active_tracer = tracer or _tracer

    if active_tracer is None:
        yield None
        return

    from opentelemetry.trace import Status, StatusCode

    with active_tracer.start_as_current_span(f"pmd.{operation}") as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
