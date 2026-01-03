"""Tests for Phoenix/OpenTelemetry instrumentation."""

import pytest
from unittest.mock import MagicMock, patch

from pmd.core.instrumentation import (
    TracingConfig,
    configure_phoenix_tracing,
    shutdown_tracing,
    traced_mlx_generate,
    traced_mlx_embed,
    traced_request,
    traced_streaming_generate,
    get_tracer,
)

# Check if OpenTelemetry is available
try:
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

requires_otel = pytest.mark.skipif(not HAS_OTEL, reason="OpenTelemetry not installed")


class TestTracingConfig:
    """Tests for TracingConfig dataclass."""

    def test_defaults(self):
        """TracingConfig has correct defaults."""
        config = TracingConfig()
        assert config.enabled is False
        assert config.phoenix_endpoint == "http://localhost:6006/v1/traces"
        assert config.service_name == "pmd"
        assert config.sample_rate == 1.0
        assert config.batch_export is True

    def test_custom_values(self):
        """TracingConfig accepts custom values."""
        config = TracingConfig(
            enabled=True,
            phoenix_endpoint="http://custom:4318/v1/traces",
            sample_rate=0.5,
        )
        assert config.enabled is True
        assert config.phoenix_endpoint == "http://custom:4318/v1/traces"
        assert config.sample_rate == 0.5


class TestDisabledTracing:
    """Tests for tracing when disabled."""

    def test_configure_disabled_returns_none(self):
        """configure_phoenix_tracing returns None when disabled."""
        config = TracingConfig(enabled=False)
        tracer = configure_phoenix_tracing(config)
        assert tracer is None

    def test_get_tracer_returns_none_when_disabled(self):
        """get_tracer returns None when tracing not configured."""
        # Ensure clean state
        shutdown_tracing()
        config = TracingConfig(enabled=False)
        configure_phoenix_tracing(config)
        assert get_tracer() is None

    def test_traced_generate_no_op_when_disabled(self):
        """traced_mlx_generate is no-op when disabled."""
        shutdown_tracing()

        result_data = None
        with traced_mlx_generate(
            model_id="test-model",
            prompt="test prompt",
            max_tokens=100,
            temperature=0.7,
        ) as meta:
            meta["output_length"] = 50
            result_data = meta

        # Should complete without error
        assert result_data["output_length"] == 50

    def test_traced_embed_no_op_when_disabled(self):
        """traced_mlx_embed is no-op when disabled."""
        shutdown_tracing()

        with traced_mlx_embed(
            model_id="test-embed-model",
            input_text="test text",
            is_query=True,
        ) as meta:
            meta["embedding_dim"] = 768
            # Should complete without error
            assert meta["embedding_dim"] == 768

    def test_traced_request_no_op_when_disabled(self):
        """traced_request is no-op when disabled."""
        shutdown_tracing()

        with traced_request("test_operation") as span:
            assert span is None

    def test_traced_streaming_passthrough_when_disabled(self):
        """traced_streaming_generate passes through when disabled."""
        shutdown_tracing()

        def gen():
            yield "hello"
            yield " "
            yield "world"

        result = list(traced_streaming_generate(
            gen(),
            model_id="test-model",
            prompt="test",
            max_tokens=100,
            temperature=0.7,
        ))

        assert result == ["hello", " ", "world"]


@requires_otel
class TestEnabledTracingMocked:
    """Tests for tracing when enabled (with mocked OTel).

    These tests require OpenTelemetry to be installed.
    """

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer with mock spans."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_tracer.start_span.return_value = mock_span

        yield {
            "tracer": mock_tracer,
            "span": mock_span,
        }

    def test_traced_generate_creates_span(self, mock_tracer):
        """traced_mlx_generate creates span with correct attributes."""
        tracer = mock_tracer["tracer"]
        span = mock_tracer["span"]

        with traced_mlx_generate(
            model_id="mlx-community/Qwen2.5-1.5B",
            prompt="Hello, world!",
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            tracer=tracer,
        ) as meta:
            meta["output_length"] = 100
            meta["success"] = True

        # Verify span was created
        tracer.start_as_current_span.assert_called_once_with("mlx_lm.generate")

        # Verify attributes were set
        calls = span.set_attribute.call_args_list
        attrs = {call[0][0]: call[0][1] for call in calls}

        assert attrs["mlx.model_id"] == "mlx-community/Qwen2.5-1.5B"
        assert attrs["mlx.device"] == "mps"
        assert attrs["gen_ai.prompt.length"] == 13
        assert attrs["gen_ai.request.max_tokens"] == 256
        assert attrs["gen_ai.request.temperature"] == 0.8
        assert attrs["gen_ai.streaming"] is False
        assert attrs["gen_ai.request.top_p"] == 0.95

    def test_traced_embed_creates_span(self, mock_tracer):
        """traced_mlx_embed creates span with correct attributes."""
        tracer = mock_tracer["tracer"]
        span = mock_tracer["span"]

        with traced_mlx_embed(
            model_id="nomic-ai/nomic-embed-text",
            input_text="This is a test document",
            is_query=False,
            batch_size=1,
            tracer=tracer,
        ) as meta:
            meta["embedding_dim"] = 768
            meta["pooling_used"] = "text_embeds"

        tracer.start_as_current_span.assert_called_once_with("mlx_embedding.embed")

        calls = span.set_attribute.call_args_list
        attrs = {call[0][0]: call[0][1] for call in calls}

        assert attrs["mlx.model_id"] == "nomic-ai/nomic-embed-text"
        assert attrs["embedding.input_length"] == 23
        assert attrs["embedding.is_query"] is False
        assert attrs["embedding.batch_size"] == 1

    def test_traced_request_creates_parent_span(self, mock_tracer):
        """traced_request creates parent span for operations."""
        tracer = mock_tracer["tracer"]
        span = mock_tracer["span"]

        with traced_request(
            "hybrid_search",
            tracer=tracer,
            attributes={"query": "test query", "limit": 10},
        ) as result_span:
            assert result_span is span

        tracer.start_as_current_span.assert_called_once_with("pmd.hybrid_search")

        calls = span.set_attribute.call_args_list
        attrs = {call[0][0]: call[0][1] for call in calls}

        assert attrs["query"] == "test query"
        assert attrs["limit"] == 10

    def test_traced_generate_records_exception(self, mock_tracer):
        """traced_mlx_generate records exceptions."""
        tracer = mock_tracer["tracer"]
        span = mock_tracer["span"]

        with pytest.raises(RuntimeError):
            with traced_mlx_generate(
                model_id="test-model",
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                tracer=tracer,
            ):
                raise RuntimeError("Generation failed")

        span.record_exception.assert_called_once()
        span.set_status.assert_called()

    def test_traced_streaming_wraps_generator(self, mock_tracer):
        """traced_streaming_generate wraps generator and tracks tokens."""
        tracer = mock_tracer["tracer"]
        span = mock_tracer["span"]

        def gen():
            yield "Hello"
            yield " "
            yield "World"

        result = list(traced_streaming_generate(
            gen(),
            model_id="test-model",
            prompt="Say hello",
            max_tokens=100,
            temperature=0.5,
            tracer=tracer,
        ))

        assert result == ["Hello", " ", "World"]

        tracer.start_span.assert_called_once_with("mlx_lm.generate")
        span.end.assert_called_once()

        # Verify streaming attribute was set
        calls = span.set_attribute.call_args_list
        attrs = {call[0][0]: call[0][1] for call in calls}
        assert attrs["gen_ai.streaming"] is True


class TestConfigIntegration:
    """Tests for config integration."""

    def test_tracing_config_from_dict(self):
        """TracingConfig can be created from dict-like updates."""
        config = TracingConfig()

        # Simulate how _update_dataclass works
        values = {
            "enabled": True,
            "phoenix_endpoint": "http://phoenix:6006/v1/traces",
            "sample_rate": 0.5,
        }
        for key, value in values.items():
            if hasattr(config, key):
                setattr(config, key, value)

        assert config.enabled is True
        assert config.phoenix_endpoint == "http://phoenix:6006/v1/traces"
        assert config.sample_rate == 0.5


class TestMissingDependencies:
    """Tests for graceful handling of missing OTel packages."""

    def test_wrappers_work_without_otel_installed(self):
        """Wrapper functions work even without OTel packages."""
        # This test verifies that our wrappers don't crash when
        # tracing is disabled (as it would be without OTel)
        shutdown_tracing()

        # All these should work without OTel
        with traced_mlx_generate(
            model_id="test",
            prompt="test",
            max_tokens=10,
            temperature=0.5,
        ) as meta:
            meta["success"] = True

        with traced_mlx_embed(
            model_id="test",
            input_text="test",
        ) as meta:
            meta["success"] = True

        with traced_request("test") as span:
            assert span is None

        # Cleanup
        shutdown_tracing()
