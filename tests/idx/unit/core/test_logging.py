"""Tests for idx.core.logging module."""

import logging
import sys
from io import StringIO

import pytest

from idx.core.logging import (
    InterceptHandler,
    configure_logging,
    get_logger,
    is_configured,
)


class TestInterceptHandler:
    """Tests for InterceptHandler class."""

    def test_handler_is_logging_handler(self) -> None:
        """InterceptHandler is a valid logging.Handler."""
        handler = InterceptHandler()
        assert isinstance(handler, logging.Handler)

    def test_emit_does_not_raise(self) -> None:
        """emit() handles log records without raising."""
        handler = InterceptHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Should not raise
        handler.emit(record)

    def test_emit_handles_all_levels(self) -> None:
        """emit() handles all standard log levels."""
        handler = InterceptHandler()

        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg=f"Message at level {level}",
                args=(),
                exc_info=None,
            )
            # Should not raise
            handler.emit(record)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self) -> None:
        """Reset logging state before each test."""
        import idx.core.logging as log_module

        log_module._configured = False

    def test_configures_with_default_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Logging uses level from settings by default."""
        monkeypatch.setenv("IDX_LOG_LEVEL", "DEBUG")

        # Clear settings cache
        from idx.core.settings import get_settings

        get_settings.cache_clear()

        configure_logging()
        assert is_configured() is True

    def test_configures_with_explicit_level(self) -> None:
        """Logging uses explicit level when provided."""
        configure_logging(level="WARNING")
        assert is_configured() is True

    def test_configures_intercept_handler(self) -> None:
        """Standard logging is intercepted when enabled."""
        configure_logging(level="INFO", intercept_standard_logging=True)

        # Check that llama_index logger has intercept handler
        llama_logger = logging.getLogger("llama_index")
        assert any(isinstance(h, InterceptHandler) for h in llama_logger.handlers)

    def test_skips_intercept_when_disabled(self) -> None:
        """Standard logging not intercepted when disabled."""
        # Create a fresh logger
        test_logger = logging.getLogger("test_no_intercept_unique")
        test_logger.handlers = []

        configure_logging(level="INFO", intercept_standard_logging=False)

        # The test logger shouldn't have the intercept handler added
        assert not any(isinstance(h, InterceptHandler) for h in test_logger.handlers)

    def test_accepts_custom_format_string(self) -> None:
        """Custom format string is accepted."""
        custom_format = "{time} | {level} | {message}"
        # Should not raise
        configure_logging(level="INFO", format_string=custom_format)
        assert is_configured() is True


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_callable_logger(self) -> None:
        """get_logger returns a logger with callable methods."""
        log = get_logger("test.module")
        assert log is not None
        assert callable(log.info)
        assert callable(log.debug)
        assert callable(log.warning)
        assert callable(log.error)

    def test_logger_can_log_messages(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Logger can log messages without raising."""
        configure_logging(level="DEBUG")
        log = get_logger("test.logging")

        # Should not raise
        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")

    def test_different_names_return_different_contexts(self) -> None:
        """Different module names result in different logger contexts."""
        log1 = get_logger("module.one")
        log2 = get_logger("module.two")
        # They should be distinct bound loggers
        # Both should be callable
        assert callable(log1.info)
        assert callable(log2.info)


class TestIsConfigured:
    """Tests for is_configured function."""

    def setup_method(self) -> None:
        """Reset logging state before each test."""
        import idx.core.logging as log_module

        log_module._configured = False

    def test_returns_false_before_configuration(self) -> None:
        """is_configured returns False before configure_logging is called."""
        assert is_configured() is False

    def test_returns_true_after_configuration(self) -> None:
        """is_configured returns True after configure_logging is called."""
        configure_logging(level="INFO")
        assert is_configured() is True


class TestLoggingOutput:
    """Tests for actual logging output."""

    def setup_method(self) -> None:
        """Reset logging state before each test."""
        import idx.core.logging as log_module

        log_module._configured = False

    def test_logs_appear_in_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Logs are written to stderr."""
        configure_logging(level="INFO")
        log = get_logger("output.test")

        log.info("Test output message")

        captured = capsys.readouterr()
        assert "Test output message" in captured.err

    def test_debug_logs_hidden_at_info_level(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Debug logs are not shown when level is INFO."""
        configure_logging(level="INFO")
        log = get_logger("level.test")

        log.debug("Should not appear")
        log.info("Should appear")

        captured = capsys.readouterr()
        assert "Should not appear" not in captured.err
        assert "Should appear" in captured.err

    def test_standard_logging_routed_to_loguru(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Standard library logs appear in loguru output."""
        configure_logging(level="INFO", intercept_standard_logging=True)

        # Use a standard logger
        std_logger = logging.getLogger("test.standard.route")
        std_logger.handlers = [InterceptHandler()]
        std_logger.setLevel(logging.INFO)
        std_logger.info("Standard library message")

        captured = capsys.readouterr()
        assert "Standard library message" in captured.err
