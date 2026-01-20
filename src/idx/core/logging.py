"""idx.core.logging - Logging configuration with loguru.

Provides unified logging for the idx library with LlamaIndex integration.
Uses loguru for structured logging and routes standard library logging
(used by LlamaIndex) to loguru.

Example usage:
    from idx.core.logging import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)
    logger.info("Starting operation")
"""

import logging
import sys
from typing import Any

from loguru import logger

__all__ = [
    "configure_logging",
    "get_logger",
    "InterceptHandler",
]

# Track if logging has been configured
_configured = False


class InterceptHandler(logging.Handler):
    """Handler that intercepts standard logging and routes to loguru.

    This allows libraries using standard logging (like LlamaIndex) to
    have their logs routed through loguru for unified formatting.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by routing it to loguru.

        Args:
            record: The log record to emit.
        """
        # Get corresponding loguru level
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller frame for proper log location
        frame = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _level_name_to_int(level: str) -> int:
    """Convert a log level name to its integer value.

    Args:
        level: Log level name (DEBUG, INFO, etc.).

    Returns:
        Integer log level value.
    """
    return getattr(logging, level.upper(), logging.INFO)


def configure_logging(
    level: str | None = None,
    *,
    format_string: str | None = None,
    intercept_standard_logging: bool = True,
    diagnose: bool = False,
) -> None:
    """Configure logging for the idx library.

    Sets up loguru logging with optional interception of standard library
    logging. Uses log level from settings if not specified.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses value from idx.core.settings.
        format_string: Custom format string for log output.
                      If None, uses a sensible default.
        intercept_standard_logging: If True, route standard logging to loguru.
        diagnose: If True, enable diagnostics in tracebacks (shows variable values).

    Example:
        configure_logging()  # Uses settings
        configure_logging(level="DEBUG")  # Override level
    """
    global _configured

    # Get level from settings if not specified
    if level is None:
        from idx.core.settings import get_settings

        settings = get_settings()
        level = settings.log_level

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Remove default handler and add configured one
    logger.remove()
    logger.add(
        sys.stderr,
        format=format_string,
        level=level.upper(),
        diagnose=diagnose,
        colorize=True,
    )

    # Intercept standard logging
    if intercept_standard_logging:
        logging.basicConfig(handlers=[InterceptHandler()], level=_level_name_to_int(level))

        # Route common library loggers through our handler
        for logger_name in [
            "llama_index",
            "llama_index.core",
            "llama_index.llms",
            "llama_index.embeddings",
            "httpx",
            "httpcore",
        ]:
            lib_logger = logging.getLogger(logger_name)
            lib_logger.handlers = [InterceptHandler()]
            lib_logger.propagate = False

    _configured = True
    logger.debug(f"Logging configured at level {level}")


def get_logger(name: str) -> Any:
    """Get a loguru logger bound to the given name.

    Returns a loguru logger with the module name bound for context.

    Args:
        name: The logger name, typically __name__.

    Returns:
        A loguru logger instance bound to the name.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logger.bind(name=name)


def is_configured() -> bool:
    """Check if logging has been configured.

    Returns:
        True if configure_logging has been called.
    """
    return _configured
