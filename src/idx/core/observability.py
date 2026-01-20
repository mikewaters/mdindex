"""idx.core.observability - Langfuse observability integration.

Provides Langfuse integration for tracing LLM calls via LlamaIndex.
Configuration is driven by settings (IDX_LANGFUSE_* environment variables).

Example usage:
    from idx.core.observability import configure_observability

    # Configure at application startup
    if configure_observability():
        print("Langfuse tracing enabled")
"""

from typing import TYPE_CHECKING

from idx.core.logging import get_logger

if TYPE_CHECKING:
    from llama_index.core.callbacks import CallbackManager

__all__ = [
    "configure_observability",
    "get_callback_manager",
    "is_langfuse_available",
]

logger = get_logger(__name__)

# Track if observability has been configured
_configured = False
_callback_manager: "CallbackManager | None" = None


def is_langfuse_available() -> bool:
    """Check if Langfuse package is installed.

    Returns:
        True if langfuse is available for import.
    """
    try:
        import langfuse  # noqa: F401

        return True
    except ImportError:
        return False


def configure_observability() -> bool:
    """Configure Langfuse observability integration.

    Checks settings to see if Langfuse is enabled and credentials are configured.
    If so, sets up the LlamaIndex callback manager with Langfuse tracing.

    Returns:
        True if Langfuse was successfully configured, False otherwise.

    Note:
        This function is idempotent - calling it multiple times is safe.
        If Langfuse is not enabled in settings or credentials are missing,
        this returns False without error.
    """
    global _configured, _callback_manager

    # Return early if already configured
    if _configured:
        logger.debug("Langfuse observability already configured")
        return _callback_manager is not None

    from idx.core.settings import get_settings

    settings = get_settings()

    # Check if Langfuse is enabled
    if not settings.langfuse.enabled:
        logger.debug("Langfuse is disabled in settings")
        _configured = True
        return False

    # Check if Langfuse package is available
    if not is_langfuse_available():
        logger.warning(
            "Langfuse is enabled but the 'langfuse' package is not installed. "
            "Install it with: pip install langfuse"
        )
        _configured = True
        return False

    # Check for required credentials
    if not settings.langfuse.public_key or not settings.langfuse.secret_key:
        logger.warning(
            "Langfuse is enabled but credentials are missing. "
            "Set IDX_LANGFUSE_PUBLIC_KEY and IDX_LANGFUSE_SECRET_KEY."
        )
        _configured = True
        return False

    # Attempt to configure Langfuse
    try:
        _callback_manager = _setup_langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.host,
        )
        _configured = True
        logger.info("Langfuse observability configured successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Langfuse: {e}")
        _configured = True
        return False


def _setup_langfuse(
    public_key: str,
    secret_key: str,
    host: str | None = None,
) -> "CallbackManager":
    """Set up Langfuse callback manager for LlamaIndex.

    Args:
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        host: Optional Langfuse host URL for self-hosted instances.

    Returns:
        A CallbackManager configured with Langfuse handler.

    Raises:
        ImportError: If required packages are not installed.
    """
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager

    try:
        # Try to import the official LlamaIndex Langfuse integration
        from langfuse.llama_index import LlamaIndexCallbackHandler

        # Create the callback handler
        handler_kwargs = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            handler_kwargs["host"] = host

        langfuse_handler = LlamaIndexCallbackHandler(**handler_kwargs)

        # Create callback manager and add handler
        callback_manager = CallbackManager([langfuse_handler])

        # Set as global callback manager in LlamaIndex Settings
        Settings.callback_manager = callback_manager

        return callback_manager

    except ImportError as e:
        logger.warning(
            f"Could not import Langfuse LlamaIndex integration: {e}. "
            "Make sure 'langfuse' is installed with LlamaIndex support."
        )
        raise


def get_callback_manager() -> "CallbackManager | None":
    """Get the configured callback manager.

    Returns:
        The CallbackManager if configured, None otherwise.

    Note:
        Call configure_observability() first to set up the callback manager.
    """
    return _callback_manager


def is_observability_configured() -> bool:
    """Check if observability has been configured.

    Returns:
        True if configure_observability has been called.
    """
    return _configured


def reset_observability() -> None:
    """Reset observability state (for testing).

    This clears the configured state and callback manager,
    allowing configure_observability to be called again.
    """
    global _configured, _callback_manager
    _configured = False
    _callback_manager = None
