"""Tests for idx.core.observability module."""

import pytest

from idx.core.observability import (
    configure_observability,
    get_callback_manager,
    is_langfuse_available,
    is_observability_configured,
    reset_observability,
)


class TestIsLangfuseAvailable:
    """Tests for is_langfuse_available function."""

    def test_returns_boolean(self) -> None:
        """Function returns a boolean."""
        result = is_langfuse_available()
        assert isinstance(result, bool)


class TestConfigureObservability:
    """Tests for configure_observability function."""

    def setup_method(self) -> None:
        """Reset observability state before each test."""
        reset_observability()

    def teardown_method(self) -> None:
        """Reset observability state after each test."""
        reset_observability()

    def test_returns_false_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when Langfuse is disabled in settings."""
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "false")

        # Clear settings cache
        from idx.core.settings import get_settings

        get_settings.cache_clear()

        result = configure_observability()
        assert result is False
        assert is_observability_configured() is True

    def test_returns_false_when_credentials_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when credentials are missing."""
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "true")
        monkeypatch.delenv("IDX_LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("IDX_LANGFUSE_SECRET_KEY", raising=False)

        from idx.core.settings import get_settings

        get_settings.cache_clear()

        result = configure_observability()
        assert result is False
        assert is_observability_configured() is True

    def test_idempotent_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple calls return same result when disabled."""
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "false")

        from idx.core.settings import get_settings

        get_settings.cache_clear()

        result1 = configure_observability()
        result2 = configure_observability()

        assert result1 == result2
        assert result1 is False

    def test_returns_false_when_package_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when langfuse package is not installed."""
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("IDX_LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("IDX_LANGFUSE_SECRET_KEY", "sk-test")

        from idx.core.settings import get_settings

        get_settings.cache_clear()

        # Mock is_langfuse_available to return False
        import idx.core.observability as obs_module

        original_func = obs_module.is_langfuse_available
        monkeypatch.setattr(obs_module, "is_langfuse_available", lambda: False)

        result = configure_observability()
        assert result is False

        # Restore
        monkeypatch.setattr(obs_module, "is_langfuse_available", original_func)


class TestGetCallbackManager:
    """Tests for get_callback_manager function."""

    def setup_method(self) -> None:
        """Reset observability state before each test."""
        reset_observability()

    def teardown_method(self) -> None:
        """Reset observability state after each test."""
        reset_observability()

    def test_returns_none_before_configuration(self) -> None:
        """Returns None before configure_observability is called."""
        result = get_callback_manager()
        assert result is None

    def test_returns_none_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns None when Langfuse is disabled."""
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "false")

        from idx.core.settings import get_settings

        get_settings.cache_clear()

        configure_observability()
        result = get_callback_manager()
        assert result is None


class TestIsObservabilityConfigured:
    """Tests for is_observability_configured function."""

    def setup_method(self) -> None:
        """Reset observability state before each test."""
        reset_observability()

    def teardown_method(self) -> None:
        """Reset observability state after each test."""
        reset_observability()

    def test_returns_false_before_configuration(self) -> None:
        """Returns False before configure_observability is called."""
        assert is_observability_configured() is False

    def test_returns_true_after_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns True after configure_observability is called."""
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "false")

        from idx.core.settings import get_settings

        get_settings.cache_clear()

        configure_observability()
        assert is_observability_configured() is True


class TestResetObservability:
    """Tests for reset_observability function."""

    def test_resets_configured_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reset_observability clears the configured state."""
        monkeypatch.setenv("IDX_LANGFUSE_ENABLED", "false")

        from idx.core.settings import get_settings

        get_settings.cache_clear()

        configure_observability()
        assert is_observability_configured() is True

        reset_observability()
        assert is_observability_configured() is False

    def test_clears_callback_manager(self) -> None:
        """reset_observability clears the callback manager."""
        reset_observability()
        assert get_callback_manager() is None
