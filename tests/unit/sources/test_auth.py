"""Tests for authentication and credential resolution."""

import os
import pytest
import base64

from pmd.sources.auth import (
    AuthConfig,
    CredentialError,
    CredentialNotFoundError,
    CredentialProvider,
    CredentialResolver,
    EnvironmentCredentials,
    KeyringCredentials,
    StaticCredentials,
    get_default_resolver,
)


class TestEnvironmentCredentials:
    """Tests for EnvironmentCredentials provider."""

    def test_get_existing_var(self):
        """Gets value from environment variable."""
        os.environ["TEST_CREDENTIAL"] = "secret123"
        try:
            provider = EnvironmentCredentials()
            value = provider.get_credential("TEST_CREDENTIAL")
            assert value == "secret123"
        finally:
            del os.environ["TEST_CREDENTIAL"]

    def test_get_missing_var(self):
        """Returns None for missing variable."""
        provider = EnvironmentCredentials()
        # Ensure it doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)
        value = provider.get_credential("NONEXISTENT_VAR")
        assert value is None

    def test_name(self):
        """Provider has correct name."""
        provider = EnvironmentCredentials()
        assert provider.name == "environment"

    def test_is_protocol_compliant(self):
        """Satisfies CredentialProvider protocol."""
        provider = EnvironmentCredentials()
        assert isinstance(provider, CredentialProvider)


class TestStaticCredentials:
    """Tests for StaticCredentials provider."""

    def test_get_credential(self):
        """Gets value from static dictionary."""
        provider = StaticCredentials({"key": "value"})
        assert provider.get_credential("key") == "value"

    def test_get_missing(self):
        """Returns None for missing key."""
        provider = StaticCredentials({})
        assert provider.get_credential("missing") is None

    def test_set_credential(self):
        """Can set credentials dynamically."""
        provider = StaticCredentials()
        provider.set_credential("new_key", "new_value")
        assert provider.get_credential("new_key") == "new_value"

    def test_name(self):
        """Provider has correct name."""
        provider = StaticCredentials()
        assert provider.name == "static"

    def test_is_protocol_compliant(self):
        """Satisfies CredentialProvider protocol."""
        provider = StaticCredentials()
        assert isinstance(provider, CredentialProvider)


class TestKeyringCredentials:
    """Tests for KeyringCredentials provider."""

    def test_name_includes_service(self):
        """Provider name includes service name."""
        provider = KeyringCredentials(service_name="myapp")
        assert provider.name == "keyring:myapp"

    def test_default_service_name(self):
        """Default service name is 'pmd'."""
        provider = KeyringCredentials()
        assert provider.name == "keyring:pmd"

    def test_returns_none_when_keyring_unavailable(self):
        """Returns None when keyring module unavailable."""
        provider = KeyringCredentials()
        # Force keyring to appear unavailable
        provider._keyring = None
        # Mock _get_keyring to return None
        provider._get_keyring = lambda: None

        value = provider.get_credential("some-key")
        assert value is None


class TestCredentialResolver:
    """Tests for CredentialResolver."""

    def test_resolve_env_reference(self):
        """Resolves $ENV: references."""
        os.environ["MY_SECRET"] = "env_value"
        try:
            resolver = CredentialResolver()
            value = resolver.resolve("$ENV:MY_SECRET")
            assert value == "env_value"
        finally:
            del os.environ["MY_SECRET"]

    def test_resolve_static_reference(self):
        """Resolves $STATIC: references."""
        resolver = CredentialResolver()
        resolver.set_static("my_key", "static_value")

        value = resolver.resolve("$STATIC:my_key")
        assert value == "static_value"

    def test_resolve_literal_value(self):
        """Returns literal value unchanged."""
        resolver = CredentialResolver()
        value = resolver.resolve("literal-token-value")
        assert value == "literal-token-value"

    def test_resolve_missing_required_raises(self):
        """Raises for missing required credential."""
        resolver = CredentialResolver()
        os.environ.pop("NONEXISTENT", None)

        with pytest.raises(CredentialNotFoundError) as exc_info:
            resolver.resolve("$ENV:NONEXISTENT", required=True)

        assert exc_info.value.key == "NONEXISTENT"
        assert exc_info.value.provider == "ENV"

    def test_resolve_missing_optional_returns_none(self):
        """Returns None for missing optional credential."""
        resolver = CredentialResolver()
        os.environ.pop("NONEXISTENT", None)

        value = resolver.resolve("$ENV:NONEXISTENT", required=False)
        assert value is None

    def test_resolve_empty_required_raises(self):
        """Raises for empty required reference."""
        resolver = CredentialResolver()

        with pytest.raises(CredentialNotFoundError):
            resolver.resolve("", required=True)

    def test_resolve_empty_optional_returns_none(self):
        """Returns None for empty optional reference."""
        resolver = CredentialResolver()
        value = resolver.resolve("", required=False)
        assert value is None

    def test_resolve_malformed_reference(self):
        """Malformed reference treated as literal."""
        resolver = CredentialResolver()
        # Missing colon
        value = resolver.resolve("$ENV")
        assert value == "$ENV"

    def test_resolve_unknown_provider(self):
        """Unknown provider returns None."""
        resolver = CredentialResolver()
        value = resolver.resolve("$UNKNOWN:key", required=False)
        assert value is None

    def test_resolve_case_insensitive_provider(self):
        """Provider names are case-insensitive."""
        resolver = CredentialResolver()
        resolver.set_static("test", "value")

        assert resolver.resolve("$static:test") == "value"
        assert resolver.resolve("$STATIC:test") == "value"
        assert resolver.resolve("$Static:test") == "value"


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_defaults(self):
        """AuthConfig has sensible defaults."""
        config = AuthConfig()
        assert config.auth_type == "none"
        assert config.token is None
        assert config.username is None
        assert config.api_key_header == "X-API-Key"
        assert config.custom_headers is None

    def test_from_dict(self):
        """Can create from dictionary."""
        data = {
            "auth_type": "bearer",
            "token": "$ENV:TOKEN",
            "username": "user",
        }
        config = AuthConfig.from_dict(data)

        assert config.auth_type == "bearer"
        assert config.token == "$ENV:TOKEN"
        assert config.username == "user"

    def test_from_dict_uses_auth_token_alias(self):
        """from_dict accepts auth_token as alias for token."""
        data = {"auth_token": "secret"}
        config = AuthConfig.from_dict(data)
        assert config.token == "secret"

    def test_get_headers_none_auth(self):
        """get_headers returns empty for no auth."""
        config = AuthConfig(auth_type="none")
        headers = config.get_headers()
        assert headers == {}

    def test_get_headers_bearer(self):
        """get_headers creates Authorization header for bearer."""
        os.environ["BEARER_TOKEN"] = "my-token"
        try:
            config = AuthConfig(auth_type="bearer", token="$ENV:BEARER_TOKEN")
            headers = config.get_headers()
            assert headers["Authorization"] == "Bearer my-token"
        finally:
            del os.environ["BEARER_TOKEN"]

    def test_get_headers_bearer_literal(self):
        """get_headers works with literal token."""
        config = AuthConfig(auth_type="bearer", token="literal-token")
        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer literal-token"

    def test_get_headers_basic(self):
        """get_headers creates Basic auth header."""
        config = AuthConfig(
            auth_type="basic",
            username="user",
            token="password",
        )
        headers = config.get_headers()

        expected = base64.b64encode(b"user:password").decode()
        assert headers["Authorization"] == f"Basic {expected}"

    def test_get_headers_api_key(self):
        """get_headers creates API key header."""
        config = AuthConfig(
            auth_type="api_key",
            token="api-key-value",
            api_key_header="X-Custom-Key",
        )
        headers = config.get_headers()
        assert headers["X-Custom-Key"] == "api-key-value"

    def test_get_headers_api_key_default_header(self):
        """get_headers uses default API key header."""
        config = AuthConfig(auth_type="api_key", token="key123")
        headers = config.get_headers()
        assert headers["X-API-Key"] == "key123"

    def test_get_headers_custom_headers(self):
        """get_headers includes custom headers."""
        config = AuthConfig(
            auth_type="none",
            custom_headers={
                "X-Custom": "value",
                "X-Env": "$ENV:CUSTOM_HEADER",
            },
        )
        os.environ["CUSTOM_HEADER"] = "env-value"
        try:
            headers = config.get_headers()
            assert headers["X-Custom"] == "value"
            assert headers["X-Env"] == "env-value"
        finally:
            del os.environ["CUSTOM_HEADER"]

    def test_get_headers_custom_resolver(self):
        """get_headers accepts custom resolver."""
        resolver = CredentialResolver()
        resolver.set_static("token", "static-token")

        config = AuthConfig(auth_type="bearer", token="$STATIC:token")
        headers = config.get_headers(resolver)

        assert headers["Authorization"] == "Bearer static-token"


class TestDefaultResolver:
    """Tests for default credential resolver."""

    def test_default_resolver_exists(self):
        """Default resolver is available."""
        resolver = get_default_resolver()
        assert isinstance(resolver, CredentialResolver)

    def test_default_resolver_singleton(self):
        """Default resolver is a singleton."""
        r1 = get_default_resolver()
        r2 = get_default_resolver()
        assert r1 is r2


class TestCredentialExceptions:
    """Tests for credential exceptions."""

    def test_credential_error_is_pmd_error(self):
        """CredentialError inherits from PMDError."""
        from pmd.core.exceptions import PMDError
        assert issubclass(CredentialError, PMDError)

    def test_credential_not_found_error(self):
        """CredentialNotFoundError contains key and provider."""
        error = CredentialNotFoundError("my-key", "ENV")
        assert error.key == "my-key"
        assert error.provider == "ENV"
        assert "my-key" in str(error)
        assert "ENV" in str(error)
