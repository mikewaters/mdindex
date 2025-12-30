"""Authentication and credentials abstraction for document sources.

This module provides a flexible way to handle credentials for remote sources,
supporting multiple credential providers (environment variables, system keyring,
configuration files, etc.).

Credential references in source configs use a URI-like format:
- $ENV:VAR_NAME - Read from environment variable
- $KEYRING:service/key - Read from system keyring
- $CONFIG:path.to.key - Read from config file
- literal value - Use the value directly (not recommended for secrets)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger

from ..core.exceptions import PMDError


# =============================================================================
# Exceptions
# =============================================================================


class CredentialError(PMDError):
    """Base exception for credential operations."""

    pass


class CredentialNotFoundError(CredentialError):
    """Credential could not be found."""

    def __init__(self, key: str, provider: str):
        self.key = key
        self.provider = provider
        super().__init__(f"Credential '{key}' not found in {provider}")


# =============================================================================
# Credential Provider Protocol
# =============================================================================


@runtime_checkable
class CredentialProvider(Protocol):
    """Protocol for credential providers.

    Credential providers retrieve secrets from various backends
    (environment, keyring, config files, etc.).
    """

    def get_credential(self, key: str) -> str | None:
        """Retrieve a credential by key.

        Args:
            key: Credential identifier (format depends on provider).

        Returns:
            Credential value if found, None otherwise.
        """
        ...

    @property
    def name(self) -> str:
        """Provider name for logging/errors."""
        ...


# =============================================================================
# Built-in Providers
# =============================================================================


class EnvironmentCredentials:
    """Read credentials from environment variables.

    Example:
        provider = EnvironmentCredentials()
        token = provider.get_credential("API_TOKEN")
    """

    @property
    def name(self) -> str:
        return "environment"

    def get_credential(self, key: str) -> str | None:
        """Get credential from environment variable.

        Args:
            key: Environment variable name.

        Returns:
            Variable value if set, None otherwise.
        """
        return os.environ.get(key)


class KeyringCredentials:
    """Read credentials from system keyring.

    Uses the `keyring` library to access the system's secure credential store.
    On macOS this uses Keychain, on Linux it uses Secret Service, etc.

    Example:
        provider = KeyringCredentials(service_name="pmd")
        token = provider.get_credential("github-token")
    """

    def __init__(self, service_name: str = "pmd"):
        """Initialize keyring provider.

        Args:
            service_name: Service name for keyring entries.
        """
        self._service_name = service_name
        self._keyring = None

    @property
    def name(self) -> str:
        return f"keyring:{self._service_name}"

    def _get_keyring(self):
        """Lazily import keyring module."""
        if self._keyring is None:
            try:
                import keyring
                self._keyring = keyring
            except ImportError:
                logger.warning("keyring module not installed, keyring credentials unavailable")
                return None
        return self._keyring

    def get_credential(self, key: str) -> str | None:
        """Get credential from system keyring.

        Args:
            key: Credential name within the service.

        Returns:
            Credential value if found, None otherwise.
        """
        keyring = self._get_keyring()
        if keyring is None:
            return None

        try:
            return keyring.get_password(self._service_name, key)
        except Exception as e:
            logger.warning(f"Failed to read from keyring: {e}")
            return None


class StaticCredentials:
    """Provide static/hardcoded credentials.

    Useful for testing or when credentials are passed directly.
    Not recommended for production use with real secrets.

    Example:
        provider = StaticCredentials({"api_key": "test-key"})
        key = provider.get_credential("api_key")
    """

    def __init__(self, credentials: dict[str, str] | None = None):
        """Initialize with static credentials.

        Args:
            credentials: Dictionary of key -> value pairs.
        """
        self._credentials = credentials or {}

    @property
    def name(self) -> str:
        return "static"

    def get_credential(self, key: str) -> str | None:
        """Get credential from static store.

        Args:
            key: Credential name.

        Returns:
            Credential value if found, None otherwise.
        """
        return self._credentials.get(key)

    def set_credential(self, key: str, value: str) -> None:
        """Set a credential (for testing).

        Args:
            key: Credential name.
            value: Credential value.
        """
        self._credentials[key] = value


# =============================================================================
# Credential Resolver
# =============================================================================


class CredentialResolver:
    """Resolves credential references to actual values.

    Parses credential reference strings and delegates to appropriate providers.

    Reference format:
    - $ENV:VAR_NAME -> EnvironmentCredentials
    - $KEYRING:key_name -> KeyringCredentials
    - $STATIC:key_name -> StaticCredentials (for testing)
    - any other value -> returned as-is (literal)

    Example:
        resolver = CredentialResolver()

        # From environment
        token = resolver.resolve("$ENV:GITHUB_TOKEN")

        # From keyring
        password = resolver.resolve("$KEYRING:db-password")

        # Literal value (not recommended for secrets)
        api_key = resolver.resolve("sk-1234567890")
    """

    def __init__(self):
        """Initialize resolver with default providers."""
        self._env = EnvironmentCredentials()
        self._keyring = KeyringCredentials()
        self._static = StaticCredentials()

    def resolve(self, reference: str, required: bool = True) -> str | None:
        """Resolve a credential reference to its value.

        Args:
            reference: Credential reference string or literal value.
            required: If True, raise error when credential not found.

        Returns:
            Resolved credential value, or None if not found and not required.

        Raises:
            CredentialNotFoundError: If required=True and credential not found.
        """
        if not reference:
            if required:
                raise CredentialNotFoundError("(empty)", "none")
            return None

        # Parse reference format: $PROVIDER:key
        if reference.startswith("$"):
            colon_idx = reference.find(":")
            if colon_idx == -1:
                # Malformed reference, treat as literal
                logger.warning(f"Malformed credential reference: {reference}")
                return reference

            provider_name = reference[1:colon_idx].upper()
            key = reference[colon_idx + 1:]

            value = self._resolve_from_provider(provider_name, key)

            if value is None and required:
                raise CredentialNotFoundError(key, provider_name)

            return value

        # Not a reference, return as literal
        return reference

    def _resolve_from_provider(self, provider: str, key: str) -> str | None:
        """Resolve credential from a specific provider.

        Args:
            provider: Provider name (ENV, KEYRING, STATIC).
            key: Credential key.

        Returns:
            Credential value or None.
        """
        if provider == "ENV":
            return self._env.get_credential(key)
        elif provider == "KEYRING":
            return self._keyring.get_credential(key)
        elif provider == "STATIC":
            return self._static.get_credential(key)
        else:
            logger.warning(f"Unknown credential provider: {provider}")
            return None

    def set_static(self, key: str, value: str) -> None:
        """Set a static credential (for testing).

        Args:
            key: Credential name.
            value: Credential value.
        """
        self._static.set_credential(key, value)


# =============================================================================
# Auth Configuration Types
# =============================================================================


@dataclass
class AuthConfig:
    """Authentication configuration for a source.

    Attributes:
        auth_type: Type of authentication (none, bearer, basic, api_key, custom).
        token: Token or password (can be a credential reference like $ENV:TOKEN).
        username: Username for basic auth.
        api_key_header: Header name for API key auth (default: X-API-Key).
        custom_headers: Additional headers to include.
    """

    auth_type: str = "none"
    token: str | None = None
    username: str | None = None
    api_key_header: str = "X-API-Key"
    custom_headers: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "AuthConfig":
        """Create AuthConfig from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            AuthConfig instance.
        """
        return cls(
            auth_type=data.get("auth_type", "none"),
            token=data.get("token") or data.get("auth_token"),
            username=data.get("username"),
            api_key_header=data.get("api_key_header", "X-API-Key"),
            custom_headers=data.get("custom_headers"),
        )

    def get_headers(self, resolver: CredentialResolver | None = None) -> dict[str, str]:
        """Get HTTP headers for this auth config.

        Args:
            resolver: Credential resolver for resolving references.

        Returns:
            Dictionary of HTTP headers.
        """
        headers: dict[str, str] = {}

        if resolver is None:
            resolver = CredentialResolver()

        if self.auth_type == "none":
            pass

        elif self.auth_type == "bearer":
            if self.token:
                resolved_token = resolver.resolve(self.token, required=True)
                if resolved_token:
                    headers["Authorization"] = f"Bearer {resolved_token}"

        elif self.auth_type == "basic":
            if self.username and self.token:
                import base64
                resolved_password = resolver.resolve(self.token, required=True)
                if resolved_password:
                    credentials = f"{self.username}:{resolved_password}"
                    encoded = base64.b64encode(credentials.encode()).decode()
                    headers["Authorization"] = f"Basic {encoded}"

        elif self.auth_type == "api_key":
            if self.token:
                resolved_key = resolver.resolve(self.token, required=True)
                if resolved_key:
                    headers[self.api_key_header] = resolved_key

        # Add custom headers
        if self.custom_headers:
            for key, value in self.custom_headers.items():
                resolved_value = resolver.resolve(value, required=False)
                if resolved_value:
                    headers[key] = resolved_value

        return headers


# =============================================================================
# Default Resolver Instance
# =============================================================================

_default_resolver: CredentialResolver | None = None


def get_default_resolver() -> CredentialResolver:
    """Get the default credential resolver.

    Returns:
        Default CredentialResolver instance.
    """
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = CredentialResolver()
    return _default_resolver
