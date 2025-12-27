"""Hashing utilities for PMD."""

import hashlib


def sha256_hash(content: str) -> str:
    """Calculate SHA256 hash of content.

    Args:
        content: Text content to hash.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    return hashlib.sha256(content.encode()).hexdigest()


def sha256_hash_bytes(content: bytes) -> str:
    """Calculate SHA256 hash of bytes.

    Args:
        content: Binary content to hash.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    return hashlib.sha256(content).hexdigest()
