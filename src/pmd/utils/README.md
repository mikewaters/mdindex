# Utils Module Architecture

**Location:** `src/pmd/utils/`

Utility functions used across the application.

## Files and Key Abstractions

### `hashing.py`

**`sha256_hash(content: str)`** - SHA256 hash of text  
**`sha256_hash_bytes(content: bytes)`** - SHA256 hash of bytes

Used for content-addressable storage in documents.
