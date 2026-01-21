# Console script example

This document describes the utility script standards for this project.. **All utility scripts** must follow this pattern.

## Utility Script Requirements

1. Uses `uv run python` to execute the script
2. Uses Python's "inline script metadata" standard to define the dependencies in the script itself, including dependencies on internal modules in `src/`.
3. The script does not need to be made executable.
 
## Integrating utility scripts with your code

If a utility script needs to make calls into one of our python modules, it should use normal python import semantics. Please see the example for more information.

**You should never** need to modify the system PATH or PYTHON_PATH, or do any shell tricks whatsoever. As long as you use `uv run python` to run the script it should be able to import any python module present in `src/`.

## Executing Utility Scripts

Scripts must be executed using `uv`:
`uv run python scripts/my-script-name.py`, from the project root directory.

## Creating Utility scripts
Utility scripts must be placed in the project's top-leve `scripts/` directory.

Use the below markdown code block as a template for new utility scripts.

```python
# /// script
# dependencies = [
#   "idx",
# ]
# ///

"""
This is an example of a console script
"""


# Note: This shouuld work without path munging:
from idx.core.logging import get_logger

logger = get_logger(__name__)

logger.debug("message")
```
