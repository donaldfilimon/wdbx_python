# WDBX Code Linting Guide

<!-- category: Development -->
<!-- priority: 70 -->
<!-- tags: linting, code quality, style guide, best practices -->

This guide covers the code linting standards and tools used in the WDBX project.

## Overview

WDBX uses several linting tools to maintain code quality:
- Ruff for Python linting and formatting
- mypy for static type checking
- markdownlint for documentation linting

## Ruff Configuration

### Installation

```bash
pip install ruff
```

### Configuration

The project uses a `pyproject.toml` file for Ruff configuration:

```toml
[tool.ruff]
line-length = 88
target-version = "py38"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "C",  # flake8-comprehensions
    "B",  # flake8-bugbear
]
ignore = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

### Running Ruff

```bash
# Check files
ruff check .

# Format files
ruff format .
```

## Type Checking with mypy

### Installation

```bash
pip install mypy
```

### Configuration

The `mypy.ini` configuration:

```ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True

[mypy.plugins.*]
ignore_missing_imports = True
```

### Running mypy

```bash
mypy src/wdbx
```

## Documentation Linting

### Installation

```bash
npm install -g markdownlint-cli
```

### Configuration

The `.markdownlint.json` configuration:

```json
{
  "default": true,
  "MD013": { "line_length": 88 },
  "MD033": false,
  "MD041": false
}
```

### Running markdownlint

```bash
markdownlint docs/**/*.md
```

## Pre-commit Hooks

### Installation

```bash
pip install pre-commit
```

### Configuration

The `.pre-commit-config.yaml` file:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/igorshubovych/markdownlint-cli
    rev: v0.37.0
    hooks:
      - id: markdownlint
```

### Using Pre-commit

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Code Style Guide

### Imports

1. Group imports in this order:
   - Standard library
   - Third-party packages
   - Local imports

2. Sort imports alphabetically within groups:

```python
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch

from wdbx.core import Database
from wdbx.utils import logger
```

### Line Length

- Maximum line length: 88 characters
- Break long lines at logical points:

```python
def long_function_name(
    parameter_1: str,
    parameter_2: int,
    parameter_3: List[str],
) -> Optional[str]:
    return None
```

### Docstrings

Use Google-style docstrings:

```python
def process_data(data: List[str], limit: int = 100) -> Dict[str, Any]:
    """Process a list of strings and return results.

    Args:
        data: List of strings to process.
        limit: Maximum number of items to process.

    Returns:
        Dictionary containing processed results.

    Raises:
        ValueError: If data is empty or limit is negative.
    """
```

### Type Hints

Always use type hints for function arguments and return values:

```python
from typing import Dict, List, Optional, Union

def get_user(
    user_id: str,
    fields: Optional[List[str]] = None,
) -> Dict[str, Union[str, int]]:
    """Get user information."""
```

## Common Issues and Solutions

### Import Order

❌ Bad:
```python
from wdbx.utils import logger
import os
import numpy as np
```

✅ Good:
```python
import os

import numpy as np

from wdbx.utils import logger
```

### Line Length

❌ Bad:
```python
result = very_long_function_name(first_parameter, second_parameter, third_parameter, fourth_parameter)
```

✅ Good:
```python
result = very_long_function_name(
    first_parameter,
    second_parameter,
    third_parameter,
    fourth_parameter,
)
```

### Type Hints

❌ Bad:
```python
def process_data(data, limit=100):
    return {"result": data[:limit]}
```

✅ Good:
```python
def process_data(
    data: List[str],
    limit: int = 100,
) -> Dict[str, List[str]]:
    return {"result": data[:limit]}
```

## Continuous Integration

The project uses GitHub Actions to run linting checks:

```yaml
name: Lint

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff mypy
      
      - name: Run Ruff
        run: ruff check .
      
      - name: Run mypy
        run: mypy src/wdbx
```

## Best Practices

1. Run linters locally before committing
2. Use pre-commit hooks
3. Keep configurations up to date
4. Document any linting exceptions
5. Review linting reports regularly

## Troubleshooting

### Common Ruff Issues

1. Import sorting:
   ```bash
   ruff check --select I --fix .
   ```

2. Line length:
   ```bash
   ruff format .
   ```

### Common mypy Issues

1. Missing type hints:
   ```bash
   mypy --warn-return-any .
   ```

2. Import errors:
   ```bash
   mypy --ignore-missing-imports .
   ```

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [markdownlint Rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)
- [Python Type Hints](https://docs.python.org/3/library/typing.html) 