# WDBX Code Quality Guide

This document outlines the code quality standards and best practices for contributing to the WDBX Python codebase. Following these guidelines ensures consistent, maintainable, and high-quality code.

## Code Style

WDBX follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code with some specific additions:

### Linting and Formatting

We use [Ruff](https://github.com/charliermarsh/ruff) as our primary linter. It combines the functionality of multiple Python linters like Flake8, isort, and more into a single fast tool.

To install Ruff:

```bash
pip install ruff
```

To check your code:

```bash
ruff check src/
```

To automatically fix many issues:

```bash
ruff check --fix src/
```

### Import Order

Imports should be organized in the following order:

1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

Within each group, imports should be sorted alphabetically.

Example:

```python
# Standard library
import json
import logging
import os
import threading
from datetime import datetime

# Third-party libraries
import numpy as np

# Local modules
from ..utils.diagnostics import time_operation
```

### Line Length

Maximum line length is 88 characters. This is slightly longer than the traditional PEP 8 recommendation (79-80 characters) but matches the default for modern formatters like Black.

For long expressions, break them over multiple lines using parentheses and appropriate indentation:

```python
# Good
result = some_function(
    argument1,
    argument2,
    another_long_named_argument,
    final_argument
)

# Also good
long_string = (
    "This is a very long string that needs to be "
    "broken up over multiple lines to stay within "
    "the line length limit."
)
```

## Documentation

### Docstrings

All modules, classes, methods, and functions should have docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):

```python
def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.
    
    Args:
        param1: The first parameter.
        param2: The second parameter.
        
    Returns:
        The return value. True for success, False otherwise.
        
    Raises:
        ValueError: If param1 is None.
        TypeError: If param2 is not a string.
    """
```

### Type Annotations

Use Python's type annotations for function signatures:

```python
def greeting(name: str) -> str:
    return f"Hello {name}"
```

## Error Handling

### Guard Clauses

Use guard clauses to validate inputs early and return/raise exceptions if necessary:

```python
def process_data(data: Dict[str, Any]) -> List[float]:
    # Guard clauses
    if not data:
        return []
        
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary")
        
    # Actual processing begins here
    ...
```

### Exception Handling

Be specific about which exceptions you catch, and always include error messages:

```python
try:
    result = some_operation()
except ValueError as e:
    logging.error(f"Invalid value encountered: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    # Consider whether to re-raise or handle gracefully
```

## Performance Optimization

### Logging

Use the `logging` module instead of print statements. Include appropriate log levels:

```python
# Debug for detailed diagnostic information
logging.debug("Processing item %s", item_id)

# Info for confirmation of expected behavior
logging.info("Application started successfully")

# Warning for potential issues
logging.warning("Database connection pool running low")

# Error for runtime errors
logging.error("Failed to connect to database: %s", str(e))

# Critical for critical failures
logging.critical("System is shutting down due to fatal error")
```

### Memory Management

For memory-intensive operations:

1. Use context managers for proper resource cleanup
2. Implement memory usage monitoring
3. Use generators for large datasets instead of loading everything into memory
4. Call garbage collection when appropriate

Example:

```python
def optimize_memory():
    """Optimize memory usage by cleaning up resources."""
    import gc
    
    # Log current memory usage
    mem_before = get_memory_usage()
    logging.info(f"Memory usage before: {mem_before.get('percent')}%")
    
    # Perform memory optimization
    gc.collect()
    
    # Log results
    mem_after = get_memory_usage()
    diff = mem_after.get('percent', 0) - mem_before.get('percent', 0)
    logging.info(f"Memory usage after: {mem_after.get('percent')}%, change: {diff:.2f}%")
```

## Testing

### Unit Tests

Write unit tests for all new functionality using pytest:

```python
def test_function_returns_expected_result():
    # Arrange
    input_data = {"key": "value"}
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_result
```

### Test Coverage

Aim for at least 80% test coverage for all new code. Use the coverage package to measure:

```bash
pip install coverage
coverage run -m pytest
coverage report
```

## Continuous Integration

All pull requests will be checked against:

1. Linting with Ruff
2. Type checking with mypy
3. Unit tests with pytest
4. Test coverage requirements

## Best Practices Checklist

Before submitting code for review, check that your changes:

- [ ] Pass all linter checks with Ruff
- [ ] Include comprehensive docstrings
- [ ] Have appropriate type annotations
- [ ] Include proper error handling
- [ ] Have associated unit tests
- [ ] Document any API changes
- [ ] Use meaningful variable and function names
- [ ] Avoid duplicated code (DRY principle)
- [ ] Handle edge cases appropriately

By following these guidelines, we can maintain a high-quality codebase that is easier to maintain, extend, and troubleshoot. 