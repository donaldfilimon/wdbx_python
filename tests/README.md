# WDBX Python Tests

This directory contains test files for the WDBX project, organized by functionality:

## Directory Structure

- `imports/`: Tests for import validation and dependency management
  - `test_imports.py`: Verify basic module imports
  - `test_optim_import.py`: Test optimized module imports

- `ml/`: Machine learning related tests
  - `test_ml_integration.py`: Test ML backend integration

- Other test directories (pre-existing):
  - `unit/`: Unit tests for individual components
  - `integration/`: Integration tests for interacting components
  - `functional/`: Functional tests for complete workflows

## Running Tests

Tests can be run using the test runner script:

```bash
python scripts/runners/run_tests.py
```

Alternatively, you can use pytest directly:

```bash
pytest tests/
```

## Adding New Tests

When adding new tests, please follow these guidelines:

1. Place tests in the appropriate subdirectory based on functionality
2. Use descriptive test names with the `test_` prefix
3. Document test requirements and setup procedures
4. Include both positive and negative test cases 