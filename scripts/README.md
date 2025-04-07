# WDBX Python Scripts

This directory contains utility scripts for the WDBX project, organized by function:

## Directory Structure

- `benchmarks/`: Performance testing and benchmarking scripts
  - `vector_store_benchmark.py`: Benchmarks for vector store operations

- `linters/`: Code quality and linting scripts
  - `fix_fstrings.py`: Fix f-string formatting
  - `fix_imports.py`: Fix import statements
  - `fix_lint.py`: General linting fixes
  - `fix_linelength.py`: Fix line length issues
  - `fix_unused_vars.py`: Remove unused variables
  - `fix_whitespace.py`: Fix whitespace issues

- `runners/`: Scripts for running the application and tests
  - `run_streamlit.py`: Launch the Streamlit visualization UI
  - `run_tests.py`: Run project tests
  - `run_wdbx.py`: Run the WDBX application

## Usage

### Direct Script Usage

Most scripts can be run directly with Python:

```bash
python scripts/runners/run_tests.py
python scripts/benchmarks/vector_store_benchmark.py
```

For linter scripts, you typically need to provide a file or directory:

```bash
python scripts/linters/fix_lint.py src/wdbx
```

### Unified Tool Launcher

For convenience, you can also use the unified tool launcher in the project root:

```bash
# Run tests
python wdbx_tool.py test

# Run benchmarks
python wdbx_tool.py benchmark

# Run linters
python wdbx_tool.py lint --path src/wdbx
```

See `python wdbx_tool.py --help` for all available commands and options.

# WDBX Linting Tools

This directory contains scripts for linting and fixing common issues in the WDBX codebase.

## Available Scripts

- `wdbx_linter.py` - General-purpose linting tool for the entire codebase
- `fix_discord_bot.py` - Specialized fixer for Discord bot-specific issues
- `run_linters.py` - Comprehensive script that runs all linters and generates a report
- `integrated_linter.py` - Integrated solution combining industry-standard tools with custom WDBX linters
- `lint_config.py` - Configuration file for linting tools

## Usage

### Integrated Linter (Recommended)

The integrated linter combines multiple industry-standard tools with our custom linters:

```bash
# Run all available linters in check-only mode
python scripts/integrated_linter.py

# Apply fixes automatically
python scripts/integrated_linter.py --fix

# Target a specific directory or file
python scripts/integrated_linter.py --target src/wdbx/plugins

# Run only specific tools
python scripts/integrated_linter.py --tools ruff,black,wdbx_linters

# Install missing linting tools
python scripts/integrated_linter.py --install-missing

# Generate a report (for WDBX custom linters)
python scripts/integrated_linter.py --report
```

### Run All WDBX Custom Linters

If you only want to run the WDBX custom linters:

```bash
# Run a dry-run (report issues without fixing)
python scripts/run_linters.py

# Apply fixes automatically
python scripts/run_linters.py --fix

# Generate an HTML report of issues
python scripts/run_linters.py --report

# Target a specific directory or file
python scripts/run_linters.py --target src/wdbx/plugins
```

### General WDBX Linter

The general linter can be used to fix common issues in Python files:

```bash
# Run in dry-run mode (default)
python scripts/wdbx_linter.py

# Apply fixes automatically
python scripts/wdbx_linter.py --fix

# Target a specific file or directory
python scripts/wdbx_linter.py --target src/wdbx/plugins/discord_bot.py
```

### Discord Bot Fixer

The Discord bot fixer is a specialized tool for fixing issues in the Discord bot plugin:

```bash
# Run the fixer with default settings
python scripts/fix_discord_bot.py

# Check for issues without fixing them
python scripts/fix_discord_bot.py --check-only

# Specify a custom path to the Discord bot file
python scripts/fix_discord_bot.py --path path/to/discord_bot.py

# Show detailed output
python scripts/fix_discord_bot.py --verbose
```

## What These Tools Fix

### Industry-Standard Tools

- **ruff**: Fast Python linter that checks for various issues and can automatically fix many of them
- **black**: Code formatter that enforces a consistent style
- **isort**: Import sorter that organizes imports according to PEP 8
- **autoflake**: Removes unused imports and variables

### General WDBX Linter (`wdbx_linter.py`)

- Indentation issues
- String-related issues (unterminated strings, quotes)
- Import-related issues (duplicate imports, wildcard imports)
- Try/except blocks without exception handling
- Missing docstrings for functions and classes

### Discord Bot Fixer (`fix_discord_bot.py`)

- Indentation issues specific to the Discord bot plugin
- String termination issues in command help text
- Try/except blocks without exception handling
- Command registration issues (duplicates)

## Development

To add a new linter or fixer:

1. Create a new class in `scripts/wdbx_linter.py` that inherits from `LintFixer`
2. Implement the `fix()` method to address specific issues
3. Add your new fixer to the list in the `main()` function

To add support for a new external tool:

1. Add configuration for the tool in `lint_config.py`
2. Create a function to run the tool in `integrated_linter.py`
3. Add the tool to the `all_tools` list in the `main()` function

## Tool Integration Notes

- The integrated linter handles various tool names and maps them to the appropriate linter. For example, you can use any of the following as tool names: `wdbx_linters`, `wdbx_linter`, `discord_bot_fixer`, or `discord`.
- The script will automatically detect whether to use the Discord bot fixer based on the target path. If you're targeting a directory or file related to the Discord bot, the fixer will be run automatically.

## Known Limitations

- These tools use regular expressions to identify and fix patterns, which may not catch all issues or might produce false positives in complex code
- The linters are designed to fix common issues but won't address all linting concerns (e.g., they don't enforce style guides like PEP 8)
- For more comprehensive linting, consider using additional tools like pylint, flake8, or black 