# WDBX Linting System

The WDBX project includes a comprehensive linting system to maintain code quality and consistency. This system combines industry-standard tools with custom fixers specifically designed for the WDBX codebase.

## Quick Start

The easiest way to run the full linting suite is to use the `fix_all_linter_issues.py` script in the project root:

```bash
python fix_all_linter_issues.py
```

This will:
1. Run all available linting tools
2. Fix common issues automatically
3. Generate a report of the changes made

## Available Linting Tools

### Industry-Standard Tools

The WDBX linting system integrates several popular Python linting and formatting tools:

- **ruff**: A fast Python linter that combines functionality from many popular linting tools, implemented in Rust
- **black**: An opinionated code formatter that enforces a consistent style
- **isort**: A utility that sorts imports alphabetically and automatically separates them into sections
- **autoflake**: A tool that removes unused imports and variables

### Custom WDBX Linters

In addition to industry-standard tools, WDBX includes custom linters designed to address specific issues in our codebase:

- **General WDBX Linter** (`wdbx_linter.py`): Fixes common issues like indentation, string formatting, and missing docstrings
- **Discord Bot Fixer** (`fix_discord_bot.py`): Addresses specific issues in the Discord bot plugin
- **Run Linters** (`run_linters.py`): Runs all custom linters and generates an HTML report

### Integrated Linter

The `integrated_linter.py` script combines all these tools into a single, comprehensive linting solution.

## How to Use

### Using the Integrated Linter

```bash
# Check for issues without fixing them
python scripts/integrated_linter.py

# Fix issues automatically
python scripts/integrated_linter.py --fix

# Target a specific directory or file
python scripts/integrated_linter.py --target src/wdbx/plugins

# Run only specific tools
python scripts/integrated_linter.py --tools ruff,black

# Install missing tools
python scripts/integrated_linter.py --install-missing

# Generate an HTML report (for WDBX custom linters)
python scripts/integrated_linter.py --report
```

### Using Individual Tools

You can also run individual linting tools directly:

```bash
# Run ruff
ruff check src/

# Run black
black src/

# Run isort
isort src/

# Run autoflake
autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive src/

# Run WDBX general linter
python scripts/wdbx_linter.py --target src/ --fix

# Run Discord bot fixer
python scripts/fix_discord_bot.py
```

## Configuration

Configuration for all linting tools is defined in two places:

1. **pyproject.toml**: Standard configuration for ruff, black, isort, and mypy
2. **scripts/lint_config.py**: Configuration for the integrated linter

### Common Settings

- **Line Length**: Set to 100 characters
- **Target Python Version**: Python 3.9
- **Excluded Directories**: Build artifacts, caches, and virtual environments

## Adding New Linting Rules

### Adding a Custom Fixer

To add a new custom fixer:

1. Create a new class in `scripts/wdbx_linter.py` that inherits from `LintFixer`
2. Implement the `fix()` method to address specific issues
3. Add your new fixer to the list in the `main()` function

Example:

```python
class MyNewFixer(LintFixer):
    """Fix my specific issue."""
    
    def __init__(self):
        super().__init__("my_fixer", "Fix my specific issue")
    
    def should_apply(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def fix(self, content: str) -> str:
        # Add your fix logic here
        return content
```

### Customizing External Tools

To customize an external tool's configuration:

1. Update the relevant section in `pyproject.toml`
2. Update the corresponding settings in `scripts/lint_config.py`

## Linting in CI/CD

The linting system is designed to be used in CI/CD pipelines. Add the following to your workflow:

```yaml
- name: Run linters
  run: python scripts/integrated_linter.py
```

To fail the build if linting issues are found, but not try to fix them automatically.

## Common Issues Fixed

### General Linter

- Indentation issues
- String termination issues (unterminated strings, mismatched quotes)
- Import-related issues (duplicate imports, wildcard imports)
- Missing try/except handlers
- Missing docstrings

### Discord Bot Fixer

- Indentation issues specific to the Discord bot
- String termination issues in command help text
- Missing exception handlers
- Command registration issues

### ruff

- Style issues (PEP 8 violations)
- Potential bugs
- Code simplification opportunities
- Unused imports and variables
- Type annotation issues

### black and isort

- Consistent code formatting
- Consistent import ordering

## Reports

The linting system can generate HTML reports showing:

- Summary of linting results
- Issues found and fixed
- Recommendations for manual fixes

Reports are saved in the `linting_reports` directory. 