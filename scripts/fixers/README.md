# Fixer Scripts

This directory contains various fixer scripts used to address linting issues and code style problems across the codebase. 

## Key Fixers

- `fix_discord_bot.py` - Fixes linting issues in the Discord bot implementation
- `fix_all_linter_issues.py` - Main script to fix all linting issues
- `fix_line_lengths.py` - Handles line length issues
- `fix_unterminated_strings.py` - Fixes unterminated string issues
- `fix_web_scraper.py` - Fixes issues in the web scraper implementation

## Usage

Most of these scripts are meant to be run as one-off fixes for specific issues. They're organized here for historical reference and in case similar issues arise in the future.

To run a fixer script:

```bash
python scripts/fixers/script_name.py
```

## Integration with Linting Tools

These fixers can be integrated with linting tools through the scripts in the parent directory. 