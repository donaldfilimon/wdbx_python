#!/usr/bin/env python3
"""
Simplified WDBX Linting Tool - Focuses on fixing specific issues in wdbx_plugins/discord_bot.py
"""

import logging
import os
import re
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fix_discord_bot(file_path):
    """Fix common issues in the Discord bot implementation."""
    logger.info(f"Processing Discord bot file: {file_path}")

    try:
        # Read the file
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        # Store original for comparison
        original_content = content

        # Fix 1: String termination issues in help text
        # Example: help="Text"" -> help="Text"
        content = re.sub(r'(help=")([^"]*)"(")', r'\1\2"', content)

        # Fix 2: Missing indentation in has_permissions decorator
        content = re.sub(
            r"(def has_permissions\(\*args, \*\*kwargs\):\n\s+def decorator\(func\):\n\s+return func\n)(\s+)return decorator",
            r"\1\2    return decorator",
            content,
        )

        # Check if content was modified
        if content != original_content:
            # Write back the modified content
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            logger.info(f"Fixed issues in {file_path}")
            return True
        else:
            logger.info(f"No issues to fix in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python simple_lint.py <path/to/discord_bot.py>")
        return 1

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1

    success = fix_discord_bot(file_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
