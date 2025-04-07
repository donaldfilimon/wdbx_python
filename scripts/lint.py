#!/usr/bin/env python3
"""
WDBX Linting Tool - Fixes common linting issues across the codebase
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Callable, Optional, Pattern, Match, Tuple, Union

# Add a basic logger setup
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_discord_bot(file_path: str) -> bool:
    """Fix linter errors in Discord bot implementation."""
    logger.info(f"Applying Discord bot specific fixes to: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        original_content = content # Keep original for comparison

        # Fix string termination issues in command help text
        # Example: help="Some help text."" -> help="Some help text."
        content = re.sub(
            r'(name="[^"]+", help="[^"]*)"(")', # Pattern to find help="...""
            r'\1"', # Replace with help="..."
            content
        )

        # Fix indentation in the mock commands.has_permissions (if present)
        # This might be overly specific if the mock structure changed
        content = re.sub(
            r'(def has_permissions\(\*args, \*\*kwargs\):\n\s+def decorator\(func\):\n\s+return func\n)(\s+)return decorator',
            r'\1\2    return decorator', # Ensure correct indentation for the final return
            content
        )

        # Fix try statements immediately followed by non-except/finally blocks
        # Adds a basic except block
        content = re.sub(
            r'^(\s+try:)(\n\s+(?!except|finally)[^\n]+)', # Looks for try: not followed by except/finally
            r'\1\2\n\1    except Exception as e:\n\1        logger.error(f"Error: {e}", exc_info=True)\n\1        # Placeholder: Add appropriate error handling, e.g., await ctx.send(...)', # Add basic exception handler
            content,
            flags=re.MULTILINE
        )

        # Fix indentation issues common after previous edits (like command definitions)
        # Example: A line indented less than the previous line unexpectedly
        # This is a heuristic and might need refinement
        lines = content.split('\n')
        corrected_lines = []
        previous_indent = 0
        for line in lines:
            stripped_line = line.lstrip()
            if not stripped_line: # Keep empty lines
                corrected_lines.append(line)
                continue

            current_indent = len(line) - len(stripped_line)

            # Basic check for unexpected de-indentation (simplistic)
            if current_indent < previous_indent and not stripped_line.startswith(('except', 'finally', 'elif', 'else')):
                # Try to guess the correct indent based on the line before the previous one
                # This is complex and error-prone, better handled by formatters like Black/Ruff
                pass # Skipping auto-indent correction for now to avoid errors

            corrected_lines.append(line)
            if stripped_line: # Update previous indent only if line wasn't empty
                 previous_indent = current_indent
        content = '\n'.join(corrected_lines)


        # Write the fixed content back ONLY if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            logger.info(f"Applied fixes to {os.path.basename(file_path)}")
            return True
        else:
            logger.info(f"No changes needed for {os.path.basename(file_path)}")
            return True # Still success if no changes needed

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error fixing {os.path.basename(file_path)}: {str(e)}", exc_info=True)
        return False

def fix_python_file(file_path: str) -> bool:
    """Fix common Python linter errors in any Python file."""
    logger.info(f"Applying generic Python fixes to: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        original_content = content

        # Fix trailing whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

        # Fix missing whitespace after comma
        content = re.sub(r',([\w\(\[\{'"`])', r', \1', content)

        # Remove the problematic f-string regex patterns that were causing syntax errors
        # Instead, just check for and warn about potentially problematic f-strings
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "f\"" in line and "\"\"\"" not in line and "'" in line:
                logger.warning(f"Line {i+1} in {os.path.basename(file_path)} may have problematic quotes in f-string")
            if "f'" in line and "'''" not in line and "\"" in line:
                logger.warning(f"Line {i+1} in {os.path.basename(file_path)} may have problematic quotes in f-string")
                
        # Basic check for overly long lines (e.g., > 100 chars), logs warning
        # Auto-fixing long lines reliably requires AST parsing or formatters
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 100 and not line.strip().startswith('#'):
                 logger.warning(f"Line {i+1} in {os.path.basename(file_path)} is longer than 100 characters.")
                 # Placeholder: Could attempt basic splitting for simple cases

        # Write the fixed content back ONLY if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            logger.info(f"Applied fixes to {os.path.basename(file_path)}")
            return True
        else:
            logger.info(f"No changes needed for {os.path.basename(file_path)}")
            return True # Still success

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error fixing {os.path.basename(file_path)}: {str(e)}", exc_info=True)
        return False

def main():
    """Main entry point for the linting tool."""
    parser = argparse.ArgumentParser(description="WDBX Linting and Auto-fix Tool")
    parser.add_argument('--target', '-t', choices=['all', 'discord', 'visualization'],
                       default='all', help='Target specific files or fix all Python files')
    parser.add_argument('--dir', '-d', default='.',
                       help='Directory to search for files (default: current directory)')
    parser.add_argument('--apply', '-a', action='store_true',
                       help='Apply fixes (without this flag, only reports issues)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show more detailed output')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Map specific files to their fixers
    file_fixers = {
        "discord_bot.py": fix_discord_bot,
        # Add more specialized fixers here, e.g., "web_scraper.py": fix_web_scraper
    }

    fixed_count = 0
    error_count = 0
    processed_count = 0

    # Determine the root directory to scan
    scan_dir = args.dir
    if not os.path.isdir(scan_dir):
        logger.error(f"Directory not found: {scan_dir}")
        return 1

    logger.info(f"Scanning directory: {os.path.abspath(scan_dir)}")

    if args.target != 'all':
        # Fix specific plugin
        # Adjust path construction to look inside wdbx_plugins relative to scan_dir
        plugin_dir = os.path.join(scan_dir, 'wdbx_plugins')
        target_file_name = f"{args.target}_bot.py" if args.target == 'discord' else f"{args.target}.py"
        file_path = os.path.join(plugin_dir, target_file_name)

        if os.path.exists(file_path):
            logger.info(f"Processing specific target: {file_path}")
            processed_count = 1
            fixer_func = file_fixers.get(target_file_name, fix_python_file)

            if args.apply:
                success = fixer_func(file_path)
                if success:
                    fixed_count += 1
                else:
                    error_count += 1
            else:
                logger.info(f"Dry run: Would attempt to fix {file_path} with {fixer_func.__name__}")
        else:
            logger.error(f"Target file not found: {file_path}")
            return 1
    else:
        # Fix all Python files in the specified directory (recursively)
        logger.info(f"Processing all Python files in {os.path.abspath(scan_dir)}...")
        for root, _, files in os.walk(scan_dir):
            # Skip virtual environments and cache directories
            if os.path.basename(root) in (".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache") or ".git" in root.split(os.path.sep):
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    processed_count += 1
                    logger.debug(f"Considering file: {file_path}")

                    # Determine the correct fixer
                    base_name = os.path.basename(file)
                    fixer_func = file_fixers.get(base_name, fix_python_file)

                    if args.apply:
                        success = fixer_func(file_path)
                        if success:
                            # Increment count only if changes were actually made or no changes needed
                            # The functions return True in both cases if no error occurred
                            fixed_count += 1 # Consider renaming 'fixed_count' to 'successful_processing_count'
                        else:
                            error_count += 1
                    else:
                         logger.info(f"Dry run: Would attempt to fix {file_path} with {fixer_func.__name__}")

    # Final summary
    logger.info("--- Linting Summary ---")
    logger.info(f"Files Processed: {processed_count}")
    if args.apply:
        # Adjusting the success count interpretation
        successful_ops = processed_count - error_count
        logger.info(f"Successful Operations: {successful_ops}")
        logger.info(f"Errors Encountered: {error_count}")
        if error_count > 0:
            logger.warning("Some files could not be processed due to errors.")
    else:
        logger.info("Dry run complete. No changes were applied.")
        logger.info("Run with --apply to apply fixes.")

    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 