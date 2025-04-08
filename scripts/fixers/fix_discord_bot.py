#!/usr/bin/env python3
"""
Discord Bot Plugin Fixer

This script specifically targets and fixes common issues in the Discord bot plugin.

Usage:
    python fix_discord_bot.py [options]

Options:
    --check-only     Check for issues without fixing them
    --path PATH      Path to the Discord bot file
    --verbose        Show more detailed output
"""

import argparse
import logging
import os
import re
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Target file paths to check
DISCORD_BOT_PATHS = ["src/wdbx/plugins/discord_bot.py", "wdbx_plugins/discord_bot.py"]


def find_discord_bot_file(custom_path=None):
    """Find the discord_bot.py file in the project."""
    # First check custom path if provided
    if custom_path:
        if os.path.exists(custom_path):
            return custom_path
        logger.warning(f"Custom path '{custom_path}' not found. Falling back to default paths.")

    # Check default paths
    for path in DISCORD_BOT_PATHS:
        if os.path.exists(path):
            return path

    # If not found in predefined paths, try to locate it
    for root, _, files in os.walk("."):
        for file in files:
            if file == "discord_bot.py":
                return os.path.join(root, file)

    return None


def fix_indentation_issues(content):
    """Fix indentation issues in the Discord bot plugin."""
    # Fix indentation in command methods
    content = re.sub(
        r"(\s+)@commands\.command\(.*\)\n\1async def (\w+)\(.*\):\n\1(\S)",
        r"\1@commands.command(...)\n\1async def \2(...):\n\1    \3",
        content,
    )

    # Fix indentation in error handlers
    content = re.sub(
        r"(\s+)@(\w+)\.error\n\1async def (\w+)_error\(.*\):\n\1(\S)",
        r"\1@\2.error\n\1async def \3_error(...):\n\1    \4",
        content,
    )

    # Fix indentation in nested blocks
    content = re.sub(r"(\s+)(if|for|while|try|with).*:\n\1([^\s])", r"\1\2...:\n\1    \3", content)

    # Fix indentation in has_permissions decorator
    content = re.sub(
        r"(\s+)def has_permissions\(\*args, \*\*kwargs\):\s*\n\1\s+def decorator\(func\):\s*\n\1\s+return func\s*\n\1return decorator",
        r"\1def has_permissions(*args, **kwargs):\n\1    def decorator(func):\n\1        return func\n\1    return decorator",
        content,
    )

    return content


def fix_try_except_issues(content):
    """Fix try/except issues in the Discord bot plugin."""
    # Find try blocks without except
    modified_content = content

    # Iterate through common patterns for try blocks without except
    try_block_patterns = [
        r"(\s+)try:\n((?:\1\s+.*\n)+)(?!\1except)",
        r"(\s+)try:\s*\n((?:\1\s+.*\n)*)(?!\1\s*except)",
    ]

    for pattern in try_block_patterns:
        try_matches = list(re.finditer(pattern, modified_content))

        # Process matches in reverse order to avoid messing up string positions
        for match in reversed(try_matches):
            indent = match.group(1)
            try_content = match.group(2)

            # Check if 'ctx' is used in the try block to determine if we need to add await ctx.send
            has_ctx = "ctx" in try_content

            if has_ctx:
                except_block = f'{indent}except Exception as e:\n{indent}    logger.error(f"Error: {{e}}", exc_info=True)\n{indent}    await ctx.send(f"❌ An error occurred: {{str(e)}}")\n'
            else:
                except_block = f'{indent}except Exception as e:\n{indent}    logger.error(f"Error: {{e}}", exc_info=True)\n'

            modified_content = (
                modified_content[: match.end()] + except_block + modified_content[match.end() :]
            )

    return modified_content


def fix_string_issues(content):
    """Fix string-related issues in the Discord bot plugin."""
    # Fix unterminated strings in command help text
    content = re.sub(
        r'(help=["\']{1,3})([^"\']*)(["\']{1,4})',
        lambda m: m.group(1) + m.group(2) + m.group(1),
        content,
    )

    # Fix string quotes that don't match
    content = re.sub(
        r'(["\'])([^"\']*)(["\']\s*\+\s*["\']\s*)(.*?)(["\']{1,3})',
        lambda m: m.group(1) + m.group(2) + m.group(4) + m.group(1),
        content,
    )

    return content


def fix_command_registration_issues(content):
    """Fix issues with command registration in the Discord bot plugin."""
    # Find duplicate command registrations
    command_defs = re.findall(r'@commands\.command\(name="(\w+)"', content)
    command_counts = {}

    for cmd in command_defs:
        command_counts[cmd] = command_counts.get(cmd, 0) + 1

    for cmd, count in command_counts.items():
        if count > 1:
            # Get first occurrence
            first_match = re.search(
                r'@commands\.command\(name="'
                + cmd
                + r'".*?\n(?:.*?\n)*?(?=\s*@commands\.command|\s*def\s|$)',
                content,
                re.DOTALL,
            )
            if first_match:
                first_match.group(0)

                # Find and comment out subsequent occurrences
                pattern = (
                    r'(@commands\.command\(name="'
                    + cmd
                    + r'".*?\n(?:.*?\n)*?(?=\s*@commands\.command|\s*def\s|$))'
                )
                matches = list(re.finditer(pattern, content, re.DOTALL))

                # Skip the first occurrence
                for match in matches[1:]:
                    replacement = "# Duplicate command - commented out\n# " + match.group(
                        0
                    ).replace("\n", "\n# ")
                    content = content[: match.start()] + replacement + content[match.end() :]

    return content


def check_discord_bot_issues(discord_bot_path, verbose=False):
    """Check for issues in the Discord bot plugin without fixing them."""
    try:
        # Read the file
        with open(discord_bot_path, encoding="utf-8") as file:
            content = file.read()

        issues_found = False

        # Check indentation issues
        modified_content = fix_indentation_issues(content)
        if modified_content != content:
            issues_found = True
            if verbose:
                logger.warning(f"Found indentation issues in {discord_bot_path}")

        # Check try/except issues
        modified_content = fix_try_except_issues(content)
        if modified_content != content:
            issues_found = True
            if verbose:
                logger.warning(f"Found try/except issues in {discord_bot_path}")

        # Check string issues
        modified_content = fix_string_issues(content)
        if modified_content != content:
            issues_found = True
            if verbose:
                logger.warning(f"Found string issues in {discord_bot_path}")

        # Check command registration issues
        modified_content = fix_command_registration_issues(content)
        if modified_content != content:
            issues_found = True
            if verbose:
                logger.warning(f"Found command registration issues in {discord_bot_path}")

        if not issues_found:
            logger.info(f"No issues found in {discord_bot_path}")
            return True
        else:
            logger.warning(f"Issues found in {discord_bot_path}")
            return False

    except Exception as e:
        logger.error(f"Error checking Discord bot: {e}")
        return False


def fix_discord_bot(discord_bot_path=None, check_only=False, verbose=False):
    """Fix common issues in the Discord bot plugin."""
    # Find the Discord bot file
    bot_path = find_discord_bot_file(discord_bot_path)

    if not bot_path:
        logger.error("Discord bot plugin file not found!")
        return False

    logger.info(f"Found Discord bot plugin at: {bot_path}")

    # If check only mode, just check for issues
    if check_only:
        return check_discord_bot_issues(bot_path, verbose)

    try:
        # Read the file
        with open(bot_path, encoding="utf-8") as file:
            content = file.read()

        # Keep track of original content to check if modified
        original_content = content

        # Fix indentation issues
        content = fix_indentation_issues(content)
        if verbose and content != original_content:
            logger.info("Fixed indentation issues")

        # Fix try/except issues
        original_after_indentation = content
        content = fix_try_except_issues(content)
        if verbose and content != original_after_indentation:
            logger.info("Fixed try/except issues")

        # Fix string issues
        original_after_try_except = content
        content = fix_string_issues(content)
        if verbose and content != original_after_try_except:
            logger.info("Fixed string issues")

        # Fix command registration issues
        original_after_strings = content
        content = fix_command_registration_issues(content)
        if verbose and content != original_after_strings:
            logger.info("Fixed command registration issues")

        # Write back only if modified
        if content != original_content:
            with open(bot_path, "w", encoding="utf-8") as file:
                file.write(content)
            logger.info(f"Successfully fixed issues in {bot_path}")
            return True
        else:
            logger.info(f"No issues to fix in {bot_path}")
            return True

    except Exception as e:
        logger.error(f"Error fixing Discord bot: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fix common issues in the Discord bot plugin.")
    parser.add_argument(
        "--check-only", action="store_true", help="Check for issues without fixing them"
    )
    parser.add_argument("--path", help="Path to the Discord bot file")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed output")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    success = fix_discord_bot(
        discord_bot_path=args.path, check_only=args.check_only, verbose=args.verbose
    )

    sys.exit(0 if success else 1)
