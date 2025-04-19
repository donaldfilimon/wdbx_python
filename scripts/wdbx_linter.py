#!/usr/bin/env python3
"""
WDBX Unified Linting Tool

This script provides a comprehensive linting solution for the WDBX codebase.
It combines functionality from multiple existing linting scripts and adds
new capabilities to address common linting issues.

Usage:
    python scripts/wdbx_linter.py [--target TARGET] [--fix] [--verbose]
"""

import argparse
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Set, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Define common linting issues and fixes
class LintFixer:
    """Base class for lint fixers."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def should_apply(self, file_path: str) -> bool:
        """
        Determine if this fixer should be applied to the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this fixer should be applied, False otherwise
        """
        return file_path.endswith(".py")

    def fix(self, content: str) -> str:
        """
        Apply fixes to the content.

        Args:
            content: Original file content

        Returns:
            Fixed content
        """
        return content


class IndentationFixer(LintFixer):
    """Fix common indentation issues."""

    def __init__(self):
        super().__init__("indentation", "Fix common indentation issues")

    def fix(self, content: str) -> str:
        """
        Fix common indentation issues.

        Args:
            content: Original file content

        Returns:
            Content with fixed indentation
        """
        # Fix function and class indentation
        content = re.sub(
            r"^(\s*)def (\w+)\(.*\):\n\1(\S)",
            r"\1def \2\3:\n\1    \3",
            content,
            flags=re.MULTILINE
        )

        # Fix indentation in nested blocks
        content = re.sub(
            r"(\n\s+)(if|for|while|try|with).*:\n\1(\S)",
            r"\1\2:\n\1    \3",
            content,
            flags=re.MULTILINE,
        )

        return content


class StringFixer(LintFixer):
    """Fix string-related issues."""

    def __init__(self):
        super().__init__("strings", "Fix string-related issues")

    def fix(self, content: str) -> str:
        """
        Fix string-related issues.

        Args:
            content: Original file content

        Returns:
            Content with fixed string issues
        """
        # Fix unterminated strings (more conservative approach to avoid false positives)
        content = re.sub(
            r'(["\']{1,3})(.*?)\1\1',
            r"\1\2\1",
            content
        )

        # Fix incorrect string concatenation
        content = re.sub(
            r'(["\'])\s*\+\s*(["\'])',
            r"\1\2",
            content
        )

        return content


class ImportFixer(LintFixer):
    """Fix import-related issues."""

    def __init__(self):
        super().__init__("imports", "Fix import-related issues")

    def fix(self, content: str) -> str:
        """
        Fix import-related issues.

        Args:
            content: Original file content

        Returns:
            Content with fixed import issues
        """
        # Fix wildcard imports
        content = re.sub(
            r"from\s+(\S+)\s+import\s+\*",
            r"# TODO: Replace wildcard import with specific imports\nfrom \1 import *",
            content,
        )

        # Fix duplicate imports
        import_pattern = r"(from\s+\S+\s+import\s+.*|import\s+.*)"
        imports = re.findall(import_pattern, content, re.MULTILINE)
        seen_imports: Set[str] = set()
        unique_imports: List[str] = []

        for imp in imports:
            if imp not in seen_imports:
                seen_imports.add(imp)
                unique_imports.append(imp)

        # Replace imports only if there were duplicates
        if len(imports) != len(unique_imports):
            for old_imp in imports:
                if old_imp not in unique_imports:
                    content = content.replace(old_imp, f"# Removed duplicate: {old_imp}", 1)

        return content


class TryExceptFixer(LintFixer):
    """Fix try/except related issues."""

    def __init__(self):
        super().__init__("try_except", "Fix try/except related issues")

    def fix(self, content: str) -> str:
        """
        Fix try/except related issues.

        Args:
            content: Original file content

        Returns:
            Content with fixed try/except blocks
        """
        # Find try blocks without except using a clearer regex pattern
        try_pattern = r"(\s+)try:\n((?:\1\s+.*\n)+)(?!\1except)"

        # Add except blocks where missing
        try_blocks = list(re.finditer(try_pattern, content))

        # Process matches in reverse to avoid offset issues when replacing
        for match in reversed(try_blocks):
            indent = match.group(1)
            block = match.group(0)
            except_block = (
                f'{indent}except Exception as e:\n'
                f'{indent}    logger.error(f"Error: {{e}}", exc_info=True)\n'
            )
            content = content.replace(block, block + except_block)

        return content


class DocstringFixer(LintFixer):
    """Fix docstring-related issues."""

    def __init__(self):
        super().__init__("docstrings", "Fix docstring-related issues")

    def fix(self, content: str) -> str:
        """
        Fix docstring-related issues including missing docstrings for classes and functions.

        Args:
            content: Original file content

        Returns:
            Content with fixed docstrings
        """
        # Fix missing docstrings for functions with improved pattern
        function_pattern = r'^(\s*)def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*\w+)?:\s*\n(?!\1\s*["\'])(\1\s+\S)'
        content = re.sub(
            function_pattern,
            lambda m: (
                f'{m.group(1)}def {m.group(2)}({m.group(3)}):\n'
                f'{m.group(1)}    """Function {m.group(2)}.\n\n'
                f'{m.group(1)}    Args:\n'
                f'{self._get_args_docstring(m.group(3), m.group(1))}'
                f'{m.group(1)}    """\n'
                f'{m.group(4)}'
            ),
            content,
            flags=re.MULTILINE,
        )

        # Fix missing docstrings for classes with improved pattern
        class_pattern = r'^(\s*)class\s+(\w+)(?:\s*\(\s*.*?\s*\))?:\s*\n(?!\1\s*["\'])(\1\s+\S)'
        content = re.sub(
            class_pattern,
            r'\1class \2...:\n\1    """Class \2."""\n\3',
            content,
            flags=re.MULTILINE
        )

        return content

    def _get_args_docstring(self, args_str: str, indent: str) -> str:
        """
        Generate docstring for function arguments.

        Args:
            args_str: String containing function arguments
            indent: Indentation string

        Returns:
            Formatted docstring for arguments
        """
        if not args_str.strip() or args_str.strip() == "self":
            return ""

        result = ""
        args = [arg.strip() for arg in args_str.split(",")]

        for arg in args:
            # Skip self and empty args
            if arg == "self" or not arg:
                continue

            # Strip type annotations and default values
            arg_name = arg.split(":")[0].split("=")[0].strip()
            result += f'{indent}        {arg_name}: Description of {arg_name}\n'

        if result:
            result += f'{indent}\n'
            result += f'{indent}    Returns:\n'
            result += f'{indent}        Description of return value\n{indent}\n'

        return result


class DiscordBotFixer(LintFixer):
    """Special fixer for Discord bot implementation."""

    def __init__(self):
        super().__init__("discord_bot", "Fix Discord bot-specific issues")

    def should_apply(self, file_path: str) -> bool:
        """
        Check if this is a Discord bot file.

        Args:
            file_path: Path to the file

        Returns:
            True if this is a Discord bot file
        """
        return "discord_bot.py" in file_path

    def fix(self, content: str) -> str:
        """
        Fix Discord bot-specific issues.

        Args:
            content: Original file content

        Returns:
            Fixed content
        """
        # Fix indentation in has_permissions decorator
        content = re.sub(
            r"(def has_permissions\(\*args, \*\*kwargs\):\n\s+def decorator\(func\):\n\s+return func\n)(\s+)return decorator",
            r"\1\2    return decorator",
            content,
        )

        # Fix string termination issues in command help text
        content = re.sub(r'(help=")([^"]*)"(")', r'\1\2"', content)

        # Fix try blocks without except clauses
        try_patterns: List[Tuple[str, str]] = [
            (
                r'@(self\.command|commands\.command)\(name="(\w+)", help="([^"]*)"\)\n\s+async def \2\(ctx(, [^)]+)?\):\n\s+""".*"""\n\s+(if .+:\n\s+.+\n\s+return\n\s+)?\n\s+try:',
                r'@\1(name="\2", help="\3")\n        async def \2(ctx\4):\n            """.*"""\n            \5\n            try:',
            ),
        ]

        for pattern, replacement in try_patterns:
            content = re.sub(pattern, replacement, content)

        # Ensure try blocks have except clauses - better approach with reversed matches
        try_blocks = list(re.finditer(r"(\s+)try:\n((?:\1\s+.*\n)+)(?!\1except)", content))

        # Process matches in reverse to avoid offset issues
        for match in reversed(try_blocks):
            indent = match.group(1)
            block = match.group(0)
            except_block = (
                f'{indent}except Exception as e:\n'
                f'{indent}    logger.error(f"Error: {{e}}", exc_info=True)\n'
                f'{indent}    await ctx.send(f"❌ An error occurred: {{str(e)}}")\n'
            )
            content = content.replace(block, block + except_block)

        return content


def fix_file(
    file_path: str, fixers: List[LintFixer], apply_fix: bool = False, verbose: bool = False
) -> bool:
    """
    Apply all relevant fixers to a file.

    Args:
        file_path: Path to the file to fix
        fixers: List of fixers to apply
        apply_fix: Whether to actually apply the fixes or just do a dry run
        verbose: Whether to show verbose output

    Returns:
        True if the operation was successful, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    try:
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        original_content = content
        file_modified = False

        # Apply each relevant fixer
        for fixer in fixers:
            if fixer.should_apply(file_path):
                if verbose:
                    logger.info(f"Applying {fixer.name} fixer to {file_path}")

                fixed_content = fixer.fix(content)

                # Check if this specific fixer changed anything
                if fixed_content != content:
                    file_modified = True
                    if verbose:
                        logger.info(f"  - {fixer.name} fixer made changes")

                content = fixed_content

        # Check if overall content was modified
        if file_modified:
            if apply_fix:
                try:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(content)
                    logger.info(f"Fixed issues in {file_path}")
                except (IOError, PermissionError) as e:
                    logger.error(f"Unable to write to {file_path}: {e}")
                    return False
            else:
                logger.info(f"Issues found in {file_path} (dry run, no changes made)")
            return True
        else:
            if verbose:
                logger.info(f"No issues to fix in {file_path}")
            return True

    except UnicodeDecodeError:
        logger.error(f"Error: {file_path} is not a valid UTF-8 text file")
        return False
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        if verbose:
            logger.exception(e)
        return False


def find_python_files(directory: str, exclude_dirs: Optional[Set[str]] = None) -> List[str]:
    """
    Find all Python files in a directory and its subdirectories.

    Args:
        directory: The directory to search
        exclude_dirs: Set of directory names to exclude

    Returns:
        List of paths to Python files
    """
    if exclude_dirs is None:
        exclude_dirs = {
            "__pycache__",
            "venv",
            ".venv",
            ".git",
            ".github",
            "build",
            "dist",
            ".pytest_cache",
        }

    python_files: List[str] = []
    directory_path = Path(directory)

    try:
        for file_path in directory_path.rglob("*.py"):
            # Check if any parent directory should be excluded
            if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                continue

            python_files.append(str(file_path))
    except Exception as e:
        logger.error(f"Error finding Python files: {e}")

    return python_files


def try_import_existing_fixers() -> List[LintFixer]:
    """
    Try to import existing lint fixing modules.

    Returns:
        List of additional fixers found
    """
    additional_fixers: List[LintFixer] = []

    # Look for lint.py and simple_lint.py in known locations
    script_paths = [
        "scripts/lint.py",
        "scripts/simple_lint.py",
        "wdbx_plugins/fix_discord_bot.py",
        "fix_discord_bot.py",
    ]

    for script_path in script_paths:
        if os.path.exists(script_path):
            try:
                # Load module dynamically
                module_name = os.path.basename(script_path).replace(".py", "")
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not create module spec for {script_path}")
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module  # Prevent import cycles
                spec.loader.exec_module(module)

                # Look for fix_discord_bot function
                if hasattr(module, "fix_discord_bot"):
                    logger.info(f"Found existing fixer in {script_path}")

                    # Create a wrapper fixer
                    class ExistingDiscordBotFixer(LintFixer):
                        def __init__(self, func, source_path):
                            super().__init__(
                                f"existing_discord_bot_{os.path.basename(source_path)}",
                                f"Existing Discord bot fixer from {source_path}",
                            )
                            self.func = func
                            self.source_path = source_path

                        def should_apply(self, file_path: str) -> bool:
                            return "discord_bot.py" in file_path

                        def fix(self, content: str) -> str:
                            # We can't directly use the function as it reads/writes files
                            # Instead, just note that we found it
                            logger.info(
                                f"Note: We found an existing Discord bot fixer in {self.source_path}."
                            )
                            logger.info(
                                "It's recommended to run it separately as it directly modifies files."
                            )
                            return content

                    additional_fixers.append(
                        ExistingDiscordBotFixer(module.fix_discord_bot, script_path)
                    )

            except ImportError as e:
                logger.warning(f"Error importing existing fixer from {script_path}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error importing from {script_path}: {e}")

    return additional_fixers


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="WDBX Unified Linting Tool")
    parser.add_argument(
        "--target",
        "-t",
        default=".",
        help="Target directory or file to process (default: current directory)",
    )
    parser.add_argument(
        "--fix",
        "-f",
        action="store_true",
        help="Apply fixes (without this flag, only reports issues)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more detailed output")
    parser.add_argument(
        "--exclude",
        "-e",
        action="append",
        help="Directories to exclude (can be specified multiple times)"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create fixers
    fixers = [
        IndentationFixer(),
        StringFixer(),
        ImportFixer(),
        TryExceptFixer(),
        DocstringFixer(),
        DiscordBotFixer(),
    ]

    # Try to import existing fixers
    additional_fixers = try_import_existing_fixers()
    fixers.extend(additional_fixers)

    exclude_dirs = {
        "__pycache__",
        "venv",
        ".venv",
        ".git",
        ".github",
        "build",
        "dist",
        ".pytest_cache",
    }

    # Add user-specified exclusions
    if args.exclude:
        exclude_dirs.update(args.exclude)

    # Process target
    if os.path.isfile(args.target):
        # Process single file
        success = fix_file(args.target, fixers, args.fix, args.verbose)
        return 0 if success else 1

    elif os.path.isdir(args.target):
        # Process directory
        python_files = find_python_files(args.target, exclude_dirs)

        if not python_files:
            logger.warning(f"No Python files found in {args.target}")
            return 0

        logger.info(f"Found {len(python_files)} Python files to process")

        success_count = 0
        failure_count = 0

        for file_path in python_files:
            success = fix_file(file_path, fixers, args.fix, args.verbose)
            if success:
                success_count += 1
            else:
                failure_count += 1

        logger.info(
            f"Processed {success_count + failure_count} files: {success_count} succeeded, {failure_count} failed"
        )
        return 0 if failure_count == 0 else 1

    else:
        logger.error(f"Target not found: {args.target}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
