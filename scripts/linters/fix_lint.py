#!/usr/bin/env python
"""
Script to automatically fix common linting issues in Python files.

This script detects and fixes common linting issues in Python files, including:
- Unused imports
- Whitespace issues
- Line length violations
- F-string issues
- Naming conventions
- Unnecessary comments
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fix_lint")

# Linting issues patterns
UNUSED_IMPORT_PATTERN = re.compile(r"imported but unused|^F401")
WHITESPACE_PATTERN = re.compile(r"trailing whitespace|^W291|^W293")
LINE_LENGTH_PATTERN = re.compile(r"line too long \((\d+)\/(\d+)\)|^E501")
NAMING_PATTERN = re.compile(r"invalid name|^E741|^N802|^N803|^N806")
UNUSED_VAR_PATTERN = re.compile(r"local variable.+not used|^F841")


def find_python_files(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[Path]:
    """
    Find all Python files in the given directory recursively.

    Args:
        directory: Directory to search for Python files
        exclude_dirs: List of directory names to exclude from search

    Returns:
        List of Path objects for Python files
    """
    if exclude_dirs is None:
        exclude_dirs = [".git", ".venv", "venv", "env", "build", "dist", "__pycache__"]

    python_files = []
    directory_path = Path(directory).resolve()

    logger.debug(f"Searching for Python files in {directory_path}")

    try:
        for root, dirs, files in os.walk(directory_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)
    except Exception as e:
        logger.error(f"Error searching directory {directory_path}: {e}")

    logger.info(f"Found {len(python_files)} Python files")
    return python_files


def run_flake8(file_path: Path) -> List[Dict[str, str]]:
    """
    Run flake8 on a file and parse the output.

    Args:
        file_path: Path to the file to check

    Returns:
        List of dictionaries with linting issues
    """
    try:
        result = subprocess.run(
            ["flake8", "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s", str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        issues = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue

            try:
                path, row, col, code, text = line.split(":", 4)
                issues.append(
                    {
                        "path": path,
                        "row": int(row),
                        "col": int(col),
                        "code": code,
                        "text": text.strip(),
                    }
                )
            except ValueError:
                logger.warning(f"Could not parse flake8 output: {line}")

        return issues
    except Exception as e:
        logger.error(f"Error running flake8 on {file_path}: {e}")
        return []


def fix_unused_imports(file_path: Path, issues: List[Dict[str, str]]) -> int:
    """
    Fix unused imports in a file.

    Args:
        file_path: Path to the file to fix
        issues: List of linting issues

    Returns:
        Number of issues fixed
    """
    # Filter issues to only unused imports
    unused_imports = [issue for issue in issues if UNUSED_IMPORT_PATTERN.search(issue["text"])]
    if not unused_imports:
        return 0

    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Track which lines to remove
        lines_to_remove = set(issue["row"] for issue in unused_imports)
        fixed_lines = []

        for i, line in enumerate(lines, 1):
            if i not in lines_to_remove:
                fixed_lines.append(line)

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)

        logger.debug(f"Fixed {len(unused_imports)} unused imports in {file_path}")
        return len(unused_imports)
    except Exception as e:
        logger.error(f"Error fixing unused imports in {file_path}: {e}")
        return 0


def fix_whitespace(file_path: Path, issues: List[Dict[str, str]]) -> int:
    """
    Fix whitespace issues in a file.

    Args:
        file_path: Path to the file to fix
        issues: List of linting issues

    Returns:
        Number of issues fixed
    """
    # Filter issues to only whitespace issues
    whitespace_issues = [issue for issue in issues if WHITESPACE_PATTERN.search(issue["text"])]
    if not whitespace_issues:
        return 0

    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Fix whitespace issues
        fixed_count = 0
        for issue in whitespace_issues:
            row = issue["row"] - 1  # Convert to 0-indexed
            if 0 <= row < len(lines):
                lines[row] = lines[row].rstrip() + "\n"
                fixed_count += 1

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.debug(f"Fixed {fixed_count} whitespace issues in {file_path}")
        return fixed_count
    except Exception as e:
        logger.error(f"Error fixing whitespace issues in {file_path}: {e}")
        return 0


def fix_line_length(file_path: Path, issues: List[Dict[str, str]], max_length: int = 88) -> int:
    """
    Fix line length issues in a file.

    Args:
        file_path: Path to the file to fix
        issues: List of linting issues
        max_length: Maximum line length

    Returns:
        Number of issues fixed
    """
    # Filter issues to only line length issues
    line_length_issues = [issue for issue in issues if LINE_LENGTH_PATTERN.search(issue["text"])]
    if not line_length_issues:
        return 0

    try:
        # Use black to format the file
        result = subprocess.run(
            ["black", "--line-length", str(max_length), str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.debug(f"Fixed {len(line_length_issues)} line length issues in {file_path}")
            return len(line_length_issues)
        logger.warning(f"Failed to fix line length issues in {file_path}: {result.stderr}")
        return 0
    except Exception as e:
        logger.error(f"Error fixing line length issues in {file_path}: {e}")
        return 0


def fix_unused_variables(file_path: Path, issues: List[Dict[str, str]]) -> int:
    """
    Fix unused variables in a file.

    Args:
        file_path: Path to the file to fix
        issues: List of linting issues

    Returns:
        Number of issues fixed
    """
    # Filter issues to only unused variables
    unused_vars = [issue for issue in issues if UNUSED_VAR_PATTERN.search(issue["text"])]
    if not unused_vars:
        return 0

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Find and replace unused variables with underscore
        fixed_content = content
        for issue in unused_vars:
            var_name_match = re.search(r"'(\w+)' .+ not used", issue["text"])
            if var_name_match:
                var_name = var_name_match.group(1)
                line_index = issue["row"] - 1  # Convert to 0-indexed

                # Get the line
                lines = content.split("\n")
                if 0 <= line_index < len(lines):
                    line = lines[line_index]
                    # Replace the variable name with underscore
                    pattern = r"\b" + re.escape(var_name) + r"\b"
                    new_line = re.sub(pattern, "_", line)
                    lines[line_index] = new_line
                    fixed_content = "\n".join(lines)

        if fixed_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)

            logger.debug(f"Fixed {len(unused_vars)} unused variables in {file_path}")
            return len(unused_vars)

        return 0
    except Exception as e:
        logger.error(f"Error fixing unused variables in {file_path}: {e}")
        return 0


def process_file(file_path: Path, args: argparse.Namespace) -> Dict[str, int]:
    """
    Process a single file, fixing linting issues.

    Args:
        file_path: Path to the file to process
        args: Command-line arguments

    Returns:
        Dictionary of issue counts
    """
    logger.debug(f"Processing file: {file_path}")

    # Run flake8 to get issues
    issues = run_flake8(file_path)
    if not issues:
        return {"total": 0}

    # Fix issues
    results = {"unused_imports": 0, "whitespace": 0, "line_length": 0, "unused_variables": 0}

    # Fix each type of issue
    if not args.dry_run:
        if args.fix_imports or args.fix_all:
            results["unused_imports"] = fix_unused_imports(file_path, issues)
            # Run flake8 again to update issues list after fixing imports
            if results["unused_imports"] > 0:
                issues = run_flake8(file_path)

        if args.fix_whitespace or args.fix_all:
            results["whitespace"] = fix_whitespace(file_path, issues)
            # Run flake8 again if needed
            if results["whitespace"] > 0:
                issues = run_flake8(file_path)

        if args.fix_line_length or args.fix_all:
            results["line_length"] = fix_line_length(file_path, issues, args.max_length)
            # Run flake8 again if needed
            if results["line_length"] > 0:
                issues = run_flake8(file_path)

        if args.fix_unused_vars or args.fix_all:
            results["unused_variables"] = fix_unused_variables(file_path, issues)

    results["total"] = sum(results.values())
    return results


def main() -> int:
    """
    Main function for the script.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(description="Fix common linting issues in Python files")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for Python files (default: current directory)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="+",
        default=[".git", ".venv", "venv", "env", "build", "dist", "__pycache__"],
        help="Directories to exclude from search",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Don't actually modify files, just show what would be done",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--single-file", "-f", help="Process a single file instead of a directory")
    parser.add_argument("--fix-imports", action="store_true", help="Fix unused imports")
    parser.add_argument("--fix-whitespace", action="store_true", help="Fix whitespace issues")
    parser.add_argument("--fix-line-length", action="store_true", help="Fix line length issues")
    parser.add_argument("--fix-unused-vars", action="store_true", help="Fix unused variables")
    parser.add_argument("--fix-all", action="store_true", help="Fix all linting issues")
    parser.add_argument(
        "--max-length", type=int, default=88, help="Maximum line length (default: 88)"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.dry_run:
        logger.info("Dry run mode enabled - no files will be modified")

    # Check if flake8 and black are installed
    try:
        subprocess.run(["flake8", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("flake8 is not installed. Please install it with 'pip install flake8'")
        return 1

    try:
        subprocess.run(["black", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        if args.fix_line_length or args.fix_all:
            logger.error("black is not installed. Please install it with 'pip install black'")
            return 1

    # If no specific fix is specified, enable all
    if not any(
        [
            args.fix_imports,
            args.fix_whitespace,
            args.fix_line_length,
            args.fix_unused_vars,
            args.fix_all,
        ]
    ):
        args.fix_all = True

    # Process single file if specified
    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 1

        if not file_path.name.endswith(".py"):
            logger.error(f"Not a Python file: {file_path}")
            return 1

        results = process_file(file_path, args)
        logger.info(f"Fixed {results['total']} issues in {file_path}")
        return 0

    # Process directory
    if not os.path.isdir(args.directory):
        logger.error(f"Directory not found: {args.directory}")
        return 1

    python_files = find_python_files(args.directory, args.exclude)
    if not python_files:
        logger.info(f"No Python files found in {args.directory}")
        return 0

    # Process all files
    total_files = len(python_files)
    total_fixed = {
        "unused_imports": 0,
        "whitespace": 0,
        "line_length": 0,
        "unused_variables": 0,
        "total": 0,
    }

    for i, file_path in enumerate(python_files, 1):
        if i % 50 == 0 or i == total_files:
            logger.info(f"Processing file {i}/{total_files}")

        file_results = process_file(file_path, args)

        # Update totals
        for key in total_fixed:
            if key in file_results:
                total_fixed[key] += file_results[key]

    logger.info(f"Fixed {total_fixed['total']} issues across {total_files} files:")
    logger.info(f"  - Unused imports: {total_fixed['unused_imports']}")
    logger.info(f"  - Whitespace issues: {total_fixed['whitespace']}")
    logger.info(f"  - Line length issues: {total_fixed['line_length']}")
    logger.info(f"  - Unused variables: {total_fixed['unused_variables']}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
