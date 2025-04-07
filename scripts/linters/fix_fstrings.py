#!/usr/bin/env python
"""
Script to fix f-strings with missing placeholders or format specifiers.

This script scans Python files to identify and correct common f-string 
issues such as missing curly braces, improper variable references, and 
incorrect format specifiers.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fix_fstrings")

# Regular expressions for detecting and fixing f-strings
FSTRING_REGEX = re.compile(r'f["\'].*?["\']', re.DOTALL)
PLACEHOLDER_REGEX = re.compile(r"{([^{}]*)}")
VARIABLE_REGEX = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
FORMAT_SPEC_REGEX = re.compile(r"{([^{}]*?):(.*?)}")


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


def extract_variable_names(code: str) -> Set[str]:
    """
    Extract variable names from a code snippet.
    
    Args:
        code: Python code snippet
        
    Returns:
        Set of variable names found in the code
    """
    return set(VARIABLE_REGEX.findall(code))


def fix_fstring(fstring: str, variables: Set[str]) -> Tuple[str, bool]:
    """
    Fix an f-string by adding missing placeholders or correcting format specifiers.
    
    Args:
        fstring: The f-string to fix
        variables: Set of available variable names
        
    Returns:
        Tuple of (fixed_fstring, was_modified)
    """
    # Remove the 'f' prefix and quotes
    inner_string = fstring[2:-1]
    original = inner_string
    
    # Find all placeholders
    placeholders = PLACEHOLDER_REGEX.findall(inner_string)
    
    # Check for variables in the string that are not in placeholders
    for var in variables:
        # Skip if variable is already in a placeholder
        if var in placeholders:
            continue
        
        # Skip common keywords
        if var in {"self", "cls", "True", "False", "None", "if", "else", "for", "while", "return"}:
            continue
        
        # Look for the variable outside placeholders
        pattern = r"\b" + re.escape(var) + r"\b(?![^{]*})"
        match = re.search(pattern, inner_string)
        if match:
            replacement = f"{{{var}}}"
            inner_string = re.sub(pattern, replacement, inner_string)
    
    # Check for invalid format specifiers
    for match in FORMAT_SPEC_REGEX.finditer(original):
        var, format_spec = match.groups()
        if format_spec.strip() and not _is_valid_format_spec(format_spec):
            # Replace with proper format specifier
            inner_string = inner_string.replace(f"{{{var}:{format_spec}}}", f"{{{var}}}")
    
    # Reconstruct the f-string
    fixed = f'f"{inner_string}"'
    
    return fixed, fixed != fstring


def _is_valid_format_spec(format_spec: str) -> bool:
    """
    Check if a format specifier is valid.
    
    Args:
        format_spec: Format specifier to check
        
    Returns:
        True if valid, False otherwise
    """
    # Basic format specifier validation
    valid_specs = {
        "d", "b", "o", "x", "X", "e", "E", "f", "F", "g", "G", "%",
        "c", "s", "a", "r", "<", ">", "^", "=", " ", "+", "-", "_"
    }
    
    # Check if format_spec contains any valid specifier
    for spec in valid_specs:
        if spec in format_spec:
            return True
    
    # Check for width and precision
    if re.match(r"^[0-9]+(\.[0-9]+)?$", format_spec):
        return True
    
    return False


def fix_file(file_path: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Fix f-strings in a file.
    
    Args:
        file_path: Path to the file to fix
        dry_run: If True, don't actually modify the file
        
    Returns:
        Tuple of (num_fstrings_found, num_fstrings_fixed)
    """
    logger.debug(f"Processing file: {file_path}")
    
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return 0, 0
    
    # Extract all variables in the file
    variables = extract_variable_names(content)
    
    # Find and fix f-strings
    fstrings_found = 0
    fstrings_fixed = 0
    modified_content = content
    
    for match in FSTRING_REGEX.finditer(content):
        fstring = match.group(0)
        fstrings_found += 1
        
        fixed_fstring, was_modified = fix_fstring(fstring, variables)
        if was_modified:
            fstrings_fixed += 1
            modified_content = modified_content.replace(fstring, fixed_fstring)
    
    # Save the modified content if needed
    if not dry_run and fstrings_fixed > 0:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            logger.info(f"Fixed {fstrings_fixed} f-strings in {file_path}")
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            return fstrings_found, 0
    
    return fstrings_found, fstrings_fixed


def main() -> int:
    """
    Main function for the script.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(description="Fix f-strings in Python files")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for Python files (default: current directory)"
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="+",
        default=[".git", ".venv", "venv", "env", "build", "dist", "__pycache__"],
        help="Directories to exclude from search"
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Don't actually modify files, just show what would be done"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--single-file",
        "-f",
        help="Process a single file instead of a directory"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.dry_run:
        logger.info("Dry run mode enabled - no files will be modified")
    
    # Process single file if specified
    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 1
        
        if not file_path.name.endswith(".py"):
            logger.error(f"Not a Python file: {file_path}")
            return 1
        
        total_found, total_fixed = fix_file(file_path, args.dry_run)
        logger.info(f"Processed 1 file: found {total_found} f-strings, fixed {total_fixed}")
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
    total_found = 0
    total_fixed = 0
    
    for i, file_path in enumerate(python_files, 1):
        if i % 100 == 0 or i == total_files:
            logger.info(f"Processing file {i}/{total_files}")
        
        found, fixed = fix_file(file_path, args.dry_run)
        total_found += found
        total_fixed += fixed
    
    logger.info(f"Processed {total_files} files: found {total_found} f-strings, fixed {total_fixed}")
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