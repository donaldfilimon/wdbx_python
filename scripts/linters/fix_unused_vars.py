#!/usr/bin/env python
"""
Script to find and fix unused variables in Python files.

This script scans Python files to identify variables declared but never used
and replaces them with underscores to comply with PEP 8 guidelines while
silencing linter warnings.
"""

import argparse
import ast
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fix_unused_vars")


class UnusedVariableFinder(ast.NodeVisitor):
    """AST visitor that finds unused variables in a function scope."""
    
    def __init__(self):
        self.used_names: Set[str] = set()
        self.defined_names: Dict[str, ast.AST] = {}
        self.name_scopes: List[Dict[str, ast.AST]] = [{}]
        self.unused_names: Dict[str, Tuple[int, int]] = {}  # Variable name -> (line, col)
        
    def visit_Name(self, node: ast.Name):
        """Visit a name node in the AST."""
        if isinstance(node.ctx, ast.Store):
            # Variable is being assigned to
            self.defined_names[node.id] = node
            self.name_scopes[-1][node.id] = node
        elif isinstance(node.ctx, ast.Load):
            # Variable is being used
            self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit a function definition, creating a new variable scope."""
        # Add function name to used names to avoid removing it
        self.used_names.add(node.name)
        
        # Add arguments to defined names for this scope
        old_scopes = self.name_scopes
        self.name_scopes = [{}]
        
        for arg in node.args.args:
            self.name_scopes[-1][arg.arg] = arg
        
        # Visit function body
        self.generic_visit(node)
        
        # Check for unused variables in this scope
        for name, var_node in self.name_scopes[-1].items():
            if name not in self.used_names and not name.startswith("_"):
                self.unused_names[name] = (var_node.lineno, getattr(var_node, "col_offset", 0))
        
        # Restore scope
        self.name_scopes = old_scopes
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit a class definition, creating a new variable scope."""
        # Add class name to used names to avoid removing it
        self.used_names.add(node.name)
        
        # Create a new variable scope
        self.name_scopes.append({})
        
        # Visit class body
        self.generic_visit(node)
        
        # Remove the class scope
        self.name_scopes.pop()
    
    def get_unused_variables(self) -> Dict[str, Tuple[int, int]]:
        """Get all unused variable names with their locations."""
        # Check global scope for unused variables
        for name, var_node in self.name_scopes[0].items():
            if name not in self.used_names and not name.startswith("_"):
                self.unused_names[name] = (var_node.lineno, getattr(var_node, "col_offset", 0))
        
        return self.unused_names


def find_unused_variables(code: str) -> Dict[str, Tuple[int, int]]:
    """
    Find unused variables in the given code.
    
    Args:
        code: Python code as a string
        
    Returns:
        Dictionary of variable names to (line, col) tuples
    """
    try:
        tree = ast.parse(code)
        finder = UnusedVariableFinder()
        finder.visit(tree)
        return finder.get_unused_variables()
    except SyntaxError as e:
        logger.warning(f"Syntax error in code: {e}")
        return {}


def fix_unused_variables(content: str, unused_vars: Dict[str, Tuple[int, int]]) -> Tuple[str, int]:
    """
    Replace unused variables with underscores.
    
    Args:
        content: Original file content
        unused_vars: Dictionary of variable names to (line, col) tuples
        
    Returns:
        Tuple of (modified content, number of replacements)
    """
    if not unused_vars:
        return content, 0
    
    lines = content.split("\n")
    replacements = 0
    
    # Sort by line number in descending order to avoid positions shifting
    sorted_vars = sorted(unused_vars.items(), key=lambda x: x[1][0], reverse=True)
    
    for var_name, (line_no, _) in sorted_vars:
        if line_no <= 0 or line_no > len(lines):
            continue
        
        # Get the line (0-indexed)
        line = lines[line_no - 1]
        
        # Create a pattern that matches the variable name as a whole word
        pattern = r"\b" + re.escape(var_name) + r"\b"
        
        # Replace only the first occurrence in the line
        # This assumes the first occurrence is the variable definition
        # We use a more specific approach to replace only the right occurrences
        new_line = re.sub(pattern, "_", line, count=1)
        
        if new_line != line:
            lines[line_no - 1] = new_line
            replacements += 1
            logger.debug(f"Replaced '{var_name}' with '_' on line {line_no}")
    
    if replacements > 0:
        return "\n".join(lines), replacements
    
    return content, 0


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


def process_file(file_path: Path, dry_run: bool = False) -> int:
    """
    Process a single file, fixing unused variables.
    
    Args:
        file_path: Path to the file to process
        dry_run: If True, don't actually modify the file
        
    Returns:
        Number of variables fixed
    """
    logger.debug(f"Processing file: {file_path}")
    
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return 0
    
    unused_vars = find_unused_variables(content)
    if not unused_vars:
        return 0
    
    logger.debug(f"Found {len(unused_vars)} unused variables in {file_path}")
    
    modified_content, replacements = fix_unused_variables(content, unused_vars)
    if replacements == 0:
        return 0
    
    if not dry_run:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            logger.info(f"Fixed {replacements} unused variables in {file_path}")
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            return 0
    else:
        logger.info(f"Would fix {replacements} unused variables in {file_path} (dry run)")
    
    return replacements


def main() -> int:
    """
    Main function for the script.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(description="Fix unused variables in Python files")
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
        
        fixed = process_file(file_path, args.dry_run)
        logger.info(f"Fixed {fixed} unused variables in total")
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
    total_fixed = 0
    
    for i, file_path in enumerate(python_files, 1):
        if i % 100 == 0 or i == total_files:
            logger.info(f"Processing file {i}/{total_files}")
        
        fixed = process_file(file_path, args.dry_run)
        total_fixed += fixed
    
    logger.info(f"Fixed {total_fixed} unused variables across {total_files} files")
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