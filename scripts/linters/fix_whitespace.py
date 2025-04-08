#!/usr/bin/env python
"""
Script to fix whitespace issues in Python files:
- W293: blank line contains whitespace
- E303: too many blank lines
"""

import os
import re
import sys


def fix_whitespace(file_path):
    """Fix whitespace issues in the specified file."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Fix blank lines with whitespace
    fixed_content = re.sub(r"[ \t]+\n", "\n", content)

    # Fix too many blank lines (more than 2 consecutive blank lines)
    fixed_content = re.sub(r"\n{4,}", "\n\n\n", fixed_content)

    if content != fixed_content:
        print(f"Fixing whitespace in {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)
        return True
    return False


def process_directory(directory):
    """Process all Python files in the given directory recursively."""
    fixed_files = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_whitespace(file_path):
                    fixed_files += 1

    return fixed_files


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_whitespace.py <directory>")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a directory")
        sys.exit(1)

    fixed = process_directory(target_dir)
    print(f"Fixed {fixed} file(s)")
