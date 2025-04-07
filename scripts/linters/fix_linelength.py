#!/usr/bin/env python
"""
Script to fix long lines in Python files by breaking them at appropriate points.
"""

import argparse
import os
import sys

MAX_LINE_LENGTH = 100

def fix_long_lines(file_path):
    """Fix lines that exceed the maximum line length."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    modified = False
    
    for i, line in enumerate(lines):
        # Skip comments and docstrings
        if line.strip().startswith("#") or '"""' in line or "'''" in line:
            continue
            
        if len(line) > MAX_LINE_LENGTH:
            # Only process the file if needed
            modified = True
            break
    
    if not modified:
        return False
    
    print(f"Processing {file_path}")
    # Use autopep8 for this specific file with line length fixing
    try:
        import subprocess
        subprocess.run(
            [
                "autopep8",
                "--in-place",
                "--select=E501",
                f"--max-line-length={MAX_LINE_LENGTH}",
                file_path
            ],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except Exception as e:
        print(f"Error fixing line length in {file_path}: {e}")
        return False

def process_directory(directory):
    """Process all Python files in the given directory recursively."""
    fixed_files = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_long_lines(file_path):
                    fixed_files += 1
    
    return fixed_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix long lines in Python files")
    parser.add_argument("directory", help="Directory to process")
    parser.add_argument("--max-line-length", type=int, default=100, 
                        help="Maximum line length (default: 100)")
                        
    args = parser.parse_args()
    
    MAX_LINE_LENGTH = args.max_line_length
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)
    
    fixed = process_directory(args.directory)
    print(f"Fixed {fixed} file(s)") 