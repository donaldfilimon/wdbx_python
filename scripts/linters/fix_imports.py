#!/usr/bin/env python
"""
Script to remove unused imports using autoflake
"""

import os
import subprocess
import sys


def run_autoflake(directory):
    """Run autoflake on all Python files in the directory"""
    try:
        subprocess.run(["pip", "install", "autoflake"], check=True)

        print(f"Running autoflake on {directory}...")
        result = subprocess.run(
            ["autoflake", "--in-place", "--remove-all-unused-imports", "--recursive", directory],
            check=True,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        print("Successfully removed unused imports")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running autoflake: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_imports.py <directory>")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a directory")
        sys.exit(1)

    success = run_autoflake(target_dir)
    sys.exit(0 if success else 1)
