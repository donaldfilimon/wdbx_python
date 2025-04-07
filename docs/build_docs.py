#!/usr/bin/env python
"""
Build script for WDBX documentation.
This script automates the process of building the Sphinx documentation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def main():
    """Build the Sphinx documentation."""
    # Get the docs directory
    docs_dir = Path(__file__).parent.absolute()
    project_root = docs_dir.parent
    
    # Output directory
    build_dir = docs_dir / "_build"
    html_dir = build_dir / "html"
    
    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)
    
    # Build the documentation
    print("Building documentation...")
    try:
        result = subprocess.run(
            ["sphinx-build", "-b", "html", str(docs_dir), str(html_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}", file=sys.stderr)
        
        # Create and copy a root index.html file
        root_index = docs_dir / "index.html"
        if root_index.exists():
            print(f"Copying {root_index} to project root...")
            shutil.copy2(root_index, project_root / "index.html")
        
        print(f"\nDocumentation built successfully in {html_dir}")
        print(f"HTML entry point: {html_dir / 'index.html'}")
        print("To view the documentation, open index.html in your browser")
        
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}", file=sys.stderr)
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 