#!/usr/bin/env python3
"""
WDBX Linter Issue Fixer

This script runs all linting tools to fix common issues in the WDBX codebase.
It uses the integrated linter to combine industry-standard tools with custom linters.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run all linting tools to fix common issues."""
    # Ensure we're in the right directory
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    # Check if scripts exist relative to the actual project root
    integrated_linter_path = project_root / "scripts" / "integrated_linter.py"
    if not integrated_linter_path.exists():
        # Fall back to run_linters.py if integrated_linter.py doesn't exist
        run_linters_path = project_root / "scripts" / "run_linters.py"
        if not run_linters_path.exists():
            logger.error(
                "Could not find linting scripts. Make sure you're running this from the project root."
            )
            return 1
        linter_script = run_linters_path
    else:
        linter_script = integrated_linter_path

    # Print header
    print("=" * 80)
    print("WDBX Linter Issue Fixer")
    print("=" * 80)
    print("\nThis script will attempt to fix common linting issues in the codebase.")

    # Determine which linter we're using
    if linter_script.name == "integrated_linter.py":
        print("\nUsing integrated linter with industry-standard tools and custom WDBX linters.")
    else:
        print("\nUsing custom WDBX linters only.")

    # Run the linters with fix and report options
    try:
        print("\nRunning linters to fix issues...")

        if linter_script.name == "integrated_linter.py":
            # For integrated linter, provide more options
            cmd = [sys.executable, str(linter_script), "--fix", "--report", "--install-missing"]
        else:
            # For run_linters.py
            cmd = [sys.executable, str(linter_script), "--fix", "--report"]

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print("\nLinting completed with some issues. Check the report for details.")
            print("You may need to manually fix some remaining issues.")
        else:
            print("\nLinting completed successfully! All fixable issues have been addressed.")

        print("\nNext steps:")
        print("1. Check the generated report in the 'linting_reports' directory")
        print("2. Run your tests to ensure no functionality was broken")
        print("3. Commit the changes if everything looks good")

        return result.returncode

    except Exception as e:
        logger.error(f"Error running linters: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
