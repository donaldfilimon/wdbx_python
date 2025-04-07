"""
Import context for tests.

This allows tests to import the package modules as if the package was installed.
"""
import sys
from pathlib import Path

# Add the project root directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import package modules

# Import testing utilities
