"""
Import context for tests.

This allows tests to import the package modules as if the package was installed.
"""

import sys
from pathlib import Path

from wdbx import WDBX

# Add the project root directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import package modules

# Import testing utilities

# Initialize
wdbx = WDBX(vector_dimension=768)

# Create collection
collection = wdbx.create_collection("documents")

# Insert data
data = {"text": "This is a test document", "metadata": {"type": "test"}}
collection.insert(data)

# Query
results = collection.query("test document")
print(results)
