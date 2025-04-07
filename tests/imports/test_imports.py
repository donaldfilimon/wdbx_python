#!/usr/bin/env python
"""
Test script to verify that imports are working correctly.
"""

import os
import sys

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_path)

def test_imports():
    """Test imports of key modules."""
    print("Testing imports...")
    
    # Import data structures
    print("- Importing data_structures module...", end=" ")
    try:
        from wdbx.data_structures import Block, EmbeddingVector, ShardInfo
        print("Success!")
    except ImportError as e:
        print(f"Failed: {e}")
        
    # Import MVCC
    print("- Importing mvcc module...", end=" ")
    try:
        from wdbx.mvcc import MVCCManager, MVCCTransaction
        print("Success!")
    except ImportError as e:
        print(f"Failed: {e}")
        
    # Import optimized module
    print("- Importing optimized module...", end=" ")
    try:
        from wdbx.ml.optimized import OptimizedBlockManager, OptimizedVectorStore
        print("Success!")
    except ImportError as e:
        print(f"Failed: {e}")
        
    print("\nAll import tests completed.")


if __name__ == "__main__":
    test_imports() 