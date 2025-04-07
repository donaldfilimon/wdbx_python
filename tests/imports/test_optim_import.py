#!/usr/bin/env python
"""
Test script to verify the imports in optimized.py.
"""

import importlib.util
import os
import sys

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, src_path)

def test_import_module(module_path):
    """Test importing a specific module."""
    print(f"Testing import of: {module_path}")
    
    try:
        # Get the absolute path to the module
        file_path = os.path.join(src_path, module_path.replace(".", os.path.sep) + ".py")
        print(f"File path: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return False
        
        # Import the module using importlib
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None:
            print(f"Error: Could not create spec for {file_path}")
            return False
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"Success: Imported {module_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Test importing the optimized module
    success = test_import_module("wdbx.ml.optimized")
    
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nTests failed!")
        sys.exit(1) 