#!/usr/bin/env python3
"""
Run the WDBX Streamlit app for vector visualization and exploration.

This script provides a convenient way to launch the Streamlit UI
without having to install WDBX as a package.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to python path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    """Run the Streamlit app."""
    # Path to the Streamlit app
    app_path = PROJECT_ROOT / "src" / "wdbx" / "ui" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        return 1
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"])
        print("Streamlit installed successfully.")
    
    # Launch the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    print(f"Launching Streamlit app: {app_path}")
    print(f"Command: {' '.join(cmd)}")
    
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 