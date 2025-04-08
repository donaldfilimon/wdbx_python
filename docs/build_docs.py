#!/usr/bin/env python
"""
WDBX Documentation Builder

This script serves as the main entry point for building WDBX documentation.
It provides multiple build options to address different use cases:
- simple: Quick, basic HTML output
- standalone: Single-file HTML with embedded styles and JavaScript
- sphinx: Full-featured documentation with comprehensive navigation
- sphinx-minimal: Lightweight Sphinx build with basic theme
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional


def check_module_installed(module_name: str) -> bool:
    """
    Check if a Python module is installed.

    Args:
        module_name: Name of the module to check

    Returns:
        True if the module is installed, False otherwise
    """
    return importlib.util.find_spec(module_name) is not None


def ensure_requirements() -> None:
    """
    Ensure required packages are installed.
    Prompts the user to install missing packages if needed.
    """
    required: Dict[str, str] = {
        "markdown": "Basic markdown processing",
        "pygments": "Syntax highlighting for code blocks",
        "bs4": "HTML processing for documentation (BeautifulSoup)",
    }

    missing: List[str] = []
    for module, description in required.items():
        if not check_module_installed(module):
            missing.append(f"- {module}: {description}")

    if missing:
        print("Some packages required for building documentation are missing:")
        for pkg in missing:
            print(pkg)

        install = input("Would you like to install them now? (y/n): ")
        if install.lower() == "y":
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *required.keys()], check=True
                )
                print("Packages installed successfully.")
            except subprocess.CalledProcessError:
                print("Error installing packages. Please install them manually.")
                sys.exit(1)
        else:
            print("Please install the required packages and try again.")
            sys.exit(1)


def build_simple() -> Optional[Path]:
    """
    Build documentation using the quick_compile.py script.

    Returns:
        Path to the output file if successful, None otherwise
    """
    print("Building documentation with quick_compile.py...")
    try:
        from quick_compile import main as quick_compile_main

        return quick_compile_main()
    except ImportError:
        print("Error: quick_compile.py not found or contains errors.")
        sys.exit(1)
    except Exception as e:
        print(f"Error building simple documentation: {e}")
        sys.exit(1)


def build_standalone() -> Optional[Path]:
    """
    Build documentation using the standalone_build.py script.

    Returns:
        Path to the output file if successful, None otherwise
    """
    print("Building standalone documentation...")
    try:
        from standalone_build import build_standalone_docs

        output_file = build_standalone_docs()
        print(f"Standalone documentation built successfully: {output_file}")
        return output_file
    except ImportError:
        print("Error: standalone_build.py not found or contains errors.")
        sys.exit(1)
    except Exception as e:
        print(f"Error building standalone documentation: {e}")
        sys.exit(1)


def build_sphinx(minimal: bool = False) -> Optional[Path]:
    """
    Build documentation using Sphinx.

    Args:
        minimal: Whether to use minimal configuration

    Returns:
        Path to the output index.html if successful, None otherwise
    """
    build_type = "minimal" if minimal else "full"
    print(f"Building documentation with Sphinx ({build_type} configuration)...")

    # Check for Sphinx and required extensions
    required_packages = ["sphinx", "myst-parser"]
    missing_packages = [pkg for pkg in required_packages if not check_module_installed(pkg)]

    if missing_packages:
        print(f"Missing required packages for Sphinx build: {', '.join(missing_packages)}")
        install = input("Would you like to install them now? (y/n): ")
        if install.lower() == "y":
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing_packages, check=True
                )
                print("Sphinx packages installed successfully.")
            except subprocess.CalledProcessError:
                print("Error installing Sphinx packages. Please install them manually.")
                sys.exit(1)
        else:
            print("Sphinx is required for this build option. Aborting.")
            sys.exit(1)

    if minimal:
        try:
            # Use minimal build script if available
            from build_minimal import main as build_minimal_main

            return build_minimal_main()
        except ImportError:
            # Fall back to direct sphinx-build command
            build_dir = Path("_build/sphinx_minimal")
            build_dir.mkdir(parents=True, exist_ok=True)

            try:
                subprocess.run(
                    ["sphinx-build", "-b", "html", "-Dhtml_theme=basic", ".", str(build_dir)],
                    check=True,
                )

                index_path = build_dir / "index.html"
                if index_path.exists():
                    return index_path
                else:
                    print(f"Build completed but index.html not found at {index_path}")
                    return None
            except subprocess.CalledProcessError as e:
                print(f"Error running sphinx-build: {e}")
                sys.exit(1)
    else:
        # Run full Sphinx build
        build_dir = Path("_build/html")
        try:
            subprocess.run(["sphinx-build", "-b", "html", ".", str(build_dir)], check=True)

            index_path = build_dir / "index.html"
            if index_path.exists():
                return index_path
            else:
                print(f"Build completed but index.html not found at {index_path}")
                return None
        except subprocess.CalledProcessError as e:
            print(f"Error running sphinx-build: {e}")
            sys.exit(1)


def main() -> None:
    """Main entry point for the documentation builder."""
    parser = argparse.ArgumentParser(
        description="WDBX Documentation Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Build methods:
  simple       - Quick, basic HTML output
  standalone   - Single-file HTML with embedded styles and JavaScript
  sphinx       - Full-featured documentation with comprehensive navigation
  sphinx-minimal - Lightweight Sphinx build with basic theme
        """,
    )
    parser.add_argument(
        "--method",
        choices=["simple", "standalone", "sphinx", "sphinx-minimal"],
        default="standalone",
        help="Build method to use (default: standalone)",
    )
    parser.add_argument(
        "--open", action="store_true", help="Open documentation in browser after building"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Custom output directory for documentation files"
    )

    args = parser.parse_args()

    # Ensure requirements are met
    ensure_requirements()

    # Set custom output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WDBX_DOCS_OUTPUT_DIR"] = str(output_dir)

    # Run the selected build method
    output_file = None

    if args.method == "simple":
        output_file = build_simple()
    elif args.method == "standalone":
        output_file = build_standalone()
    elif args.method == "sphinx":
        output_file = build_sphinx(minimal=False)
    elif args.method == "sphinx-minimal":
        output_file = build_sphinx(minimal=True)

    # Open in browser if requested and output file exists
    if args.open and output_file and output_file.exists():
        url = f"file://{output_file.absolute()}"
        print(f"Opening documentation at {url}")
        webbrowser.open(url)
    elif args.open and not output_file:
        print("Cannot open documentation: output file not found")


if __name__ == "__main__":
    main()
