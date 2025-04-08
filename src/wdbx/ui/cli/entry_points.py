"""
WDBX entry points for console scripts.

This module provides the entry points for the console scripts defined in setup.py,
including the main CLI, web UI, terminal UI, and benchmarking tools.
"""

import importlib.util
import logging
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wdbx.entry_points")

# Constants
STREAMLIT_MIN_VERSION = "1.24.0"
PLOTLY_MIN_VERSION = "5.14.0"
PANDAS_MIN_VERSION = "2.0.0"

# Check if we are in a virtual environment
IN_VENV = sys.prefix != sys.base_prefix


def print_banner() -> None:
    """Print a colorful WDBX banner."""
    banner = r"""
    \033[1;36m __      __________  ______  __
    \033[1;36m/  \    /  \______ \ \____ \|  |__  ___  ___
    \033[1;36m\   \/\/   /|    |  \|  |_> >  |  \ \  \/  /
    \033[1;36m \        / |    `   \   __/|   Y  \ >    <
    \033[1;36m  \__/\  / /_______  /__|   |___|  //__/\_ \
    \033[1;36m       \/          \/            \/       \/
    \033[0m
    \033[1;32mWide Distributed Block Exchange\033[0m - v1.0.0
    \033[0;37mPowered by Vector Database Technology\033[0m
    """

    lines = banner.split("\n")
    for line in lines:
        print(line)
    print("\033[0m")


def parse_custom_args(args: List[str]) -> Dict[str, Any]:
    """
    Parse custom command line arguments manually.

    Args:
        args: Command line arguments

    Returns:
        Dictionary of parsed arguments
    """
    result = {
        "debug": False,
        "theme": "default",
        "host": "127.0.0.1",
        "port": 8501,
        "server.address": "127.0.0.1",
        "server.port": 8501,
        "headless": False,
        "browser.serverAddress": "127.0.0.1",
        "browser.serverPort": 8501,
    }

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle flag arguments
        if arg == "--debug":
            result["debug"] = True
            i += 1
            continue

        if arg == "--headless":
            result["headless"] = True
            i += 1
            continue

        # Handle key-value arguments
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
            result[key] = value
            i += 1
            continue

        # Handle space-separated key-value arguments
        if arg.startswith("--") and i + 1 < len(args):
            key = arg[2:]
            value = args[i + 1]
            result[key] = value
            i += 2
            continue

        i += 1

    return result


def check_dependencies(packages: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
    """
    Check if required packages are installed with minimum versions.

    Args:
        packages: List of (package_name, min_version) tuples

    Returns:
        Tuple of (all_installed, missing_packages)
    """
    missing = []
    all_installed = True

    for package, min_version in packages:
        try:
            module = importlib.import_module(package)
            if hasattr(module, "__version__"):
                version = module.__version__
                if version < min_version:
                    logger.warning(
                        f"{package} version {version} is older than required {min_version}"
                    )
                    missing.append(f"{package}>={min_version}")
                    all_installed = False
        except ImportError:
            missing.append(f"{package}>={min_version}")
            all_installed = False

    return all_installed, missing


def install_dependencies(packages: List[str]) -> bool:
    """
    Install required dependencies.

    Args:
        packages: List of packages to install

    Returns:
        True if installation succeeded, False otherwise
    """
    if not IN_VENV:
        logger.warning("Installing packages outside of a virtual environment")

    try:
        pip_install_cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
        logger.info(f"Installing packages: {', '.join(packages)}")
        subprocess.check_call(pip_install_cmd + packages)
        logger.info("Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def handle_keyboard_interrupt(signum, frame):
    """
    Handle keyboard interrupt (Ctrl+C) gracefully.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    print("\n\033[1;33mOperation interrupted by user\033[0m")
    sys.exit(130)  # Standard exit code for Ctrl+C


def ensure_environment() -> bool:
    """
    Ensure the environment is set up correctly.

    Returns:
        True if the environment is ready, False otherwise
    """
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False

    return True


def cli_main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the 'wdbx' command-line tool.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]

    # Print banner for CLI
    print_banner()

    # Ensure environment is ready
    if not ensure_environment():
        return 1

    # Parse custom args
    parsed_args = parse_custom_args(args)
    debug_mode = parsed_args.get("debug", False)

    # Configure logging level
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        # Import lazily to avoid circular imports
        import importlib

        logger.debug("Importing CLI module")
        cli_module = importlib.import_module("wdbx.ui.cli.cli")
        main_func = cli_module.main

        # Set sys.argv for argparse to use
        sys.argv = [sys.argv[0]] + (args if args else [])

        # Run the main CLI function
        logger.debug("Starting CLI main function")
        main_func()
        return 0
    except KeyboardInterrupt:
        print("\n\033[1;33mOperation interrupted by user\033[0m", file=sys.stderr)
        return 130  # Standard exit code for Ctrl+C
    except ImportError as e:
        logger.error(f"Failed to import required module: {e}")
        if debug_mode:
            traceback.print_exc()
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        if debug_mode:
            traceback.print_exc()
        return 1


def web_main(args: Optional[List[str]] = None) -> int:
    """
    Entry point for the 'wdbx-web' command-line tool.

    This function launches the Streamlit-based UI with enhanced features.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]

    # Ensure environment is ready
    if not ensure_environment():
        return 1

    # Parse custom args
    parsed_args = parse_custom_args(args)
    debug_mode = parsed_args.get("debug", False)
    headless = parsed_args.get("headless", False)

    # Configure logging level
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Find the streamlit app
    module_dir = Path(__file__).parent.parent.parent
    streamlit_app_path = module_dir / "ui" / "streamlit_app.py"

    if not streamlit_app_path.exists():
        logger.error(f"Streamlit app not found at {streamlit_app_path}")
        return 1

    # Check if required packages are installed
    required_packages = [
        ("streamlit", STREAMLIT_MIN_VERSION),
        ("plotly", PLOTLY_MIN_VERSION),
        ("pandas", PANDAS_MIN_VERSION),
    ]

    all_installed, missing = check_dependencies(required_packages)

    if not all_installed:
        logger.warning(f"Missing required packages: {', '.join(missing)}")

        # Ask for confirmation before installing
        if not headless:
            install_confirm = input("\nDo you want to install required packages? [Y/n]: ")
            should_install = install_confirm.lower() in ["", "y", "yes"]
        else:
            logger.info(
                "Running in headless mode, attempting to install dependencies automatically"
            )
            should_install = True

        if should_install:
            if not install_dependencies(missing):
                logger.error("Failed to install dependencies. Please install manually:")
                print(f"\npip install {' '.join(missing)}")
                return 1

            # Re-check dependencies after installation
            all_installed, still_missing = check_dependencies(required_packages)
            if not all_installed:
                logger.error(
                    f"Failed to install all dependencies. Still missing: {', '.join(still_missing)}"
                )
                return 1
        else:
            logger.error("Required packages must be installed to continue")
            print(f"\npip install {' '.join(missing)}")
            return 1

    # Parse streamlit-specific args
    host = parsed_args.get("server.address", "127.0.0.1")
    port = int(parsed_args.get("server.port", 8501))

    # Launch the Streamlit app with enhanced features
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]

    # Add additional Streamlit args
    for key, value in parsed_args.items():
        if key.startswith("browser.") or key.startswith("server.") or key.startswith("theme."):
            if isinstance(value, bool):
                if value:
                    cmd.extend([f"--{key}"])
            else:
                cmd.extend([f"--{key}", str(value)])

    logger.info(f"Launching Streamlit UI from {streamlit_app_path}")
    logger.info(f"Access the web interface at http://{host}:{port} when ready")

    try:
        process = subprocess.Popen(cmd)

        # For better user experience, show a waiting message
        if not headless:
            print("\n\033[1;32mStarting Streamlit server...\033[0m")
            for i in range(5):
                time.sleep(0.5)
                print(f"\r\033[1;32mStarting Streamlit server... {'.' * (i+1)}\033[0m", end="")
            print(
                f"\n\n\033[1;36mAccess the web interface at \033[1;33mhttp://{host}:{port}\033[0m"
            )
            print("\n\033[0;37mPress Ctrl+C to stop the server\033[0m")

        # Wait for the process to complete
        return_code = process.wait()
        return return_code
    except KeyboardInterrupt:
        print("\n\033[1;33mStreamlit UI stopped by user\033[0m", file=sys.stderr)
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"Error launching Streamlit: {e}")
        if debug_mode:
            traceback.print_exc()
        return 1


def dashboard_main(args: Optional[List[str]] = None) -> int:
    """
    Entry point for the 'wdbx-dashboard' command-line tool.

    This function launches the terminal UI dashboard.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]

    # Ensure environment is ready
    if not ensure_environment():
        return 1

    # Parse custom args
    parsed_args = parse_custom_args(args)
    debug_mode = parsed_args.get("debug", False)

    # Configure logging level
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        # Import the terminal UI
        # Import and create WDBX instance
        from ...core import DEFAULT_DATA_DIR, SHARD_COUNT, VECTOR_DIMENSION, create_wdbx
        from .terminal_ui import run_terminal_ui

        # Get parameters from parsed args
        vector_dimension = int(parsed_args.get("dimension", VECTOR_DIMENSION))
        num_shards = int(parsed_args.get("shards", SHARD_COUNT))
        data_dir = parsed_args.get("data_dir", DEFAULT_DATA_DIR)

        # Create WDBX instance
        logger.info(
            f"Creating WDBX instance with {vector_dimension}-d vectors, {num_shards} shards"
        )
        wdbx = create_wdbx(
            vector_dimension=vector_dimension,
            num_shards=num_shards,
            data_dir=data_dir,
        )

        # Launch the terminal UI
        print_banner()
        print("\033[1;32mStarting terminal dashboard...\033[0m")
        run_terminal_ui(wdbx)
        return 0
    except ImportError as e:
        logger.error(f"Failed to import terminal UI: {e}")
        print("\nPlease install the required packages:")
        print("pip install rich textual")
        return 1
    except KeyboardInterrupt:
        print("\n\033[1;33mDashboard stopped by user\033[0m", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        if debug_mode:
            traceback.print_exc()
        return 1


def benchmark_main(args: Optional[List[str]] = None) -> int:
    """
    Entry point for the 'wdbx-benchmark' command-line tool.

    This function runs benchmarks for WDBX.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]

    # Ensure environment is ready
    if not ensure_environment():
        return 1

    # Parse custom args
    parsed_args = parse_custom_args(args)
    debug_mode = parsed_args.get("debug", False)

    # Configure logging level
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        # Print banner
        print_banner()
        print("\033[1;32mRunning WDBX benchmarks...\033[0m")

        # Import benchmarking module
        try:
            from ...benchmarks import run_benchmarks
        except ImportError:
            logger.error("Benchmarking module not found")
            return 1

        # Run benchmarks
        benchmark_type = parsed_args.get("type", "all")
        output_format = parsed_args.get("format", "text")
        iterations = int(parsed_args.get("iterations", 3))

        result = run_benchmarks(
            benchmark_type=benchmark_type,
            output_format=output_format,
            iterations=iterations,
        )

        # Print results
        if output_format == "json":
            import json

            print(json.dumps(result, indent=2))

        return 0
    except KeyboardInterrupt:
        print("\n\033[1;33mBenchmarks interrupted by user\033[0m", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        if debug_mode:
            traceback.print_exc()
        return 1
