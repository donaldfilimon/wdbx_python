#!/usr/bin/env python3
"""
WDBX Unified Tool Launcher

A single entry point to launch any of the WDBX tools and scripts.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

SCRIPT_DIR = Path(__file__).parent / "scripts"
PROJECT_ROOT = Path(__file__).parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wdbx_tool")


def setup_environment(env_file: Optional[str] = None) -> Dict[str, str]:
    """Set up environment variables from .env file or defaults."""
    env_vars = {
        "WDBX_DATA_DIR": str(PROJECT_ROOT / "data"),
        "WDBX_LOG_LEVEL": "INFO",
        "WDBX_MAX_MEMORY_PERCENT": "85.0",
        "WDBX_MEMORY_CHECK_INTERVAL": "10",
    }

    # Load from .env file if it exists
    if env_file and os.path.exists(env_file):
        logger.info("Loading environment from %s", env_file)
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

    # Set environment variables
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug("Set %s=%s", key, value)

    return env_vars


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WDBX Tool Launcher")

    # Global arguments
    parser.add_argument("--env", help="Path to .env file for configuration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--data-dir", help="Override data directory")

    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Application runners
    run_parser = subparsers.add_parser("run", help="Run WDBX application")
    run_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    run_parser.add_argument("--port", "-p", type=int, default=8080, help="HTTP port")
    run_parser.add_argument("--config", "-c", help="Path to configuration file")

    # Web UI
    web_parser = subparsers.add_parser("web", help="Run WDBX web UI")
    web_parser.add_argument("--port", "-p", type=int, default=3000, help="Web UI port")
    web_parser.add_argument(
        "--theme", choices=["light", "dark", "auto"], default="auto", help="UI theme"
    )

    # Streamlit
    streamlit_parser = subparsers.add_parser("streamlit", help="Run Streamlit visualization")
    streamlit_parser.add_argument("--port", type=int, default=8501, help="Streamlit port")

    # Tests
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    test_parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    test_parser.add_argument("--xml", action="store_true", help="Generate XML report")
    test_parser.add_argument("--pattern", help="Test name pattern to match")

    # Benchmarks
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--vectors", type=int, default=10000, help="Number of vectors")
    benchmark_parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    benchmark_parser.add_argument(
        "--backend",
        choices=["numpy", "torch", "jax", "all"],
        default="all",
        help="ML backend to benchmark",
    )
    benchmark_parser.add_argument("--output", help="Path to save benchmark results")

    # Linters
    lint_parser = subparsers.add_parser("lint", help="Run linters")
    lint_parser.add_argument("--fix", action="store_true", help="Fix issues automatically")
    lint_parser.add_argument("--path", type=str, default="src", help="Path to lint")

    # Create new subparser for database management
    db_parser = subparsers.add_parser("db", help="Database management")
    db_parser.add_argument(
        "action",
        choices=["init", "status", "cleanup", "backup", "restore"],
        help="Database action to perform",
    )
    db_parser.add_argument("--target", help="Target file for backup/restore")

    # Create metrics collection subparser
    metrics_parser = subparsers.add_parser("metrics", help="Collect performance metrics")
    metrics_parser.add_argument("--output-dir", "-o", help="Directory to store metrics output")
    metrics_parser.add_argument(
        "--interval", "-i", type=int, default=5, help="Collection interval in seconds (default: 5)"
    )
    metrics_parser.add_argument(
        "--duration", "-d", type=int, help="Duration in seconds (default: run until interrupted)"
    )
    metrics_parser.add_argument("--server", "-s", help="Connect to WDBX server at host:port")

    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    """Build the command list based on parsed arguments."""
    command_name = args.command
    cmd = [sys.executable]  # Start with the python interpreter

    if command_name == "run":
        script_path = SCRIPT_DIR / "runners" / "run_wdbx.py"
        cmd.append(str(script_path))
        if args.interactive:
            cmd.append("--interactive")
        if args.port:
            cmd.extend(["--port", str(args.port)])
        if args.config:
            cmd.extend(["--config", args.config])

    elif command_name == "web":
        script_path = SCRIPT_DIR / "runners" / "run_wdbx.py"
        cmd.extend([str(script_path), "--web"])
        if args.port:
            cmd.extend(["--port", str(args.port)])
        if args.theme:
            cmd.extend(["--theme", args.theme])

    elif command_name == "streamlit":
        script_path = SCRIPT_DIR / "runners" / "run_streamlit.py"
        cmd.append(str(script_path))
        if args.port:
            cmd.extend(["--port", str(args.port)])

    elif command_name == "test":
        script_path = SCRIPT_DIR / "runners" / "run_tests.py"
        cmd.append(str(script_path))
        if args.coverage:
            cmd.append("--coverage")
        if args.unit:
            cmd.append("--unit")
        if args.integration:
            cmd.append("--integration")
        if args.xml:
            cmd.append("--junit-xml")
        if args.pattern:
            cmd.extend(["--pattern", args.pattern])

    elif command_name == "benchmark":
        script_path = SCRIPT_DIR / "benchmarks" / "vector_store_benchmark.py"
        cmd.append(str(script_path))
        if args.vectors:
            cmd.extend(["--vectors", str(args.vectors)])
        if args.dimension:
            cmd.extend(["--dimension", str(args.dimension)])
        if args.backend != "all":
            cmd.extend(["--backend", args.backend])
        if args.output:
            cmd.extend(["--output", args.output])

    elif command_name == "lint":
        script_path = SCRIPT_DIR / "linters" / "fix_lint.py"
        cmd.append(str(script_path))
        if args.fix:
            cmd.append("--fix")
        if args.path:
            cmd.append(args.path)

    elif command_name == "db":
        script_path = SCRIPT_DIR / "runners" / "db_manager.py"
        if not os.path.exists(script_path):
            # Fallback location if not found in scripts
            script_path = PROJECT_ROOT / "src" / "wdbx" / "tools" / "db_manager.py"
        cmd.extend([str(script_path), args.action])
        if args.target:
            cmd.extend(["--target", args.target])

    elif command_name == "metrics":
        script_path = SCRIPT_DIR / "runners" / "metrics_collector.py"
        cmd.append(str(script_path))
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        if args.interval:
            cmd.extend(["--interval", str(args.interval)])
        if args.duration:
            cmd.extend(["--duration", str(args.duration)])
        if args.server:
            cmd.extend(["--server", args.server])
        if args.verbose:  # Pass verbose flag to metrics script
            cmd.append("--verbose")

    else:
        # This case should not be reached due to `required=True` in subparsers
        logger.error(f"Unhandled command: {command_name}")
        sys.exit(1)

    return cmd


def run_command(cmd: List[str]) -> int:
    """Run the constructed command using subprocess.run for better safety."""
    logger.info("Running: %s", " ".join(cmd))
    try:
        # Use subprocess.run instead of subprocess.call
        # check=True will raise CalledProcessError if command fails (non-zero exit code)
        result = subprocess.run(cmd, check=False)  # Set check=False to handle errors manually
        if result.returncode != 0:
            logger.error("Command failed with exit code %d", result.returncode)
            return result.returncode
        return 0  # Success
    except FileNotFoundError:
        logger.error("Error: The command or script '%s' was not found.", cmd[1])
        return 1
    except Exception as e:
        logger.error("Error running command: %s", e)
        return 1


def main() -> int:
    """Main entry point for the WDBX tool launcher."""
    args = parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    # Set up environment
    setup_environment(args.env)

    # Override data dir if specified
    if args.data_dir:
        os.environ["WDBX_DATA_DIR"] = args.data_dir
        logger.info("Using data directory: %s", args.data_dir)

    # Build the command based on arguments
    cmd_to_run = build_command(args)

    # Run the command
    exit_code = run_command(cmd_to_run)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
