#!/usr/bin/env python3
"""
WDBX Integrated Linter

This script provides a comprehensive linting solution that combines:
1. Industry-standard linting tools (ruff, black, isort, autoflake)
2. Custom WDBX linters for project-specific issues

Usage:
    python scripts/integrated_linter.py [--target TARGET] [--fix] [--verbose] [--tools TOOLS]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Import lint configuration
sys.path.insert(0, str(Path(__file__).parent))
try:
    import lint_config as config
except ImportError:
    print("Error: Could not import lint_config.py")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_tool_availability(tools: List[str]) -> Dict[str, bool]:
    """Check if requested tools are available on the system."""
    availability = {}
    for tool in tools:
        if tool in ["wdbx_linter", "run_linters", "wdbx_linters"]:
            # Our custom tools should be available as scripts
            script_path = Path(__file__).parent / f"{tool}.py"
            availability[tool] = script_path.exists()
        elif tool == "discord_bot_fixer":
            # Special case for discord bot fixer which has a different filename
            script_path = Path(__file__).parent / "fix_discord_bot.py"
            availability[tool] = script_path.exists()
        else:
            # Check external tools with 'which' (Unix) or 'where' (Windows)
            cmd = "where" if sys.platform == "win32" else "which"
            try:
                result = subprocess.run([cmd, tool], capture_output=True, text=True)
                availability[tool] = result.returncode == 0
            except Exception:
                availability[tool] = False

    return availability


def install_missing_tools(missing_tools: List[str]) -> bool:
    """Try to install missing tools."""
    for tool in missing_tools:
        if tool in config.TOOL_INSTALL_COMMANDS:
            logger.info(f"Installing {tool}...")
            try:
                subprocess.run(
                    config.TOOL_INSTALL_COMMANDS[tool].split(), check=True, capture_output=True
                )
                logger.info(f"Successfully installed {tool}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {tool}: {e}")
                return False
        else:
            logger.warning(f"No installation command found for {tool}")
            return False

    return True


def run_ruff(target: str, fix: bool, verbose: bool) -> bool:
    """Run ruff linter/fixer."""
    if not config.RUFF_CONFIG["enabled"]:
        return True

    logger.info("Running ruff...")

    cmd = ["ruff", "check", target]

    # Add configuration options
    cmd.extend(["--select", ",".join(config.RUFF_CONFIG["select"])])

    if config.RUFF_CONFIG["ignore"]:
        cmd.extend(["--ignore", ",".join(config.RUFF_CONFIG["ignore"])])

    if fix and config.RUFF_CONFIG["fix"]:
        cmd.append("--fix")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            if verbose:
                logger.warning(f"Ruff found issues:\n{result.stdout}")
            else:
                logger.warning("Ruff found issues. Run with --verbose for details.")
            return False
        else:
            logger.info("Ruff ran successfully")
            return True

    except Exception as e:
        logger.error(f"Error running ruff: {e}")
        return False


def run_black(target: str, fix: bool, verbose: bool) -> bool:
    """Run black formatter."""
    if not config.BLACK_CONFIG["enabled"]:
        return True

    logger.info("Running black...")

    cmd = ["black", target]

    # Add configuration options
    cmd.extend(["--line-length", str(config.BLACK_CONFIG["line_length"])])

    if not fix:
        cmd.append("--check")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            if verbose:
                logger.warning(f"Black found formatting issues:\n{result.stdout}")
            else:
                logger.warning("Black found formatting issues. Run with --verbose for details.")
            return False
        else:
            logger.info("Black ran successfully")
            return True

    except Exception as e:
        logger.error(f"Error running black: {e}")
        return False


def run_isort(target: str, fix: bool, verbose: bool) -> bool:
    """Run isort import sorter."""
    if not config.ISORT_CONFIG["enabled"]:
        return True

    logger.info("Running isort...")

    cmd = ["isort", target]

    # Add configuration options
    cmd.extend(["--profile", config.ISORT_CONFIG["profile"]])
    cmd.extend(["--line-length", str(config.ISORT_CONFIG["line_length"])])

    if not fix:
        cmd.append("--check")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            if verbose:
                logger.warning(f"isort found import ordering issues:\n{result.stdout}")
            else:
                logger.warning(
                    "isort found import ordering issues. Run with --verbose for details."
                )
            return False
        else:
            logger.info("isort ran successfully")
            return True

    except Exception as e:
        logger.error(f"Error running isort: {e}")
        return False


def run_autoflake(target: str, fix: bool, verbose: bool) -> bool:
    """Run autoflake to remove unused imports and variables."""
    if not config.AUTOFLAKE_CONFIG["enabled"]:
        return True

    logger.info("Running autoflake...")

    cmd = ["autoflake", "--recursive"]

    if config.AUTOFLAKE_CONFIG["remove_unused_variables"]:
        cmd.append("--remove-unused-variables")

    if config.AUTOFLAKE_CONFIG["remove_all_unused_imports"]:
        cmd.append("--remove-all-unused-imports")

    if config.AUTOFLAKE_CONFIG["remove_duplicate_keys"]:
        cmd.append("--remove-duplicate-keys")

    if fix:
        cmd.append("--in-place")

    cmd.append(target)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            if verbose:
                logger.warning(f"autoflake found issues:\n{result.stdout}")
            else:
                logger.warning("autoflake found issues. Run with --verbose for details.")
            return False
        else:
            logger.info("autoflake ran successfully")
            return True

    except Exception as e:
        logger.error(f"Error running autoflake: {e}")
        return False


def run_wdbx_linters(target: str, fix: bool, verbose: bool, report: bool = False) -> bool:
    """Run WDBX custom linters."""
    logger.info("Running WDBX custom linters...")

    # Check if the run_linters.py script exists (preferred approach)
    run_linters_path = Path(__file__).parent / "run_linters.py"

    if run_linters_path.exists():
        # Use the unified run_linters.py script if available
        cmd = [sys.executable, str(run_linters_path), "--target", target]

        if fix:
            cmd.append("--fix")

        if verbose:
            cmd.append("--verbose")

        if report:
            cmd.append("--report")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                if verbose:
                    logger.warning(f"WDBX linters found issues:\n{result.stdout}")
                    if result.stderr:
                        logger.error(f"Errors:\n{result.stderr}")
                else:
                    logger.warning("WDBX linters found issues. Run with --verbose for details.")
                return False
            else:
                logger.info("WDBX linters ran successfully")
                return True

        except Exception as e:
            logger.error(f"Error running WDBX linters: {e}")
            return False
    else:
        # Fallback to running individual linters if run_linters.py is not available
        success = True

        # Run wdbx_linter.py
        wdbx_linter_path = Path(__file__).parent / "wdbx_linter.py"
        if wdbx_linter_path.exists():
            logger.info("Running general WDBX linter...")
            cmd = [sys.executable, str(wdbx_linter_path), "--target", target]

            if fix:
                cmd.append("--fix")

            if verbose:
                cmd.append("--verbose")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    success = False
                    if verbose:
                        logger.warning(f"WDBX general linter found issues:\n{result.stdout}")
                    else:
                        logger.warning("WDBX general linter found issues.")
                else:
                    logger.info("WDBX general linter ran successfully")

            except Exception as e:
                logger.error(f"Error running WDBX general linter: {e}")
                success = False

        # Run fix_discord_bot.py if target contains Discord bot files
        discord_bot_fixer_path = Path(__file__).parent / "fix_discord_bot.py"
        if discord_bot_fixer_path.exists():
            # Only run if target is appropriate (contains discord_bot.py)
            target_path = Path(target)
            should_run_discord_bot_fixer = False

            if target_path.is_file():
                # If target is a file, check if it's discord_bot.py
                should_run_discord_bot_fixer = target_path.name == "discord_bot.py"
            else:
                # If target is a directory, run if it contains discord_bot.py or wdbx_plugins
                should_run_discord_bot_fixer = (
                    "discord_bot" in target
                    or "wdbx_plugins" in target
                    or Path(target, "discord_bot.py").exists()
                    or Path(target, "wdbx_plugins").exists()
                )

            if should_run_discord_bot_fixer:
                logger.info("Running Discord bot fixer...")
                cmd = [sys.executable, str(discord_bot_fixer_path)]

                if not fix:
                    cmd.append("--check-only")

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if result.returncode != 0:
                        success = False
                        if verbose:
                            logger.warning(f"Discord bot fixer found issues:\n{result.stdout}")
                        else:
                            logger.warning("Discord bot fixer found issues.")
                    else:
                        logger.info("Discord bot fixer ran successfully")

                except Exception as e:
                    logger.error(f"Error running Discord bot fixer: {e}")
                    success = False

        return success


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WDBX Integrated Linting Tool")
    parser.add_argument(
        "--target",
        "-t",
        default=".",
        help="Target directory or file to process (default: current directory)",
    )
    parser.add_argument(
        "--fix",
        "-f",
        action="store_true",
        help="Apply fixes (without this flag, only reports issues)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more detailed output")
    parser.add_argument(
        "--tools", "-l", default="all", help="Comma-separated list of tools to run (default: all)"
    )
    parser.add_argument(
        "--report", "-r", action="store_true", help="Generate HTML report for WDBX linters"
    )
    parser.add_argument(
        "--install-missing", "-i", action="store_true", help="Try to install missing tools"
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine which tools to run
    all_tools = ["ruff", "black", "isort", "autoflake", "wdbx_linters"]
    if args.tools.lower() == "all":
        tools_to_run = all_tools
    else:
        tools_to_run = [t.strip() for t in args.tools.split(",")]

        # Map any alternate names to their correct tool name
        tool_name_map = {
            "discord_bot_fixer": "wdbx_linters",
            "discord": "wdbx_linters",
            "wdbx_linter": "wdbx_linters",
            "run_linters": "wdbx_linters",
        }

        # Replace any alternate names with the standardized name
        for i, tool in enumerate(tools_to_run):
            if tool in tool_name_map:
                tools_to_run[i] = tool_name_map[tool]

        # Remove duplicates while preserving order
        seen = set()
        tools_to_run = [tool for tool in tools_to_run if not (tool in seen or seen.add(tool))]

    # Check tool availability
    tool_availability = check_tool_availability(tools_to_run)
    missing_tools = [tool for tool, available in tool_availability.items() if not available]

    if missing_tools:
        logger.warning(f"The following tools are not available: {', '.join(missing_tools)}")

        if args.install_missing:
            if install_missing_tools(missing_tools):
                logger.info("Successfully installed missing tools")
                # Update availability
                tool_availability = check_tool_availability(tools_to_run)
                missing_tools = [
                    tool for tool, available in tool_availability.items() if not available
                ]
            else:
                logger.error("Failed to install some missing tools")

        # Remove tools that are still missing
        tools_to_run = [tool for tool in tools_to_run if tool_availability.get(tool, False)]

        if not tools_to_run:
            logger.error("No linting tools available. Please install required tools.")
            return 1

    # Run each tool
    success = True

    if "ruff" in tools_to_run:
        success = run_ruff(args.target, args.fix, args.verbose) and success

    if "black" in tools_to_run:
        success = run_black(args.target, args.fix, args.verbose) and success

    if "isort" in tools_to_run:
        success = run_isort(args.target, args.fix, args.verbose) and success

    if "autoflake" in tools_to_run:
        success = run_autoflake(args.target, args.fix, args.verbose) and success

    if "wdbx_linters" in tools_to_run:
        success = run_wdbx_linters(args.target, args.fix, args.verbose, args.report) and success

    # Summary
    if success:
        logger.info("All linting tools completed successfully!")
    else:
        logger.warning("Some linting tools reported issues.")
        if not args.fix:
            logger.info("Consider running with --fix to automatically resolve fixable issues.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
