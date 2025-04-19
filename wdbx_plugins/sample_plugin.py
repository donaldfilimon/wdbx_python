"""
WDBX Sample Plugin - Demonstrates plugin structure and launch configuration

This plugin demonstrates the structure of a WDBX plugin and includes
examples of how to make it compatible with VS Code launch configurations.
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from wdbx_plugins if available
try:
    from wdbx_plugins import WDBXPlugin
except ImportError:
    # Fallback for direct execution or development
    class WDBXPlugin:
        """Base plugin class if wdbx_plugins is not available."""
        NAME = "base_plugin"
        VERSION = "0.1.0"
        DESCRIPTION = "Base WDBX plugin"

        def __init__(self, config=None):
            self.config = config or {}

        def initialize(self):
            return True

        def shutdown(self):
            return True

# Set up logging
logger = logging.getLogger("wdbx.plugins.sample")


class SamplePlugin(WDBXPlugin):
    """Sample plugin demonstrating WDBX plugin structure and VS Code launch capabilities."""

    NAME = "sample_plugin"
    VERSION = "1.0.0"
    DESCRIPTION = "Sample plugin for demonstration purposes"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the sample plugin with optional configuration."""
        super().__init__(config)
        self.is_initialized = False
        self.debug_mode = os.environ.get("DEBUG", "0") == "1"

        # Demo of accessing environment variables set by launch.json
        self.pythonpath = os.environ.get("PYTHONPATH", "")

        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug(
                f"Sample plugin initialized in DEBUG mode with PYTHONPATH: {self.pythonpath}")

    def initialize(self) -> bool:
        """Initialize the plugin."""
        logger.info(f"Initializing {self.NAME} v{self.VERSION}")
        try:
            # Example initialization that would work with launch.json
            self.is_initialized = True
            logger.info(f"Successfully initialized {self.NAME}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize plugin: {e}")
            return False

    def shutdown(self) -> bool:
        """Clean up resources when shutting down."""
        logger.info(f"Shutting down {self.NAME}")
        self.is_initialized = False
        return True

    def get_sample_data(self) -> Dict[str, Any]:
        """Return sample data that demonstrates plugin functionality."""
        return {
            "name": self.NAME,
            "version": self.VERSION,
            "initialized": self.is_initialized,
            "debug_mode": self.debug_mode,
            "config": self.config,
            "environment": {
                "python_path": self.pythonpath,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data according to plugin logic."""
        if not self.is_initialized:
            logger.warning("Plugin not initialized. Cannot process data.")
            return {"error": "Plugin not initialized"}

        # Sample processing logic
        result = {
            "processed_by": f"{self.NAME} v{self.VERSION}",
            "original_data_keys": list(data.keys()),
            "result": "Successfully processed"
        }

        logger.debug(f"Processed data: {result}")
        return result

    def get_vs_code_launch_examples(self) -> List[Dict[str, Any]]:
        """Return examples of VS Code launch configurations for this plugin."""
        # Get base configurations from parent class
        base_configs = super().get_vs_code_launch_examples()

        # Add plugin-specific configurations
        additional_configs = [
            {
                "name": "Python: Sample Plugin Demo",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/wdbx_plugins/sample_plugin.py",
                "console": "integratedTerminal",
                "args": ["--demo"],
                "env": {
                    "PYTHONPATH": "${workspaceFolder}",
                    "DEBUG": "1"
                }
            },
            {
                "name": "Python: Sample Plugin (Module)",
                "type": "python",
                "request": "launch",
                "module": "wdbx_plugins.sample_plugin",
                "console": "integratedTerminal",
                "justMyCode": False,
                "args": ["--demo"],
                "env": {
                    "PYTHONPATH": "${workspaceFolder}",
                    "DEBUG": "1"
                }
            }
        ]

        # Combine base and additional configurations
        return base_configs + additional_configs


def run_demo(config_path: Optional[str] = None) -> None:
    """Run a demonstration of the sample plugin."""
    # Set up console logging for demo
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    # Load config if provided
    config = None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse configuration file: {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    # Initialize plugin
    plugin = SamplePlugin(config)
    if not plugin.initialize():
        logger.error("Failed to initialize plugin")
        return

    # Display plugin information
    logger.info(f"Sample Plugin {plugin.VERSION} initialized successfully")
    logger.info(f"Debug mode: {'Enabled' if plugin.debug_mode else 'Disabled'}")

    # Example data processing
    sample_data = {
        "example_key": "example_value",
        "numbers": [1, 2, 3, 4, 5],
        "nested": {"a": 1, "b": 2}
    }

    result = plugin.process_data(sample_data)
    logger.info(f"Processing result: {json.dumps(result, indent=2)}")

    # Get and print VS Code launch examples
    if plugin.debug_mode:
        launch_examples = plugin.get_vs_code_launch_examples()
        logger.info("VS Code Launch Configuration Examples:")
        for i, example in enumerate(launch_examples, 1):
            logger.info(f"Example {i}:\n{json.dumps(example, indent=2)}")

    # Shutdown plugin
    plugin.shutdown()
    logger.info("Sample plugin demo completed")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WDBX Sample Plugin Demo")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")

    args = parser.parse_args()

    # Set debug environment variable if specified
    if args.debug:
        os.environ["DEBUG"] = "1"

    # Run demo if requested or no arguments provided
    if args.demo or len(sys.argv) == 1:
        run_demo(args.config)
    else:
        plugin = SamplePlugin()
        if plugin.initialize():
            logger.info(f"Plugin {plugin.NAME} v{plugin.VERSION} initialized successfully")
            logger.info(f"Use --demo to run a demonstration")
            plugin.shutdown()
