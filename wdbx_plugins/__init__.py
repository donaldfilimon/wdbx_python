"""
WDBX Plugin System - Dynamic plugin loading and management for WDBX.

This module implements a plugin discovery and loading system for WDBX,
allowing extensions to be dynamically loaded at runtime.
"""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Callable

logger = logging.getLogger("wdbx.plugins")

# Define base plugin class for type checking
class WDBXPlugin:
    """Base class for all WDBX plugins."""
    
    # Class properties that should be defined by plugins
    NAME: str = "base_plugin"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Base WDBX plugin"
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the plugin with optional configuration."""
        self.config = config or {}
        
    def initialize(self) -> bool:
        """Initialize the plugin. Should be overridden by child classes."""
        return True
        
    def shutdown(self) -> bool:
        """Clean up resources when shutting down. Should be overridden by child classes."""
        return True
        
    def get_vs_code_launch_examples(self) -> List[Dict[str, Any]]:
        """
        Return examples of VS Code launch configurations for this plugin.
        
        This method should be overridden by plugin implementations to provide
        specific launch configurations tailored to each plugin.
        
        Returns:
            List of dictionaries with launch configuration examples
        """
        return [
            {
                "name": f"WDBX: Run {self.NAME}",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/wdbx_plugins/" + f"{self.NAME}.py",
                "args": ["--config", "${workspaceFolder}/config.json"],
                "console": "integratedTerminal",
                "justMyCode": True,
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                }
            },
            {
                "name": f"WDBX: Debug {self.NAME}",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/wdbx_plugins/" + f"{self.NAME}.py",
                "args": ["--config", "${workspaceFolder}/config.json", "--debug"],
                "console": "integratedTerminal",
                "justMyCode": False,
                "env": {
                    "PYTHONPATH": "${workspaceFolder}",
                    "DEBUG": "1"
                }
            }
        ]

# Global registry of discovered plugins
_plugin_registry: Dict[str, Type[WDBXPlugin]] = {}

def discover_plugins(plugin_dirs: Optional[List[str]] = None) -> Dict[str, Type[WDBXPlugin]]:
    """
    Discover available plugins in the specified directories.
    
    Args:
        plugin_dirs: List of directories to search for plugins. If None, uses default locations.
        
    Returns:
        Dictionary mapping plugin names to plugin classes.
    """
    if plugin_dirs is None:
        # Look in standard locations
        plugin_dirs = [
            # Built-in plugins
            os.path.dirname(os.path.abspath(__file__)),
            # User plugins from current workspace
            os.path.join(os.getcwd(), "wdbx_plugins"),
            # User plugins from home directory
            os.path.expanduser("~/.wdbx/plugins"),
        ]
        
        # Add src/wdbx/plugins if running from repository
        src_plugins = os.path.join(os.getcwd(), "src", "wdbx", "plugins")
        if os.path.exists(src_plugins):
            plugin_dirs.append(src_plugins)
            
    global _plugin_registry
    _plugin_registry = {}
    
    for plugin_dir in plugin_dirs:
        if not os.path.exists(plugin_dir):
            continue
            
        logger.debug(f"Searching for plugins in {plugin_dir}")
        
        # Get all Python files in the directory
        for file_path in Path(plugin_dir).glob("*.py"):
            if file_path.name == "__init__.py":
                continue
                
            try:
                module_name = file_path.stem
                
                # Skip if the module is already loaded
                if module_name in _plugin_registry:
                    continue
                    
                # Import the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        obj.__module__ == module_name and
                        hasattr(obj, "NAME") and 
                        hasattr(obj, "initialize")):
                        
                        # Register the plugin
                        _plugin_registry[obj.NAME] = obj
                        logger.debug(f"Discovered plugin: {obj.NAME} ({obj.DESCRIPTION})")
                        
            except Exception as e:
                logger.error(f"Error loading plugin from {file_path}: {e}")
                
    return _plugin_registry

def get_plugin_class(name: str) -> Optional[Type[WDBXPlugin]]:
    """
    Get a plugin class by name.
    
    Args:
        name: Name of the plugin to get.
        
    Returns:
        Plugin class if found, None otherwise.
    """
    global _plugin_registry
    
    # Ensure plugins are discovered
    if not _plugin_registry:
        discover_plugins()
        
    return _plugin_registry.get(name)

def initialize_all_plugins(config: Optional[Dict] = None) -> Dict[str, WDBXPlugin]:
    """
    Initialize all discovered plugins with the given configuration.
    
    Args:
        config: Configuration dictionary for plugins.
        
    Returns:
        Dictionary mapping plugin names to initialized plugin instances.
    """
    global _plugin_registry
    
    # Ensure plugins are discovered
    if not _plugin_registry:
        discover_plugins()
        
    initialized_plugins = {}
    
    for plugin_name, plugin_class in _plugin_registry.items():
        try:
            plugin_instance = plugin_class(config)
            if plugin_instance.initialize():
                initialized_plugins[plugin_name] = plugin_instance
                logger.info(f"Initialized plugin: {plugin_name}")
            else:
                logger.warning(f"Failed to initialize plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Error initializing plugin {plugin_name}: {e}")
            
    return initialized_plugins

def get_all_launch_configurations() -> Dict[str, Any]:
    """
    Get all available VS Code launch configurations from discovered plugins.
    
    Returns:
        Dictionary with complete launch.json structure containing all plugin configurations
    """
    global _plugin_registry
    
    # Ensure plugins are discovered
    if not _plugin_registry:
        discover_plugins()
    
    # Create base launch.json structure
    launch_config = {
        "version": "0.2.0",
        "configurations": [],
        "compounds": []
    }
    
    # Collect all plugin configurations
    for plugin_name, plugin_class in _plugin_registry.items():
        try:
            # Create temporary instance to get configurations
            plugin_instance = plugin_class()
            if hasattr(plugin_instance, "get_vs_code_launch_examples"):
                examples = plugin_instance.get_vs_code_launch_examples()
                if examples and isinstance(examples, list):
                    for config in examples:
                        if config not in launch_config["configurations"]:
                            launch_config["configurations"].append(config)
        except Exception as e:
            logger.error(f"Error getting launch configurations from {plugin_name}: {e}")
    
    # Add compound configurations for multi-plugin scenarios
    launch_config["compounds"].append({
        "name": "WDBX: Start All Plugins",
        "configurations": [config["name"] for config in launch_config["configurations"] 
                          if "Debug" not in config["name"]]
    })
    
    return launch_config

def export_launch_configurations(output_path: Optional[str] = None) -> str:
    """
    Export all plugin launch configurations to a launch.json file.
    
    Args:
        output_path: Path to save the launch.json file. If None, saves to .vscode/launch.json
        
    Returns:
        Path to the generated launch.json file
    """
    # Get all launch configurations
    launch_config = get_all_launch_configurations()
    
    # Determine output path
    if output_path is None:
        vscode_dir = os.path.join(os.getcwd(), '.vscode')
        os.makedirs(vscode_dir, exist_ok=True)
        output_path = os.path.join(vscode_dir, 'launch.json')
    
    # Write configurations to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(launch_config, f, indent=4)
        logger.info(f"Exported launch configurations to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to export launch configurations: {e}")
        return ""

if __name__ == "__main__":
    import argparse
    
    # Configure argument parser
    parser = argparse.ArgumentParser(description="WDBX Plugin System Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add discover command
    discover_parser = subparsers.add_parser("discover", help="Discover available plugins")
    discover_parser.add_argument("--list", action="store_true", help="List discovered plugins")
    
    # Add launch-config command
    launch_config_parser = subparsers.add_parser("launch-config", help="Generate VS Code launch configurations")
    launch_config_parser.add_argument("--output", "-o", type=str, help="Output file path")
    launch_config_parser.add_argument("--pretty", "-p", action="store_true", help="Pretty print JSON output")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Execute commands
    if args.command == "discover":
        plugins = discover_plugins()
        if args.list:
            print(f"Discovered {len(plugins)} plugins:")
            for name, plugin_class in plugins.items():
                print(f"  - {name} (v{plugin_class.VERSION}): {plugin_class.DESCRIPTION}")
        else:
            print(f"Discovered {len(plugins)} plugins. Use --list to see details.")
    
    elif args.command == "launch-config":
        output_path = args.output
        output_path = export_launch_configurations(output_path)
        
        # Display success message
        if output_path:
            print(f"Successfully generated launch configurations at: {output_path}")
            
            # Print the configurations if requested
            if args.pretty:
                with open(output_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                print(json.dumps(config_data, indent=2))
    
    else:
        parser.print_help()
