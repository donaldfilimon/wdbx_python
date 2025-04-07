"""
WDBX Plugins Module

This package contains plugin implementations for the WDBX system.
"""

import os
import sys
import importlib
from typing import Dict, Any, Optional, Type

# Dictionary to store loaded plugins
_loaded_plugins: Dict[str, Type[Any]] = {}


def register_plugin(name: str, plugin_class: Type[Any]) -> None:
    """
    Register a plugin with the WDBX system.
    
    Args:
        name: Unique name of the plugin
        plugin_class: The plugin class to register
    """
    global _loaded_plugins
    _loaded_plugins[name] = plugin_class


def get_plugin(name: str) -> Optional[Type[Any]]:
    """
    Get a registered plugin by name.
    
    Args:
        name: Name of the plugin to retrieve
        
    Returns:
        The plugin class if found, otherwise None
    """
    return _loaded_plugins.get(name)


def list_plugins() -> Dict[str, Type[Any]]:
    """
    List all registered plugins.
    
    Returns:
        Dictionary of plugin names to plugin classes
    """
    return dict(_loaded_plugins)


def load_plugin_module(module_path: str) -> bool:
    """
    Dynamically load a plugin module.
    
    Args:
        module_path: Dot-notation path to the module
        
    Returns:
        True if the plugin was loaded successfully, False otherwise
    """
    try:
        importlib.import_module(module_path)
        return True
    except ImportError as e:
        print(f"Failed to load plugin module {module_path}: {e}")
        return False


# Automatically discover plugins in this directory
def discover_plugins():
    """Automatically discover and load plugins in the plugins directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            try:
                module = importlib.import_module(f"wdbx_plugins.{module_name}")
                
                # If the module has a register_plugin function, call it
                if hasattr(module, 'register_plugin') and callable(module.register_plugin):
                    module.register_plugin()
            except ImportError as e:
                print(f"Failed to import plugin {module_name}: {e}")


# Discover plugins when the module is imported
discover_plugins() 