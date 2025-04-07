""
WDBX Plugin System

Defines the base interface for all WDBX plugins and provides
a registry mechanism for plugin discovery and management.
""

import abc
import importlib
import logging
import sys
from typing import Dict, List, Optional, Type, Any, Callable

logger = logging.getLogger("wdbx.plugins")

class WDBXPlugin(abc.ABC):
    ""Base class for all WDBX plugins."
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        ""Get the plugin's unique name."
        pass
    
    @property
    def description(self) -> str:
        ""Get the plugin's description."
        return "No description provided"
    
    @property
    def version(self) -> str:
        ""Get the plugin's version."
        return "0.1.0"
    
    @abc.abstractmethod
    def initialize(self, api: Any) -> bool:
        ""
        Initialize the plugin with the WDBX API instance.
        
        Args:
            api: The WDBX API instance.
            
        Returns:
            bool: True if initialization was successful, False otherwise.
        ""
        pass
    
    @abc.abstractmethod
    def shutdown(self) -> bool:
        ""
        Perform cleanup when the plugin is shutting down.
        
        Returns:
            bool: True if shutdown was successful, False otherwise.
        ""
        pass

# Plugin registry
_plugins: Dict[str, Type[WDBXPlugin]] = {}

def register_plugin(plugin_class: Type[WDBXPlugin]) -> None:
    ""
    Register a plugin with WDBX.
    
    Args:
        plugin_class: The plugin class to register.
    ""
    _plugins[plugin_class.name] = plugin_class
    logger.info(f"Registered plugin: {plugin_class.name} v{plugin_class.version}")

def get_plugin(name: str) -> Optional[Type[WDBXPlugin]]:
    ""
    Get a registered plugin by name.
    
    Args:
        name: The name of the plugin to get.
        
    Returns:
        The plugin class, or None if the plugin is not registered.
    ""
    return _plugins.get(name)

def get_all_plugins() -> List[Type[WDBXPlugin]]:
    ""
    Get all registered plugins.
    
    Returns:
        A list of all registered plugin classes.
    ""
    return list(_plugins.values())

def load_plugin_module(module_path: str) -> bool:
    ""
    Load a plugin module by import path.
    
    Args:
        module_path: The import path of the module to load.
        
    Returns:
        bool: True if the module was loaded successfully, False otherwise.
    ""
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return True
    except ImportError as e:
        logger.error(f"Failed to load plugin module {module_path}: {e}")
   except Exception as e:
       logger.error(f"Error: {e}", exc_info=True)
        return False
   except Exception as e:
       logger.error(f"Error: {e}", exc_info=True)

def discover_plugins() -> List[str]:
    ""
    Discover available plugins in the plugins directory.
    
    Returns:
        A list of plugin module names that were discovered.
    ""
    import pkgutil
    
    discovered = []
    
    # Get the package path
    import wdbx.plugins as plugins_pkg
    package_path = plugins_pkg.__path__
    
    # Discover all modules in the plugins package
    for _, name, is_pkg in pkgutil.iter_modules(package_path):
        if not is_pkg and name != "__init__":
            discovered.append(f"wdbx.plugins.{name}")
            
    return discovered 