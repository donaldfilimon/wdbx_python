"""
Configuration management for WDBX.

This module provides a unified interface for configuration management, 
supporting multiple sources (environment variables, configuration files, etc.)
and environments (development, production, testing).
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable

import yaml

from ..utils.logging_utils import get_logger
from ..utils.import_utils import import_later

logger = get_logger("WDBX.Config")


class ConfigSource(Enum):
    """Enum representing different configuration sources."""
    
    ENV_VARS = "env_vars"
    JSON_FILE = "json_file"
    YAML_FILE = "yaml_file"
    PYTHON_MODULE = "python_module"
    DEFAULT = "default"
    MEMORY = "memory"


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


@dataclass
class ConfigItem:
    """
    A configuration item with metadata.
    
    Attributes:
        key: The configuration key
        default: Default value if not provided
        value_type: Expected type of the value
        description: Human-readable description
        sensitive: Whether the value is sensitive (e.g. password)
        source: Where the value came from
        required: Whether the value is required
        validators: List of validator functions
        deprecated: Whether the key is deprecated
        environment_variable: Name of environment variable for this config
    """
    
    key: str
    default: Any = None
    value_type: Any = str
    description: str = ""
    sensitive: bool = False
    source: Optional[ConfigSource] = None
    required: bool = False
    validators: List[Callable[[Any], bool]] = field(default_factory=list)
    deprecated: bool = False
    environment_variable: Optional[str] = None
    
    def __post_init__(self):
        """Set the environment variable name if not provided."""
        if self.environment_variable is None:
            self.environment_variable = f"WDBX_{self.key.upper()}"
    
    def validate(self, value: Any) -> List[str]:
        """
        Validate a value against this config item's constraints.
        
        Args:
            value: The value to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check if required but None
        if self.required and value is None:
            errors.append(f"Config '{self.key}' is required but not set")
            return errors
        
        # Skip other validations if value is None
        if value is None:
            return errors
        
        # Type check
        if self.value_type is not None:
            if self.value_type == bool and isinstance(value, str):
                # Special handling for boolean strings
                if value.lower() not in ('true', 'false', '1', '0', 'yes', 'no', 'y', 'n'):
                    errors.append(f"Config '{self.key}' has invalid boolean string: {value}")
            elif not isinstance(value, self.value_type):
                errors.append(f"Config '{self.key}' should be of type {self.value_type.__name__} "
                              f"but got {type(value).__name__}")
        
        # Run custom validators
        for validator in self.validators:
            try:
                if not validator(value):
                    errors.append(f"Config '{self.key}' failed validation: {validator.__name__}")
            except Exception as e:
                errors.append(f"Config '{self.key}' validator error: {str(e)}")
        
        return errors
    
    def format_value(self, value: Any) -> Any:
        """
        Format a value based on the expected type.
        
        Args:
            value: The value to format
            
        Returns:
            Formatted value
        """
        if value is None:
            return self.default
        
        if self.value_type == bool and isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'y')
        
        if self.value_type == int and isinstance(value, str):
            return int(value)
        
        if self.value_type == float and isinstance(value, str):
            return float(value)
        
        if self.value_type == list and isinstance(value, str):
            return [item.strip() for item in value.split(',')]
        
        return value


class Environment(Enum):
    """Enum representing different deployment environments."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @classmethod
    def from_string(cls, value: str) -> 'Environment':
        """
        Get an Environment enum from a string.
        
        Args:
            value: String representation of the environment
            
        Returns:
            Environment enum value
        """
        try:
            return cls[value.upper()]
        except KeyError:
            # Default to development for unknown environments
            logger.warning(f"Unknown environment: {value}, defaulting to DEVELOPMENT")
            return cls.DEVELOPMENT
    
    @classmethod
    def current(cls) -> 'Environment':
        """
        Get the current environment.
        
        Returns:
            Current environment based on WDBX_ENV or default to DEVELOPMENT
        """
        env_name = os.environ.get("WDBX_ENV", "development")
        return cls.from_string(env_name)


class ConfigManager:
    """
    Configuration manager for WDBX.
    
    Manages configuration from multiple sources with validation, 
    type conversion, and environment-specific values.
    """
    
    def __init__(
        self,
        environment: Optional[Environment] = None,
        config_dir: Optional[Union[str, Path]] = None,
        auto_load: bool = True
    ):
        """
        Initialize a ConfigManager.
        
        Args:
            environment: The environment to use (defaults to Environment.current())
            config_dir: Directory containing configuration files
            auto_load: Whether to automatically load configs on initialization
        """
        self.environment = environment or Environment.current()
        self.config_dir = Path(config_dir) if config_dir else Path(os.environ.get(
            "WDBX_CONFIG_DIR", 
            Path(__file__).parent / "configs"
        ))
        
        # Configuration storage
        self._config_values: Dict[str, Any] = {}
        self._config_items: Dict[str, ConfigItem] = {}
        self._loaded_sources: Set[ConfigSource] = set()
        
        # Import validations lazily to avoid circular imports
        self._has_loaded_validators = False
        
        if auto_load:
            self.load_all()
    
    def register_config(self, config_item: ConfigItem) -> None:
        """
        Register a configuration item.
        
        Args:
            config_item: The configuration item to register
        """
        if config_item.key in self._config_items:
            logger.warning(f"Overriding existing config definition for {config_item.key}")
        
        self._config_items[config_item.key] = config_item
        
        # If we already have a value and it's from a higher priority source, keep it
        if config_item.key in self._config_values and config_item.source == ConfigSource.DEFAULT:
            return
        
        # Otherwise, set the default value
        if config_item.default is not None:
            self.set(config_item.key, config_item.default, source=ConfigSource.DEFAULT)
    
    def register_configs(self, config_items: List[ConfigItem]) -> None:
        """
        Register multiple configuration items.
        
        Args:
            config_items: List of configuration items to register
        """
        for item in config_items:
            self.register_config(item)
    
    def load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        for key, item in self._config_items.items():
            env_var = item.environment_variable
            if env_var in os.environ:
                value = os.environ[env_var]
                # Format the value based on expected type
                formatted_value = item.format_value(value)
                self.set(key, formatted_value, source=ConfigSource.ENV_VARS)
        
        self._loaded_sources.add(ConfigSource.ENV_VARS)
        logger.debug("Loaded configuration from environment variables")
    
    def load_file(self, file_path: Union[str, Path], source_type: ConfigSource) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            source_type: Type of configuration file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.debug(f"Config file does not exist: {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                if source_type == ConfigSource.JSON_FILE:
                    config_data = json.load(f)
                elif source_type == ConfigSource.YAML_FILE:
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config source type: {source_type}")
            
            # Process the loaded configuration
            # First, look for environment-specific settings
            env_key = self.environment.value.lower()
            if env_key in config_data:
                env_data = config_data[env_key]
                self._process_config_dict(env_data, source_type)
            
            # Then process common settings
            if "common" in config_data:
                common_data = config_data["common"]
                self._process_config_dict(common_data, source_type)
            
            # Process any top-level settings
            for key, value in config_data.items():
                if key not in ("common", env_key) and not isinstance(value, dict):
                    if key in self._config_items:
                        formatted_value = self._config_items[key].format_value(value)
                        self.set(key, formatted_value, source=source_type)
            
            self._loaded_sources.add(source_type)
            logger.debug(f"Loaded configuration from {file_path}")
        
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {str(e)}")
    
    def load_json(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
        """
        self.load_file(file_path, ConfigSource.JSON_FILE)
    
    def load_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
        """
        self.load_file(file_path, ConfigSource.YAML_FILE)
    
    def load_python_module(self, module_name: str) -> None:
        """
        Load configuration from a Python module.
        
        Args:
            module_name: Fully qualified name of the module
        """
        try:
            @import_later
            def load_module():
                """
                IMPORTS:
                import importlib
                """
                module = importlib.import_module(module_name)
                config_data = {}
                
                # Get all uppercase attributes as config values
                for attr_name in dir(module):
                    if attr_name.isupper():
                        config_data[attr_name.lower()] = getattr(module, attr_name)
                
                return config_data
            
            config_data = load_module()
            self._process_config_dict(config_data, ConfigSource.PYTHON_MODULE)
            
            self._loaded_sources.add(ConfigSource.PYTHON_MODULE)
            logger.debug(f"Loaded configuration from Python module {module_name}")
        
        except Exception as e:
            logger.error(f"Error loading config from module {module_name}: {str(e)}")
    
    def load_all(self) -> None:
        """Load all configuration from standard locations."""
        # Load environment variables first (lowest priority)
        self.load_env_vars()
        
        # Try to load from config files
        try:
            # Load common configs first
            common_yaml = self.config_dir / "config.yaml"
            common_json = self.config_dir / "config.json"
            
            if common_yaml.exists():
                self.load_yaml(common_yaml)
            elif common_json.exists():
                self.load_json(common_json)
            
            # Then load environment-specific configs
            env_yaml = self.config_dir / f"config.{self.environment.value.lower()}.yaml"
            env_json = self.config_dir / f"config.{self.environment.value.lower()}.json"
            
            if env_yaml.exists():
                self.load_yaml(env_yaml)
            elif env_json.exists():
                self.load_json(env_json)
            
            # Load local overrides last (highest priority)
            local_yaml = self.config_dir / "config.local.yaml"
            local_json = self.config_dir / "config.local.json"
            
            if local_yaml.exists():
                self.load_yaml(local_yaml)
            elif local_json.exists():
                self.load_json(local_json)
        
        except Exception as e:
            logger.error(f"Error loading config files: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if the key doesn't exist
            
        Returns:
            The configuration value or default
        """
        if key in self._config_values:
            return self._config_values[key]
        return default
    
    def set(self, key: str, value: Any, source: Optional[ConfigSource] = None) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key
            value: The value to set
            source: The source of the configuration
        """
        # Check priority - don't override values from higher priority sources
        current_source = None
        if key in self._config_values and key in self._config_items:
            current_source = self._config_items[key].source
        
        if current_source and source:
            # Priority: ENV_VARS > JSON/YAML > PYTHON_MODULE > DEFAULT
            source_priority = {
                ConfigSource.ENV_VARS: 4,
                ConfigSource.JSON_FILE: 3,
                ConfigSource.YAML_FILE: 3,
                ConfigSource.PYTHON_MODULE: 2,
                ConfigSource.DEFAULT: 1,
                ConfigSource.MEMORY: 5  # Memory (programmatic) has highest priority
            }
            
            if source_priority.get(current_source, 0) > source_priority.get(source, 0):
                logger.debug(f"Not overriding {key} from {current_source} with {source}")
                return
        
        # If we have a config item definition, validate and format the value
        if key in self._config_items:
            item = self._config_items[key]
            
            # Update the source
            if source:
                item.source = source
            
            # Format the value
            value = item.format_value(value)
            
            # Validate the value
            errors = item.validate(value)
            if errors:
                error_msg = "; ".join(errors)
                if item.required:
                    raise ConfigValidationError(f"Invalid configuration for {key}: {error_msg}")
                else:
                    logger.warning(f"Invalid configuration for {key}: {error_msg}")
        
        # Store the value
        self._config_values[key] = value
        
        # Log the change (but don't log sensitive values)
        if key in self._config_items and self._config_items[key].sensitive:
            logger.debug(f"Set {key} = [SENSITIVE]")
        else:
            logger.debug(f"Set {key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return dict(self._config_values)
    
    def get_sources(self) -> Dict[str, ConfigSource]:
        """
        Get the sources of all configuration values.
        
        Returns:
            Dictionary mapping configuration keys to their sources
        """
        return {
            key: item.source
            for key, item in self._config_items.items()
            if key in self._config_values and item.source is not None
        }
    
    def validate_all(self) -> List[str]:
        """
        Validate all configuration values.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        for key, item in self._config_items.items():
            value = self.get(key)
            
            item_errors = item.validate(value)
            errors.extend(item_errors)
        
        return errors
    
    def export_to_env_vars(self) -> Dict[str, str]:
        """
        Export all configuration values as environment variables.
        
        Returns:
            Dictionary mapping environment variable names to string values
        """
        env_vars = {}
        
        for key, item in self._config_items.items():
            if key in self._config_values:
                value = self._config_values[key]
                
                # Skip None values
                if value is None:
                    continue
                
                # Convert value to string
                if isinstance(value, bool):
                    str_value = str(value).lower()
                elif isinstance(value, (list, tuple)):
                    str_value = ','.join(str(x) for x in value)
                else:
                    str_value = str(value)
                
                env_vars[item.environment_variable] = str_value
        
        return env_vars
    
    def as_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Get all configuration values as a dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive values
            
        Returns:
            Dictionary of configuration values
        """
        if include_sensitive:
            return self.get_all()
        
        return {
            key: value
            for key, value in self._config_values.items()
            if key not in self._config_items or not self._config_items[key].sensitive
        }
    
    def _process_config_dict(self, config_data: Dict[str, Any], source: ConfigSource) -> None:
        """
        Process a dictionary of configuration values.
        
        Args:
            config_data: Dictionary of configuration values
            source: Source of the configuration
        """
        for key, value in config_data.items():
            # Skip dictionaries for now
            if isinstance(value, dict):
                continue
            
            # Use lowercase keys for consistency
            key = key.lower()
            
            # If the key is registered, format and validate the value
            if key in self._config_items:
                formatted_value = self._config_items[key].format_value(value)
                self.set(key, formatted_value, source=source)
            else:
                # For unregistered keys, just set the value without validation
                self.set(key, value, source=source)
    
    def save_to_file(self, file_path: Union[str, Path], include_sensitive: bool = False) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Path to the file to save to
            include_sensitive: Whether to include sensitive values
        """
        file_path = Path(file_path)
        
        # Get the values to save
        values = self.as_dict(include_sensitive=include_sensitive)
        
        # Create the parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine the file format from the extension
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(values, f, indent=2)
        elif file_path.suffix.lower() in ('.yaml', '.yml'):
            with open(file_path, 'w') as f:
                yaml.dump(values, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.debug(f"Saved configuration to {file_path}")


# Singleton instance for reuse
_CONFIG_MANAGER_INSTANCE: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the configuration manager instance.
    
    Returns:
        The singleton ConfigManager instance
    """
    global _CONFIG_MANAGER_INSTANCE
    
    if _CONFIG_MANAGER_INSTANCE is None:
        _CONFIG_MANAGER_INSTANCE = ConfigManager()
    
    return _CONFIG_MANAGER_INSTANCE


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        key: The configuration key
        default: Default value if the key doesn't exist
        
    Returns:
        The configuration value or default
    """
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    Set a configuration value.
    
    Args:
        key: The configuration key
        value: The value to set
    """
    get_config_manager().set(key, value, source=ConfigSource.MEMORY)


def register_config(config_item: ConfigItem) -> None:
    """
    Register a configuration item.
    
    Args:
        config_item: The configuration item to register
    """
    get_config_manager().register_config(config_item)


def current_environment() -> Environment:
    """
    Get the current environment.
    
    Returns:
        Current environment based on WDBX_ENV or default
    """
    return Environment.current()


# Standard configuration items
DEFAULT_CONFIGS = [
    ConfigItem(
        key="log_level",
        default="INFO",
        value_type=str,
        description="Logging level for the application",
        validators=[lambda x: x.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")]
    ),
    ConfigItem(
        key="data_dir",
        default="./data",
        value_type=str,
        description="Directory for storing application data"
    ),
    ConfigItem(
        key="max_memory_percent",
        default=85.0,
        value_type=float,
        description="Maximum memory usage percentage before optimization",
        validators=[lambda x: 0 < x < 100]
    ),
    ConfigItem(
        key="memory_optimization_enabled",
        default=True,
        value_type=bool,
        description="Whether memory optimization is enabled"
    ),
    ConfigItem(
        key="debug_mode",
        default=False,
        value_type=bool,
        description="Whether to run in debug mode"
    ),
    ConfigItem(
        key="api_key",
        default=None,
        value_type=str,
        description="API key for external services",
        sensitive=True
    ),
    ConfigItem(
        key="max_concurrent_requests",
        default=10,
        value_type=int,
        description="Maximum number of concurrent requests",
        validators=[lambda x: x > 0]
    ),
    ConfigItem(
        key="request_timeout",
        default=30,
        value_type=int,
        description="Timeout for external requests in seconds",
        validators=[lambda x: x > 0]
    ),
    ConfigItem(
        key="embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        value_type=str,
        description="Model to use for embeddings"
    ),
    ConfigItem(
        key="ml_backend",
        default="numpy",
        value_type=str,
        description="Backend to use for machine learning operations",
        validators=[lambda x: x.lower() in ("numpy", "pytorch", "jax", "faiss", "auto")]
    ),
    ConfigItem(
        key="ml_device",
        default="cpu",
        value_type=str,
        description="Device to use for machine learning operations",
        validators=[lambda x: x.lower() in ("cpu", "cuda", "mps", "auto")]
    )
]


# Initialize the config manager with default configs
def _initialize_config():
    """Initialize the configuration manager with default configuration items."""
    manager = get_config_manager()
    manager.register_configs(DEFAULT_CONFIGS)
    return manager 