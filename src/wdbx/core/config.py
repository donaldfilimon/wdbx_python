"""
Configuration handling for WDBX.

This module provides configuration management for the WDBX system, including:
- Configuration loading from files
- Environment variable overrides
- Command line argument processing
- Configuration validation and type checking

Configuration settings control all aspects of the WDBX system including
performance parameters, network settings, and feature flags.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

from .constants import (
    ContentFilterLevel,
    ThemeType,
)

# Create a logger just for configuration
logger = logging.getLogger(__name__)

# Type variable for generic config class handling
T = TypeVar("T", bound="WDBXConfig")


@dataclass
class WDBXConfig:
    """
    Configuration settings for the WDBX system.

    This class defines all configurable parameters for WDBX with sensible
    defaults. Configuration can be loaded from files, environment variables,
    or set programmatically.
    """

    # Core system settings
    vector_dimension: int = 1024
    num_shards: int = 8
    data_dir: str = "./wdbx_data"

    # Performance settings
    parallelism: int = os.cpu_count() or 4
    use_jit: bool = False
    use_compression: bool = False
    use_mmap: bool = True
    enable_caching: bool = True
    cache_size: int = 1024  # Max number of items in LRU cache
    cache_ttl: Optional[int] = 3600  # Cache TTL in seconds (None for no TTL)

    # Server settings
    http_host: str = "127.0.0.1"
    http_port: int = 8080
    http_workers: int = 4
    socket_host: str = "127.0.0.1"
    socket_port: int = 9090

    # Feature flags
    enable_persona_management: bool = True
    enable_content_filter: bool = True
    content_filter_level: str = ContentFilterLevel.MEDIUM.value

    # Security settings
    auth_required: bool = False
    jwt_secret: Optional[str] = None
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None

    # Testing settings
    asyncio_default_fixture_loop_scope: str = "function"

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Plugin settings
    enabled_plugins: List[str] = field(default_factory=list)
    plugin_paths: List[str] = field(default_factory=lambda: ["./wdbx_plugins"])

    # UI settings
    theme: str = ThemeType.DEFAULT.value

    def __post_init__(self) -> None:
        """
        Validate configuration settings after initialization.

        Ensures all directories exist, validates enum values, and checks
        for configuration inconsistencies.
        """
        # Ensure data directory exists
        if self.data_dir:
            os.makedirs(self.data_dir, exist_ok=True)

        # Validate content filter level
        if self.content_filter_level not in ContentFilterLevel.values():
            logger.warning(
                f"Invalid content filter level: {self.content_filter_level}. "
                f"Valid values are: {', '.join(ContentFilterLevel.values())}. "
                f"Using {ContentFilterLevel.MEDIUM.value}."
            )
            self.content_filter_level = ContentFilterLevel.MEDIUM.value

        # Validate theme
        if self.theme not in ThemeType.values():
            logger.warning(
                f"Invalid theme: {self.theme}. "
                f"Valid values are: {', '.join(ThemeType.values())}. "
                f"Using {ThemeType.DEFAULT.value}."
            )
            self.theme = ThemeType.DEFAULT.value

        # Validate cache settings
        if self.enable_caching and self.cache_size <= 0:
            logger.warning(f"Invalid cache_size: {self.cache_size}. Disabling caching.")
            self.enable_caching = False
        if self.cache_ttl is not None and self.cache_ttl <= 0:
            logger.warning(f"Invalid cache_ttl: {self.cache_ttl}. Setting TTL to None.")
            self.cache_ttl = None

        # Validate SSL settings
        if self.ssl_cert_file and not os.path.exists(self.ssl_cert_file):
            logger.warning(f"SSL certificate file not found: {self.ssl_cert_file}")
        if self.ssl_key_file and not os.path.exists(self.ssl_key_file):
            logger.warning(f"SSL key file not found: {self.ssl_key_file}")

        # Validate plugin paths
        for path in self.plugin_paths:
            if not os.path.exists(path):
                logger.warning(f"Plugin path not found: {path}")

        # Validate port ranges
        if not (1 <= self.http_port <= 65535):
            logger.warning(f"Invalid HTTP port: {self.http_port}. Setting to 8080.")
            self.http_port = 8080
        if not (1 <= self.socket_port <= 65535):
            logger.warning(f"Invalid socket port: {self.socket_port}. Setting to 9090.")
            self.socket_port = 9090

        # Validate worker count
        if self.http_workers <= 0:
            logger.warning(f"Invalid worker count: {self.http_workers}. Setting to 4.")
            self.http_workers = 4

        # Check for port collision
        if self.http_port == self.socket_port and self.http_host == self.socket_host:
            logger.warning("HTTP and socket ports are the same. This will cause conflicts.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of all configuration settings
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary of configuration values

        Returns:
            New configuration object with values from the dictionary
        """
        # Filter out keys not in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_env(cls: Type[T], prefix: str = "WDBX_") -> T:
        """
        Create configuration from environment variables.

        Environment variables should be named with the prefix followed by the
        uppercased configuration key, e.g. WDBX_DATA_DIR for data_dir.

        Args:
            prefix: Prefix for environment variables (default: "WDBX_")

        Returns:
            New configuration object with values from environment variables
        """
        # Start with default configuration
        config = cls()

        # Get all field names
        config_fields = {f.name: f for f in fields(cls)}

        # Collect environment variables with the prefix
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

        # Process each environment variable
        for env_name, env_value in env_vars.items():
            # Convert from ENV_VAR_NAME to env_var_name
            config_name = env_name[len(prefix) :].lower()

            # Skip if not a valid config field
            if config_name not in config_fields:
                continue

            # Get the field type and convert the value
            field = config_fields[config_name]
            field_type = field.type

            try:
                # Convert the string value to the appropriate type
                if field_type == bool or field_type == "bool":
                    # Handle boolean values
                    typed_value = env_value.lower() in ("true", "1", "yes", "y", "on")
                elif field_type == int or field_type == "int":
                    typed_value = int(env_value)
                elif field_type == float or field_type == "float":
                    typed_value = float(env_value)
                elif field_type == List[str] or str(field_type).startswith("typing.List"):
                    # Handle lists (comma-separated)
                    typed_value = [item.strip() for item in env_value.split(",") if item.strip()]
                else:
                    # String and other types
                    typed_value = env_value

                # Set the value on the config object
                setattr(config, config_name, typed_value)
                logger.debug(
                    f"Set {config_name} = {typed_value} from environment variable {env_name}"
                )

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to parse environment variable {env_name}={env_value} "
                    f"as {field_type}: {e}"
                )

        # Run validation after setting all values
        config.__post_init__()
        return config

    def save_to_file(self, file_path: str) -> bool:
        """
        Save configuration to a JSON file.

        Args:
            file_path: Path to save the configuration file

        Returns:
            True if successful, False if there was an error
        """
        try:
            config_dir = os.path.dirname(file_path)
            if config_dir:  # Skip for current directory
                os.makedirs(config_dir, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

            logger.info(f"Saved configuration to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Dictionary containing configuration values.
        Returns an empty dictionary if the file doesn't exist or fails to load.
    """
    if not config_path or not os.path.exists(config_path):
        if config_path:
            logger.warning(f"Configuration file not found: {config_path}")
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config_data
    except json.JSONDecodeError as err:
        logger.error(f"Error decoding JSON configuration file {config_path}: {err}")
        return {}
    except Exception as err:
        logger.error(f"Error loading configuration file {config_path}: {err}")
        return {}


def merge_config(cli_args: Dict[str, Any], config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge command-line arguments and configuration file data.
    CLI arguments take precedence over config file values.
    Only non-None CLI arguments override config values.

    Args:
        cli_args: Dictionary of parsed command-line arguments.
        config_data: Dictionary loaded from the configuration file.

    Returns:
        A merged configuration dictionary.
    """
    # Start with config file data as the base
    merged_config = config_data.copy()

    # Override with CLI arguments if they are not None
    for key, value in cli_args.items():
        if value is not None:  # Only override if a value was provided
            merged_config[key] = value

    return merged_config


def create_config(
    cli_args: Optional[Dict[str, Any]] = None,
    config_file: Optional[str] = None,
    use_env: bool = True,
) -> WDBXConfig:
    """
    Create a configuration object from CLI arguments, config file, and environment.

    The priority order is:
    1. Command line arguments (highest)
    2. Environment variables
    3. Configuration file
    4. Default values (lowest)

    Args:
        cli_args: Optional dictionary of command-line arguments
        config_file: Optional path to a JSON configuration file
        use_env: Whether to use environment variables (default: True)

    Returns:
        A WDBXConfig object with the merged configuration settings
    """
    # Load config file if specified
    config_data = {}
    if config_file:
        config_data = load_config(config_file)

    # Start with a config from the file, or defaults if no file
    config = WDBXConfig.from_dict(config_data)

    # Override with environment variables if enabled
    if use_env:
        env_config = WDBXConfig.from_env()
        # Only override values that differ from defaults
        default_config = WDBXConfig()
        for field_obj in fields(WDBXConfig):
            field_name = field_obj.name
            env_value = getattr(env_config, field_name)
            default_value = getattr(default_config, field_name)

            # If the env value is different from the default, use it
            if env_value != default_value:
                setattr(config, field_name, env_value)

    # Override with CLI args if provided
    if cli_args:
        # Only update fields that exist and have non-None values
        valid_fields = {f.name for f in fields(WDBXConfig)}
        for key, value in cli_args.items():
            if key in valid_fields and value is not None:
                setattr(config, key, value)

    # Run validation again after all updates
    config.__post_init__()

    return config


def print_config_help() -> None:
    """
    Print help information about configuration options.

    Displays all available configuration options with their types and default values.
    """
    # Create a default configuration to get default values
    default_config = WDBXConfig()

    print("\nWDBX Configuration Options:")
    print("===========================\n")

    # Group fields by category
    categories = {
        "Core system settings": ["vector_dimension", "num_shards", "data_dir"],
        "Performance settings": [
            "parallelism",
            "use_jit",
            "use_compression",
            "use_mmap",
            "enable_caching",
            "cache_size",
            "cache_ttl",
        ],
        "Server settings": ["http_host", "http_port", "http_workers", "socket_host", "socket_port"],
        "Feature flags": [
            "enable_persona_management",
            "enable_content_filter",
            "content_filter_level",
        ],
        "Security settings": ["auth_required", "jwt_secret", "ssl_cert_file", "ssl_key_file"],
        "Logging settings": ["log_level", "log_file"],
        "Plugin settings": ["enabled_plugins", "plugin_paths"],
        "UI settings": ["theme"],
        "Other settings": ["asyncio_default_fixture_loop_scope"],
    }

    # Track fields that have been displayed
    displayed_fields = set()

    # Display fields by category
    for category, field_names in categories.items():
        print(f"{category}:")
        for field_name in field_names:
            field = next((f for f in fields(WDBXConfig) if f.name == field_name), None)
            if field:
                displayed_fields.add(field_name)
                default_value = getattr(default_config, field_name)
                env_var = f"WDBX_{field_name.upper()}"

                # Format the default value for display
                if isinstance(default_value, str):
                    display_value = f'"{default_value}"'
                elif default_value is None:
                    display_value = "None"
                elif isinstance(default_value, list):
                    if default_value:
                        display_value = f"[{', '.join(repr(x) for x in default_value)}]"
                    else:
                        display_value = "[]"
                else:
                    display_value = str(default_value)

                print(f"  {field_name} ({field.type})")
                print(f"    Default: {display_value}")
                print(f"    Environment: {env_var}")
        print()

    # Display any fields that weren't categorized
    uncategorized = []
    for field in fields(WDBXConfig):
        if field.name not in displayed_fields:
            uncategorized.append(field)

    if uncategorized:
        print("Uncategorized settings:")
        for field in uncategorized:
            default_value = getattr(default_config, field.name)
            env_var = f"WDBX_{field.name.upper()}"
            print(f"  {field.name} ({field.type})")
            print(f"    Default: {default_value}")
            print(f"    Environment: {env_var}")
        print()
