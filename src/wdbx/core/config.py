"""
Configuration handling for WDBX.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WDBXConfig:
    """Configuration settings for the WDBX system."""
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
    content_filter_level: str = "medium"

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
    theme: str = "default"

    def __post_init__(self) -> None:
        """Validate configuration settings."""
        # Ensure data directory exists
        if self.data_dir:
            os.makedirs(self.data_dir, exist_ok=True)

        # Validate content filter level
        valid_filter_levels = ["none", "low", "medium", "high"]
        if self.content_filter_level not in valid_filter_levels:
            logger.warning(
                f"Invalid content filter level: {
                    self.content_filter_level}. Using 'medium'.")
            self.content_filter_level = "medium"

        # Validate theme
        valid_themes = ["default", "dark", "light"]
        if self.theme not in valid_themes:
            logger.warning(f"Invalid theme: {self.theme}. Using 'default'.")
            self.theme = "default"

        # Validate cache settings
        if self.enable_caching and self.cache_size <= 0:
            logger.warning(f"Invalid cache_size: {self.cache_size}. Disabling caching.")
            self.enable_caching = False
        if self.cache_ttl is not None and self.cache_ttl <= 0:
            logger.warning(f"Invalid cache_ttl: {self.cache_ttl}. Setting TTL to None.")
            self.cache_ttl = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WDBXConfig":
        """Create configuration from dictionary."""
        # Filter out keys not in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


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
        with open(config_path) as f:
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
        if value is not None:
            merged_config[key] = value

    return merged_config


def create_config(cli_args: Optional[Dict[str, Any]] = None,
                  config_file: Optional[str] = None) -> WDBXConfig:
    """
    Create a configuration object from CLI arguments and/or a config file.

    Args:
        cli_args: Optional dictionary of command-line arguments
        config_file: Optional path to a JSON configuration file

    Returns:
        A WDBXConfig object with the merged configuration settings
    """
    # Load config file if specified
    config_data = {}
    if config_file:
        config_data = load_config(config_file)

    # Merge with CLI args if provided
    if cli_args:
        merged_data = merge_config(cli_args, config_data)
    else:
        merged_data = config_data

    # Create config object
    return WDBXConfig.from_dict(merged_data)
