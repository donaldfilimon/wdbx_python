"""
Configuration module for WDBX.

This module provides functionality for setting up and managing WDBX configurations.
"""
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml

# Set up logging
logger = logging.getLogger("wdbx.config")

# Type variable for WDBXConfig to support proper return typing in class methods
T = TypeVar("T", bound="WDBXConfig")


@dataclass
class WDBXConfig:
    """
    Configuration class for WDBX.
    
    Attributes:
        data_dir: Directory for storing database files
        vector_dimension: Dimension of the vector embeddings
        num_shards: Number of shards to partition the database
        enable_persona_management: Whether to enable the persona management feature
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        plugin_paths: List of directories containing plugin files
        api_keys: Dictionary of API keys for external services
        model_config: Configuration for machine learning models
        cache_enabled: Whether to enable caching
        cache_size: Maximum number of items to store in cache
        cache_ttl: Time-to-live for cached items in seconds (0 means no expiration)
        security_level: Security level (low, medium, high)
    """
    data_dir: str = "./wdbx_data"
    vector_dimension: int = 768
    num_shards: int = 1
    enable_persona_management: bool = False
    log_level: str = "INFO"
    plugin_paths: List[str] = field(default_factory=lambda: ["./wdbx_plugins"])
    api_keys: Dict[str, str] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    security_level: str = "medium"
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        
        # Convert string paths to Path objects and normalize
        self.data_dir = os.path.normpath(os.path.expanduser(self.data_dir))
        self.plugin_paths = [os.path.normpath(os.path.expanduser(p)) for p in self.plugin_paths]
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate numeric values
        if self.vector_dimension <= 0:
            raise ValueError("Vector dimension must be positive")
        if self.num_shards <= 0:
            raise ValueError("Number of shards must be positive")
        if self.cache_size <= 0:
            raise ValueError("Cache size must be positive")
        if self.cache_ttl < 0:
            raise ValueError("Cache TTL must be non-negative")
            
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.log_level}. Must be one of {valid_log_levels}")
        self.log_level = self.log_level.upper()
            
        # Validate security level
        valid_security_levels = ["low", "medium", "high"]
        if self.security_level.lower() not in valid_security_levels:
            raise ValueError(f"Invalid security level: {self.security_level}. Must be one of {valid_security_levels}")
        self.security_level = self.security_level.lower()
            
        # Validate plugin paths
        for path in self.plugin_paths:
            if not isinstance(path, str):
                raise TypeError(f"Plugin path must be a string: {path}")
                
        # Validate API keys
        for key, value in self.api_keys.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(f"API key and value must be strings: {key}={value}")
            # Check for common API key patterns
            if not re.match(r"^[A-Za-z0-9_.\-]+$", key):
                logger.warning(f"API key name contains unusual characters: {key}")
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create a config object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            A new WDBXConfig instance
        """
        # Filter out unknown parameters
        known_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_file(cls: Type[T], file_path: Union[str, Path]) -> T:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            A new WDBXConfig instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Get file extension (lowercase)
        ext = path.suffix.lower()
        
        with open(path) as f:
            if ext == ".json":
                config_dict = json.load(f)
            elif ext in (".yaml", ".yml"):
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def save_to_file(self, file_path: Union[str, Path], format: str = "json") -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path where the configuration file will be saved
            format: File format ('json' or 'yaml')
            
        Raises:
            ValueError: If the format is unsupported
        """
        path = Path(file_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(path, "w") as f:
            if format.lower() == "json":
                json.dump(config_dict, f, indent=2)
            elif format.lower() in ("yaml", "yml"):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
    
    def update(self: T, **kwargs: Any) -> T:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            Self for method chaining
            
        Raises:
            AttributeError: If an invalid parameter is specified
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown configuration parameter: {key}")
            setattr(self, key, value)
            
        # Validate after updates
        self._validate_config()
        
        return self


def create_config(config_dict: Optional[Dict[str, Any]] = None) -> WDBXConfig:
    """
    Create a WDBXConfig object from a configuration dictionary.
    
    Args:
        config_dict: Dictionary containing configuration parameters
        
    Returns:
        A WDBXConfig object with the specified parameters
    """
    # Start with default configuration
    if config_dict is None:
        return WDBXConfig()
    
    return WDBXConfig.from_dict(config_dict)


def load_config_from_file(config_path: str) -> WDBXConfig:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        A WDBXConfig object with the loaded parameters
    """
    return WDBXConfig.from_file(config_path)


def merge_configs(base_config: WDBXConfig, override_config: Dict[str, Any]) -> WDBXConfig:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base values
        
    Returns:
        A new WDBXConfig with merged values
    """
    # Start with base config as dictionary
    merged_dict = base_config.to_dict()
    
    # Override with values from override_config
    for key, value in override_config.items():
        if key in merged_dict:
            if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                merged_dict[key].update(value)
            else:
                # Override scalar values
                merged_dict[key] = value
    
    # Create new config from merged dictionary
    return WDBXConfig.from_dict(merged_dict)


# Added to fix imports
def fields(cls):
    """
    Return a list of dataclass fields.
    
    This is a simple implementation to avoid importing dataclasses.fields
    which might not be available in all Python versions.
    """
    return cls.__dataclass_fields__.values() 