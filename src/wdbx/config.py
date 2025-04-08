"""
Configuration management for WDBX.

This module handles all configuration settings for the WDBX framework,
including environment variables, defaults, and validation.
"""

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator


class WDBXConfig(BaseModel):
    """Main configuration class for WDBX."""

    # Database settings
    db_host: str = Field(default="localhost", env="WDBX_DB_HOST")
    db_port: int = Field(default=5432, env="WDBX_DB_PORT")
    db_name: str = Field(default="wdbx", env="WDBX_DB_NAME")
    db_user: str = Field(default="wdbx", env="WDBX_DB_USER")
    db_password: str = Field(default="", env="WDBX_DB_PASSWORD")

    # Cache settings
    cache_size: int = Field(default=1000, env="WDBX_CACHE_SIZE")
    cache_ttl: int = Field(default=3600, env="WDBX_CACHE_TTL")

    # Security settings
    api_key: str = Field(default="", env="WDBX_API_KEY")
    jwt_secret: str = Field(default="", env="WDBX_JWT_SECRET")
    cors_origins: list[str] = Field(default=["*"], env="WDBX_CORS_ORIGINS")

    # Monitoring settings
    prometheus_enabled: bool = Field(default=True, env="WDBX_PROMETHEUS_ENABLED")
    health_check_interval: int = Field(default=30, env="WDBX_HEALTH_CHECK_INTERVAL")

    # Logging settings
    log_level: str = Field(default="INFO", env="WDBX_LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="WDBX_LOG_FILE")

    # Performance settings
    max_workers: int = Field(default=4, env="WDBX_MAX_WORKERS")
    batch_size: int = Field(default=1000, env="WDBX_BATCH_SIZE")

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("db_port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    class Config:
        env_prefix = "WDBX_"
        case_sensitive = False


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("WDBX_CONFIG_PATH")
        self._config: Optional[WDBXConfig] = None

    def load(self) -> WDBXConfig:
        """Load configuration from file and environment."""
        if self._config is not None:
            return self._config

        # Load from environment first
        config = WDBXConfig()

        # Override with file if exists
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path) as f:
                file_config = json.load(f)
                config = WDBXConfig(**{**config.dict(), **file_config})

        self._config = config
        return config

    def save(self, config: WDBXConfig) -> None:
        """Save configuration to file."""
        if not self.config_path:
            raise ValueError("No config path specified")

        config_dict = config.dict(exclude_unset=True)
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        self._config = config

    def get(self) -> WDBXConfig:
        """Get current configuration."""
        return self.load()

    def update(self, **kwargs) -> WDBXConfig:
        """Update configuration with new values."""
        current = self.load()
        updated = current.copy(update=kwargs)
        self.save(updated)
        return updated


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> WDBXConfig:
    """Get the current configuration."""
    return config_manager.get()


def update_config(**kwargs) -> WDBXConfig:
    """Update the configuration with new values."""
    return config_manager.update(**kwargs)
