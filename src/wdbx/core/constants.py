# wdbx/constants.py
"""
Core constants for WDBX.

This module defines constants that are used throughout the WDBX codebase.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Type annotations for NumPy arrays
# These are referenced throughout the codebase
from numpy.typing import NDArray
import numpy as np

# Type aliases for consistent typing
VectorType = NDArray[np.float32]
MetadataType = Dict[str, Any]
SimilarityScore = float
VectorId = str


def _getenv_int(key: str, default: int) -> int:
    """
    Get an integer environment variable with a default, handling errors.
    
    Args:
        key: The environment variable name
        default: Default value to use if not found or invalid
        
    Returns:
        The integer value from the environment or the default
    """
    value_str = os.getenv(key)
    if value_str is None:
        return default
    try:
        return int(value_str)
    except ValueError:
        logging.warning(
            f"Invalid value for environment variable {key}: '{value_str}'. "
            f"Using default {default}."
        )
        return default


def _getenv_float(key: str, default: float) -> float:
    """
    Get a float environment variable with a default, handling errors.
    
    Args:
        key: The environment variable name
        default: Default value to use if not found or invalid
        
    Returns:
        The float value from the environment or the default
    """
    value_str = os.getenv(key)
    if value_str is None:
        return default
    try:
        return float(value_str)
    except ValueError:
        logging.warning(
            f"Invalid value for environment variable {key}: '{value_str}'. "
            f"Using default {default}."
        )
        return default


def _getenv_bool(key: str, default: bool = False) -> bool:
    """
    Get a boolean environment variable with a default.
    
    Args:
        key: The environment variable name
        default: Default value to use if not found
        
    Returns:
        The boolean value from the environment or the default
    """
    value_str = os.getenv(key)
    if value_str is None:
        return default
    return value_str.lower() in ("true", "1", "yes", "y")


# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wdbx")

# Core constants
VECTOR_DIMENSION: int = _getenv_int("WDBX_VECTOR_DIMENSION", 768)

# System configuration
SHARD_COUNT: int = _getenv_int("WDBX_SHARD_COUNT", 4)  # Default number of shards

# Similarity thresholds
DEFAULT_SIMILARITY_THRESHOLD: float = _getenv_float("WDBX_DEFAULT_SIMILARITY_THRESHOLD", 0.7)
HIGH_SIMILARITY_THRESHOLD: float = _getenv_float("WDBX_HIGH_SIMILARITY_THRESHOLD", 0.85)
LOW_SIMILARITY_THRESHOLD: float = _getenv_float("WDBX_LOW_SIMILARITY_THRESHOLD", 0.5)

# Performance settings
DEFAULT_CACHE_SIZE: int = _getenv_int("WDBX_DEFAULT_CACHE_SIZE", 10000)
DEFAULT_BATCH_SIZE: int = _getenv_int("WDBX_DEFAULT_BATCH_SIZE", 100)
MAX_WORKERS: int = _getenv_int("WDBX_MAX_WORKERS", 32)

# Storage settings
DEFAULT_MAP_SIZE: int = _getenv_int("WDBX_DEFAULT_MAP_SIZE", 1024 * 1024 * 1024 * 10)  # 10GB

# Environment variable names
ENV_DATA_DIR: str = "WDBX_DATA_DIR"
ENV_CACHE_SIZE: str = "WDBX_CACHE_SIZE"
ENV_LOG_LEVEL: str = "WDBX_LOG_LEVEL"
ENV_DEBUG: str = "WDBX_DEBUG"

# Default paths
DEFAULT_DATA_DIR: str = os.getenv(ENV_DATA_DIR, "./wdbx_data")

# Search constants
DEFAULT_TOP_K: int = _getenv_int("WDBX_DEFAULT_TOP_K", 10)  # Default number of results to return

# Cache constants
DEFAULT_CACHE_TTL: int = _getenv_int("WDBX_DEFAULT_CACHE_TTL", 3600)  # Default cache TTL (in seconds)

# Network constants
HTTP_HOST: str = os.getenv("WDBX_HTTP_HOST", "127.0.0.1")  # Default HTTP server host
HTTP_PORT: int = _getenv_int("WDBX_HTTP_PORT", 5000)  # Default HTTP server port

# Content filter levels
CONTENT_FILTER_LEVELS: List[str] = ["none", "low", "medium", "high"]
DEFAULT_CONTENT_FILTER_LEVEL: str = os.getenv("WDBX_DEFAULT_CONTENT_FILTER_LEVEL", "medium")

# Persona constants
DEFAULT_PERSONA_COUNT: int = _getenv_int("WDBX_DEFAULT_PERSONA_COUNT", 8)  # Default number of personas to initialize
DEFAULT_PERSONA_EMBEDDING_DIMENSION: int = _getenv_int("WDBX_DEFAULT_PERSONA_EMBEDDING_DIMENSION", 64)  # Dimensionality of persona embeddings

# Block constants
MAX_BLOCK_SIZE: int = _getenv_int("WDBX_MAX_BLOCK_SIZE", 1024 * 1024)  # Maximum size of a block in bytes (1MB)
DEFAULT_BLOCK_TIMEOUT: int = _getenv_int("WDBX_DEFAULT_BLOCK_TIMEOUT", 30)  # Default timeout for block operations (in seconds)

# Blockchain constants
GENESIS_BLOCK_ID: str = "0000000000000000000000000000000000000000000000000000000000000000"
DIFFICULTY_ADJUSTMENT_INTERVAL: int = _getenv_int("WDBX_DIFFICULTY_ADJUSTMENT_INTERVAL", 2016)  # Blocks
TARGET_TIMESPAN: int = _getenv_int("WDBX_TARGET_TIMESPAN", 14 * 24 * 60 * 60)  # Two weeks in seconds
BLOCKCHAIN_DIFFICULTY: int = _getenv_int("BLOCKCHAIN_DIFFICULTY", 2)

# MVCC constants
DEFAULT_ISOLATION_LEVEL: str = os.getenv("WDBX_DEFAULT_ISOLATION_LEVEL", "snapshot")  # Default isolation level for MVCC

# System constants
DEFAULT_THREAD_POOL_SIZE: int = _getenv_int("WDBX_DEFAULT_THREAD_POOL_SIZE", 4)  # Default thread pool size for parallel operations
DEFAULT_POLLING_INTERVAL: float = _getenv_float("WDBX_DEFAULT_POLLING_INTERVAL", 0.1)  # Default polling interval (in seconds)

# Memory monitoring and optimization
MEMORY_CHECK_INTERVAL: float = _getenv_float("WDBX_MEMORY_CHECK_INTERVAL", 30.0)  # in seconds
MAX_MEMORY_PERCENT: float = _getenv_float("WDBX_MAX_MEMORY_PERCENT", 85.0)  # percentage
MEMORY_OPTIMIZATION_ENABLED: bool = _getenv_bool("WDBX_MEMORY_OPTIMIZATION_ENABLED", True)

# Feature flags
ENABLE_NEURAL_BACKTRACKING: bool = _getenv_bool("WDBX_ENABLE_NEURAL_BACKTRACKING", True)  # Enable neural backtracking by default
ENABLE_DISTRIBUTED_QUERY_PLANNING: bool = _getenv_bool("WDBX_ENABLE_DISTRIBUTED_QUERY_PLANNING", True)  # Enable distributed query planning by default
ENABLE_ATTENTION: bool = _getenv_bool("WDBX_ENABLE_ATTENTION", True)  # Enable attention mechanism by default

# UI constants
UI_THEMES: Dict[str, Dict[str, str]] = {
    "default": {
        "primary_color": "#4a86e8",
        "secondary_color": "#6aa84f",
        "background_color": "#ffffff",
        "text_color": "#333333",
    },
    "dark": {
        "primary_color": "#4a86e8",
        "secondary_color": "#6aa84f",
        "background_color": "#2d2d2d",
        "text_color": "#e0e0e0",
    },
    "light": {
        "primary_color": "#4a86e8",
        "secondary_color": "#6aa84f",
        "background_color": "#f5f5f5",
        "text_color": "#333333",
    },
}

# Core system constants
NETWORK_OVERHEAD: float = _getenv_float("NETWORK_OVERHEAD", 0.02)  # seconds
AES_KEY_SIZE: int = _getenv_int("AES_KEY_SIZE", 256)
MVCC_WRITE_LOCK_TIMEOUT: float = _getenv_float("MVCC_WRITE_LOCK_TIMEOUT", 5.0)
MAX_BATCH_SIZE: int = _getenv_int("MAX_BATCH_SIZE", 128)
MAX_RETRIES: int = _getenv_int("MAX_RETRIES", 3)
READ_TIMEOUT: float = _getenv_float("READ_TIMEOUT", 30.0)
WRITE_TIMEOUT: float = _getenv_float("WRITE_TIMEOUT", 60.0)

# Default paths
CACHE_DIR: str = os.getenv("WDBX_CACHE_DIR", os.path.join(DEFAULT_DATA_DIR, "cache"))
LOG_DIR: str = os.getenv("WDBX_LOG_DIR", os.path.join(DEFAULT_DATA_DIR, "logs"))
TEMP_DIR: str = os.getenv("WDBX_TEMP_DIR", os.path.join(DEFAULT_DATA_DIR, "temp"))

# ML Backend constants
ML_BACKENDS: List[str] = ["numpy", "torch", "jax", "faiss"]
DEFAULT_ML_BACKEND: str = os.getenv("WDBX_ML_BACKEND", "auto")

# Ensure directories exist
for directory in [DEFAULT_DATA_DIR, CACHE_DIR, LOG_DIR, TEMP_DIR]:
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Log error but don't necessarily exit; app might handle it
        logging.error(f"Could not create directory {directory}: {e}")

# HTTP server settings
HTTP_WORKERS: int = _getenv_int("WDBX_HTTP_WORKERS", 4)
HTTP_TIMEOUT: int = _getenv_int("WDBX_HTTP_TIMEOUT", 60)
HTTP_MAX_REQUEST_SIZE: int = _getenv_int("WDBX_HTTP_MAX_REQUEST_SIZE", 10 * 1024 * 1024)  # 10MB

# Socket server settings
SOCKET_HOST: str = os.getenv("WDBX_SOCKET_HOST", "127.0.0.1")
SOCKET_PORT: int = _getenv_int("WDBX_SOCKET_PORT", 9090)
SOCKET_BACKLOG: int = _getenv_int("WDBX_SOCKET_BACKLOG", 100)
SOCKET_TIMEOUT: int = _getenv_int("WDBX_SOCKET_TIMEOUT", 30)

# Logging configuration
LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE: str = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "wdbx.log"))
LOG_MAX_SIZE: int = _getenv_int("LOG_MAX_SIZE", 10 * 1024 * 1024)  # 10MB
LOG_BACKUP_COUNT: int = _getenv_int("LOG_BACKUP_COUNT", 5)

# Configure logging with rotating file handler
# Ensure this happens only once
if not logging.getLogger("WDBX").hasHandlers():
    try:
        log_level_val = getattr(logging, LOG_LEVEL_STR, logging.INFO)
        if not isinstance(log_level_val, int):
            log_level_val = logging.INFO  # Fallback if getattr fails weirdly
            logging.warning(f"Invalid LOG_LEVEL '{LOG_LEVEL_STR}'. Using INFO.")

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.setLevel(log_level_val)
        
        # Also add to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_val)
        root_logger.addHandler(file_handler)
        
        # Add console handler if in debug mode
        if _getenv_bool(ENV_DEBUG):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            console_handler.setLevel(log_level_val)
            root_logger.addHandler(console_handler)
    except Exception as e:
        logging.error(f"Failed to configure logging: {e}")

# Persona-related constants
DEFAULT_PERSONAS = ["Abbey", "Aviva", "Abi", "Chloe", "Dana", "Eliza"]
PERSONA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Abbey": {"style": "analytical", "verbosity": "high"},
    "Aviva": {"style": "concise", "verbosity": "low"},
    "Abi": {"style": "friendly", "verbosity": "medium"},
    "Chloe": {"style": "technical", "verbosity": "high"},
    "Dana": {"style": "casual", "verbosity": "medium"},
    "Eliza": {"style": "formal", "verbosity": "high"}
}

# Performance optimization flags
USE_JIT = _getenv_bool("WDBX_USE_JIT")
USE_COMPRESSION = _getenv_bool("WDBX_USE_COMPRESSION")
USE_MMAP = _getenv_bool("WDBX_USE_MMAP", default=True)  # Default MMAP to True
ENABLE_CACHING = _getenv_bool("WDBX_ENABLE_CACHING", default=True)  # Default caching to True
CACHE_TTL = _getenv_int("WDBX_CACHE_TTL", 3600)  # seconds
PARALLELISM = _getenv_int("WDBX_PARALLELISM", os.cpu_count() or 1)

# Security settings
AUTH_REQUIRED = _getenv_bool("WDBX_AUTH_REQUIRED")
JWT_SECRET = os.getenv("WDBX_JWT_SECRET", "change_me_in_production")
JWT_EXPIRATION = _getenv_int("WDBX_JWT_EXPIRATION", 86400)  # 24 hours
ENCRYPTION_ENABLED = _getenv_bool("WDBX_ENCRYPTION_ENABLED")
SSL_CERT_FILE: Optional[str] = os.getenv("WDBX_SSL_CERT") or None  # Use Optional for paths
SSL_KEY_FILE: Optional[str] = os.getenv("WDBX_SSL_KEY") or None

# Network settings
LOG_LEVEL = os.environ.get("WDBX_LOG_LEVEL", "INFO")
HOST = os.environ.get("WDBX_HOST", "127.0.0.1")
PORT = int(os.environ.get("WDBX_PORT", "8080"))
TIMEOUT = int(os.environ.get("WDBX_TIMEOUT", "30"))
MAX_CONNECTIONS = int(os.environ.get("WDBX_MAX_CONNECTIONS", "100"))
NETWORK_OVERHEAD = float(os.environ.get("WDBX_NETWORK_OVERHEAD", "0.1"))

# Security settings
AUTH_ENABLED = os.environ.get("WDBX_AUTH_ENABLED", "false").lower() == "true"
AUTH_TOKEN = os.environ.get("WDBX_AUTH_TOKEN", "")
SSL_ENABLED = os.environ.get("WDBX_SSL_ENABLED", "false").lower() == "true"
SSL_CERT = os.environ.get("WDBX_SSL_CERT", "")
SSL_KEY = os.environ.get("WDBX_SSL_KEY", "")
