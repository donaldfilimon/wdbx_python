# wdbx/core/constants.py
"""
Core constants for WDBX.

This module defines constants that are used throughout the WDBX codebase.
It handles environment variable loading with validation and provides
typed constants for use in the application.
"""
import enum
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np

# Type annotations for NumPy arrays
# These are referenced throughout the codebase
from numpy.typing import NDArray

# ============================================================================
# Type aliases for consistent typing
# ============================================================================
VectorType = NDArray[np.float32]
MetadataType = Dict[str, Any]
SimilarityScore = float
VectorId = str


# ============================================================================
# Enums for better type checking and validation
# ============================================================================
class ContentFilterLevel(str, enum.Enum):
    """Valid content filter levels for the system."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def values(cls) -> List[str]:
        """Get all valid values as strings."""
        return [e.value for e in cls]


class ThemeType(str, enum.Enum):
    """Valid UI themes for the system."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"

    @classmethod
    def values(cls) -> List[str]:
        """Get all valid values as strings."""
        return [e.value for e in cls]


class MLBackend(str, enum.Enum):
    """Valid ML backends for vector operations."""

    NUMPY = "numpy"
    TORCH = "torch"
    JAX = "jax"
    FAISS = "faiss"
    AUTO = "auto"

    @classmethod
    def values(cls) -> List[str]:
        """Get all valid values as strings."""
        return [e.value for e in cls]


class IsolationLevel(str, enum.Enum):
    """Valid isolation levels for MVCC."""

    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SNAPSHOT = "snapshot"
    SERIALIZABLE = "serializable"

    @classmethod
    def values(cls) -> List[str]:
        """Get all valid values as strings."""
        return [e.value for e in cls]


# ============================================================================
# Environment variable helpers with improved validation
# ============================================================================
def _getenv_int(
    key: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None
) -> int:
    """
    Get an integer environment variable with a default, handling errors and validation.

    Args:
        key: The environment variable name
        default: Default value to use if not found or invalid
        min_value: Optional minimum allowed value
        max_value: Optional maximum allowed value

    Returns:
        The integer value from the environment or the default
    """
    value_str = os.getenv(key)
    if value_str is None:
        result = default
    else:
        try:
            result = int(value_str)
        except ValueError:
            logging.warning(
                f"Invalid value for environment variable {key}: '{value_str}'. "
                f"Using default {default}."
            )
            result = default

    # Validate range if specified
    if min_value is not None and result < min_value:
        logging.warning(
            f"Value for {key} ({result}) is below minimum ({min_value}). " f"Using minimum value."
        )
        result = min_value

    if max_value is not None and result > max_value:
        logging.warning(
            f"Value for {key} ({result}) is above maximum ({max_value}). " f"Using maximum value."
        )
        result = max_value

    return result


def _getenv_float(
    key: str, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None
) -> float:
    """
    Get a float environment variable with a default, handling errors and validation.

    Args:
        key: The environment variable name
        default: Default value to use if not found or invalid
        min_value: Optional minimum allowed value
        max_value: Optional maximum allowed value

    Returns:
        The float value from the environment or the default
    """
    value_str = os.getenv(key)
    if value_str is None:
        result = default
    else:
        try:
            result = float(value_str)
        except ValueError:
            logging.warning(
                f"Invalid value for environment variable {key}: '{value_str}'. "
                f"Using default {default}."
            )
            result = default

    # Validate range if specified
    if min_value is not None and result < min_value:
        logging.warning(
            f"Value for {key} ({result}) is below minimum ({min_value}). " f"Using minimum value."
        )
        result = min_value

    if max_value is not None and result > max_value:
        logging.warning(
            f"Value for {key} ({result}) is above maximum ({max_value}). " f"Using maximum value."
        )
        result = max_value

    return result


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
    return value_str.lower() in ("true", "1", "yes", "y", "on")


def _getenv_enum(key: str, enum_class: Any, default_value: str) -> str:
    """
    Get an enum environment variable with validation.

    Args:
        key: The environment variable name
        enum_class: The enum class to validate against
        default_value: Default value to use if not found or invalid

    Returns:
        The validated enum value from environment or default
    """
    value_str = os.getenv(key)
    if value_str is None:
        return default_value

    valid_values = cast(Set[str], set(e.value for e in enum_class))
    if value_str not in valid_values:
        logging.warning(
            f"Invalid value for environment variable {key}: '{value_str}'. "
            f"Valid values are: {', '.join(valid_values)}. "
            f"Using default {default_value}."
        )
        return default_value

    return value_str


# ============================================================================
# Configure module logger
# ============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wdbx")

# ============================================================================
# Core system constants
# ============================================================================
VECTOR_DIMENSION: int = _getenv_int("WDBX_VECTOR_DIMENSION", 768, min_value=1)
SHARD_COUNT: int = _getenv_int("WDBX_SHARD_COUNT", 4, min_value=1)  # Default number of shards
DEFAULT_BIAS_SCORE: float = 1.0

# ============================================================================
# Similarity thresholds
# ============================================================================
DEFAULT_SIMILARITY_THRESHOLD: float = _getenv_float(
    "WDBX_DEFAULT_SIMILARITY_THRESHOLD", 0.7, min_value=0.0, max_value=1.0
)
HIGH_SIMILARITY_THRESHOLD: float = _getenv_float(
    "WDBX_HIGH_SIMILARITY_THRESHOLD", 0.85, min_value=0.0, max_value=1.0
)
LOW_SIMILARITY_THRESHOLD: float = _getenv_float(
    "WDBX_LOW_SIMILARITY_THRESHOLD", 0.5, min_value=0.0, max_value=1.0
)

# ============================================================================
# Performance settings
# ============================================================================
DEFAULT_CACHE_SIZE: int = _getenv_int("WDBX_DEFAULT_CACHE_SIZE", 10000, min_value=1)
DEFAULT_BATCH_SIZE: int = _getenv_int("WDBX_DEFAULT_BATCH_SIZE", 100, min_value=1)
MAX_WORKERS: int = _getenv_int("WDBX_MAX_WORKERS", 32, min_value=1)
DEFAULT_THREAD_POOL_SIZE: int = _getenv_int(
    "WDBX_DEFAULT_THREAD_POOL_SIZE", 4, min_value=1
)  # Default thread pool size for parallel operations
DEFAULT_POLLING_INTERVAL: float = _getenv_float(
    "WDBX_DEFAULT_POLLING_INTERVAL", 0.1, min_value=0.01
)  # Default polling interval (in seconds)

# ============================================================================
# Storage settings
# ============================================================================
DEFAULT_MAP_SIZE: int = _getenv_int(
    "WDBX_DEFAULT_MAP_SIZE", 10 * 1024 * 1024 * 1024, min_value=1024 * 1024
)  # 10GB

# ============================================================================
# Environment variable names
# ============================================================================
ENV_DATA_DIR: str = "WDBX_DATA_DIR"
ENV_CACHE_SIZE: str = "WDBX_CACHE_SIZE"
ENV_LOG_LEVEL: str = "WDBX_LOG_LEVEL"
ENV_DEBUG: str = "WDBX_DEBUG"

# ============================================================================
# Default paths
# ============================================================================
DEFAULT_DATA_DIR: str = os.getenv(ENV_DATA_DIR, "./wdbx_data")
CACHE_DIR: str = os.getenv("WDBX_CACHE_DIR", os.path.join(DEFAULT_DATA_DIR, "cache"))
LOG_DIR: str = os.getenv("WDBX_LOG_DIR", os.path.join(DEFAULT_DATA_DIR, "logs"))
TEMP_DIR: str = os.getenv("WDBX_TEMP_DIR", os.path.join(DEFAULT_DATA_DIR, "temp"))

# ============================================================================
# Search constants
# ============================================================================
DEFAULT_TOP_K: int = _getenv_int(
    "WDBX_DEFAULT_TOP_K", 10, min_value=1
)  # Default number of results to return

# ============================================================================
# Cache constants
# ============================================================================
DEFAULT_CACHE_TTL: int = _getenv_int(
    "WDBX_DEFAULT_CACHE_TTL", 3600, min_value=1
)  # Default cache TTL (in seconds)

# ============================================================================
# Network constants
# ============================================================================
HTTP_HOST: str = os.getenv("WDBX_HTTP_HOST", "127.0.0.1")  # Default HTTP server host
HTTP_PORT: int = _getenv_int(
    "WDBX_HTTP_PORT", 5000, min_value=1, max_value=65535
)  # Default HTTP server port
HTTP_WORKERS: int = _getenv_int("WDBX_HTTP_WORKERS", 4, min_value=1)
HTTP_TIMEOUT: int = _getenv_int("WDBX_HTTP_TIMEOUT", 60, min_value=1)
HTTP_MAX_REQUEST_SIZE: int = _getenv_int(
    "WDBX_HTTP_MAX_REQUEST_SIZE", 10 * 1024 * 1024, min_value=1024
)  # 10MB
NETWORK_OVERHEAD: float = _getenv_float("NETWORK_OVERHEAD", 0.02, min_value=0.0)  # seconds

# ============================================================================
# Content filtering
# ============================================================================
CONTENT_FILTER_LEVELS: List[str] = ContentFilterLevel.values()
DEFAULT_CONTENT_FILTER_LEVEL: str = _getenv_enum(
    "WDBX_DEFAULT_CONTENT_FILTER_LEVEL", ContentFilterLevel, ContentFilterLevel.MEDIUM.value
)

# ============================================================================
# Persona constants
# ============================================================================
DEFAULT_PERSONA_COUNT: int = _getenv_int(
    "WDBX_DEFAULT_PERSONA_COUNT", 8, min_value=1
)  # Default number of personas to initialize
DEFAULT_PERSONA_EMBEDDING_DIMENSION: int = _getenv_int(
    "WDBX_DEFAULT_PERSONA_EMBEDDING_DIMENSION", 64, min_value=1
)  # Dimensionality of persona embeddings

# ============================================================================
# Block constants
# ============================================================================
MAX_BLOCK_SIZE: int = _getenv_int(
    "WDBX_MAX_BLOCK_SIZE", 1024 * 1024, min_value=1024
)  # Maximum size of a block in bytes (1MB)
DEFAULT_BLOCK_TIMEOUT: int = _getenv_int(
    "WDBX_DEFAULT_BLOCK_TIMEOUT", 30, min_value=1
)  # Default timeout for block operations (in seconds)

# ============================================================================
# Blockchain constants
# ============================================================================
GENESIS_BLOCK_ID: str = "0000000000000000000000000000000000000000000000000000000000000000"
DIFFICULTY_ADJUSTMENT_INTERVAL: int = _getenv_int(
    "WDBX_DIFFICULTY_ADJUSTMENT_INTERVAL", 2016, min_value=1
)  # Blocks
TARGET_TIMESPAN: int = _getenv_int(
    "WDBX_TARGET_TIMESPAN", 14 * 24 * 60 * 60, min_value=1
)  # Two weeks in seconds
BLOCKCHAIN_DIFFICULTY: int = _getenv_int("BLOCKCHAIN_DIFFICULTY", 2, min_value=1)

# ============================================================================
# MVCC constants
# ============================================================================
DEFAULT_ISOLATION_LEVEL: str = _getenv_enum(
    "WDBX_DEFAULT_ISOLATION_LEVEL", IsolationLevel, IsolationLevel.SNAPSHOT.value
)
MVCC_WRITE_LOCK_TIMEOUT: float = _getenv_float("MVCC_WRITE_LOCK_TIMEOUT", 5.0, min_value=0.1)

# ============================================================================
# Memory monitoring and optimization
# ============================================================================
MEMORY_CHECK_INTERVAL: float = _getenv_float(
    "WDBX_MEMORY_CHECK_INTERVAL", 30.0, min_value=1.0
)  # in seconds
MAX_MEMORY_PERCENT: float = _getenv_float(
    "WDBX_MAX_MEMORY_PERCENT", 85.0, min_value=10.0, max_value=95.0
)  # percentage
MEMORY_OPTIMIZATION_ENABLED: bool = _getenv_bool("WDBX_MEMORY_OPTIMIZATION_ENABLED", True)

# ============================================================================
# Feature flags
# ============================================================================
ENABLE_NEURAL_BACKTRACKING: bool = _getenv_bool("WDBX_ENABLE_NEURAL_BACKTRACKING", True)
ENABLE_DISTRIBUTED_QUERY_PLANNING: bool = _getenv_bool(
    "WDBX_ENABLE_DISTRIBUTED_QUERY_PLANNING", True
)
ENABLE_ATTENTION: bool = _getenv_bool("WDBX_ENABLE_ATTENTION", True)

# ============================================================================
# UI constants
# ============================================================================
UI_THEMES: Dict[str, Dict[str, str]] = {
    ThemeType.DEFAULT.value: {
        "primary_color": "#4a86e8",
        "secondary_color": "#6aa84f",
        "background_color": "#ffffff",
        "text_color": "#333333",
    },
    ThemeType.DARK.value: {
        "primary_color": "#4a86e8",
        "secondary_color": "#6aa84f",
        "background_color": "#2d2d2d",
        "text_color": "#e0e0e0",
    },
    ThemeType.LIGHT.value: {
        "primary_color": "#4a86e8",
        "secondary_color": "#6aa84f",
        "background_color": "#f5f5f5",
        "text_color": "#333333",
    },
}

# ============================================================================
# Security constants
# ============================================================================
AES_KEY_SIZE: int = _getenv_int("AES_KEY_SIZE", 256, min_value=128)

# ============================================================================
# Operation limits
# ============================================================================
MAX_BATCH_SIZE: int = _getenv_int("MAX_BATCH_SIZE", 128, min_value=1)
MAX_RETRIES: int = _getenv_int("MAX_RETRIES", 3, min_value=0)
READ_TIMEOUT: float = _getenv_float("READ_TIMEOUT", 30.0, min_value=0.1)
WRITE_TIMEOUT: float = _getenv_float("WRITE_TIMEOUT", 60.0, min_value=0.1)

# ============================================================================
# ML Backend constants
# ============================================================================
ML_BACKENDS: List[str] = MLBackend.values()
DEFAULT_ML_BACKEND: str = _getenv_enum("WDBX_ML_BACKEND", MLBackend, MLBackend.AUTO.value)


# ============================================================================
# Directory initialization
# ============================================================================
def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    directories = [DEFAULT_DATA_DIR, CACHE_DIR, LOG_DIR, TEMP_DIR]
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Log error but don't necessarily exit; app might handle it
            logging.error(f"Could not create directory {directory}: {e}")


# Initialize directories
ensure_directories_exist()
