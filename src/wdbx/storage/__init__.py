"""
Storage module for WDBX.

This module provides components for storing and retrieving data in the WDBX system,
including vector storage, blockchain-based data integrity, multiversion concurrency
control, and shard management.
"""

import logging
import os
import sys
from importlib import import_module
from typing import Any, Dict, Optional, Type, TypeVar

# Set up logger
logger = logging.getLogger("wdbx.storage")

# Type variable for generic class type hints
T = TypeVar("T")

# Cache for lazy loaded classes
_module_cache: Dict[str, Any] = {}


def _lazy_load_class(
    module_name: str, class_name: str, fallback_class: Optional[Type[T]] = None
) -> Type[T]:
    """
    Lazily load a class from a module.

    Args:
        module_name: Name of the module to import
        class_name: Name of the class to import
        fallback_class: Optional fallback class to return if import fails

    Returns:
        The loaded class or fallback class

    Raises:
        ImportError: If import fails and no fallback is provided
    """
    cache_key = f"{module_name}.{class_name}"
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    try:
        module = import_module(f".{module_name}", "wdbx.storage")
        class_obj = getattr(module, class_name)
        _module_cache[cache_key] = class_obj
        return class_obj
    except (ImportError, AttributeError) as e:
        if fallback_class is not None:
            logger.warning(f"Failed to load {class_name} from {module_name}: {e}. Using fallback.")
            return fallback_class
        raise ImportError(f"Failed to load {class_name} from {module_name}: {e}")


# Lazy-loadable classes
class VectorStore:
    """
    Placeholder for VectorStore that will be lazily loaded.

    This class dynamically forwards attribute access to the real implementation.
    """

    def __new__(cls, *args, **kwargs):
        real_class = _lazy_load_class("vector_store", "VectorStore", VectorStore)
        if real_class is VectorStore:
            # Handle case where we got the fallback (ourselves)
            raise ImportError("VectorStore implementation not available")
        return real_class(*args, **kwargs)


class VectorOperations:
    """
    Placeholder for VectorOperations that will be lazily loaded.

    This class dynamically forwards attribute access to the real implementation.
    """

    @staticmethod
    def __getattr__(name):
        real_class = _lazy_load_class("vector_store", "VectorOperations", VectorOperations)
        if real_class is VectorOperations:
            raise AttributeError(f"VectorOperations.{name} not available")
        return getattr(real_class, name)


class BlockChainManager:
    """
    Placeholder for BlockChainManager that will be lazily loaded.

    This class dynamically forwards attribute access to the real implementation.
    """

    def __new__(cls, *args, **kwargs):
        real_class = _lazy_load_class("blockchain", "BlockChainManager", BlockChainManager)
        if real_class is BlockChainManager:
            raise ImportError("BlockChainManager implementation not available")
        return real_class(*args, **kwargs)


class MVCCTransaction:
    """
    Placeholder for MVCCTransaction that will be lazily loaded.

    This class dynamically forwards attribute access to the real implementation.
    """

    def __new__(cls, *args, **kwargs):
        real_class = _lazy_load_class("mvcc", "MVCCTransaction", MVCCTransaction)
        if real_class is MVCCTransaction:
            raise ImportError("MVCCTransaction implementation not available")
        return real_class(*args, **kwargs)


class MVCCManager:
    """
    Placeholder for MVCCManager that will be lazily loaded.

    This class dynamically forwards attribute access to the real implementation.
    """

    def __new__(cls, *args, **kwargs):
        real_class = _lazy_load_class("mvcc", "MVCCManager", MVCCManager)
        if real_class is MVCCManager:
            raise ImportError("MVCCManager implementation not available")
        return real_class(*args, **kwargs)


class ShardManager:
    """
    Placeholder for ShardManager that will be lazily loaded.

    This class dynamically forwards attribute access to the real implementation.
    """

    def __new__(cls, *args, **kwargs):
        real_class = _lazy_load_class("shard_manager", "ShardManager", ShardManager)
        if real_class is ShardManager:
            raise ImportError("ShardManager implementation not available")
        return real_class(*args, **kwargs)


# Path normalization function for cross-platform compatibility
def normalize_path(path: str) -> str:
    """
    Normalize a file path for cross-platform compatibility.

    Args:
        path: Path to normalize

    Returns:
        Normalized path
    """
    # Convert Windows backslashes to forward slashes for consistency
    normalized = path.replace("\\", "/")
    # Handle Windows drive letters in absolute paths
    if sys.platform == "win32" and len(normalized) > 1 and normalized[1] == ":":
        normalized = "/" + normalized[0].lower() + normalized[1:]
    return normalized


__all__ = [
    "VectorStore",
    "VectorOperations",
    "BlockChainManager",
    "MVCCManager",
    "MVCCTransaction",
    "ShardManager",
    "normalize_path",
]
