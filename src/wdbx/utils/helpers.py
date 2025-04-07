"""
Helper functions for the WDBX package.

This module contains common utility functions that are
used throughout the WDBX package.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

# Configure module logger
logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: The directory path to ensure exists

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_time_delta(seconds: float) -> str:
    """
    Format a time delta in seconds to a human-readable string.

    Args:
        seconds: The time delta in seconds

    Returns:
        A human-readable string representation
    """
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, remainder = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{remainder}s")

    return " ".join(parts)


def safe_json_load(path: Union[str, Path], default: Any = None) -> Dict[str, Any]:
    """
    Safely load a JSON file, returning a default value if the file
    doesn't exist or is invalid.

    Args:
        path: Path to the JSON file
        default: Default value to return if the file can't be loaded

    Returns:
        The loaded JSON data as a dictionary
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load JSON from {path}: {e}")
        return {} if default is None else default


def safe_json_save(data: Any, path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely save data to a JSON file, creating parent directories if needed.

    Args:
        data: Data to save (must be JSON serializable)
        path: Path to save the JSON file
        indent: Indentation level for the JSON file

    Returns:
        True if save was successful, False otherwise
    """
    try:
        path_obj = Path(path)
        # Create directory if it doesn't exist
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return True
    except (TypeError, OSError) as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        return False


def get_timestamp() -> str:
    """
    Get a formatted timestamp string for the current time.

    Returns:
        Formatted timestamp string (ISO format)
    """
    return datetime.now().isoformat()


def chunked_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to split into chunks
        chunk_size: Maximum size of each chunk

    Returns:
        List of chunks, where each chunk is a list
    """
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def bytes_to_human_readable(size_bytes: int) -> str:
    """
    Convert bytes to a human-readable string (e.g., KB, MB, GB).

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable string representation
    """
    if size_bytes < 0:
        raise ValueError("Size must be non-negative")
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"
