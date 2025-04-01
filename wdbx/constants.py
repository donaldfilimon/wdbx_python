# wdbx/constants.py
"""
Constants for the WDBX system.

This module contains configuration constants and default values
used throughout the WDBX system.
"""
import logging
import os

# Core system constants
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1024))
SHARD_COUNT = int(os.getenv("SHARD_COUNT", 8))
NETWORK_OVERHEAD = float(os.getenv("NETWORK_OVERHEAD", 0.02))  # seconds
AES_KEY_SIZE = int(os.getenv("AES_KEY_SIZE", 256))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", 0.75))
MVCC_WRITE_LOCK_TIMEOUT = float(os.getenv("MVCC_WRITE_LOCK_TIMEOUT", 5.0))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 128))
BLOCKCHAIN_DIFFICULTY = int(os.getenv("BLOCKCHAIN_DIFFICULTY", 2))

# Default paths
DEFAULT_DATA_DIR = os.getenv("WDBX_DATA_DIR", "./wdbx_data")

# HTTP server settings
HTTP_HOST = os.getenv("WDBX_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("WDBX_HTTP_PORT", 8080))

# Socket server settings
SOCKET_HOST = os.getenv("WDBX_SOCKET_HOST", "127.0.0.1")
SOCKET_PORT = int(os.getenv("WDBX_SOCKET_PORT", 9090))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WDBX")

# Persona-related constants
DEFAULT_PERSONAS = ["Abbey", "Aviva", "Abi"]

# Performance optimization flags
USE_JIT = os.getenv("WDBX_USE_JIT", "false").lower() == "true"
USE_COMPRESSION = os.getenv("WDBX_USE_COMPRESSION", "false").lower() == "true"