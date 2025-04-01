# wdbx/constants.py
"""
Constants for the WDBX system.

This module contains configuration constants and default values
used throughout the WDBX system.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

# Core system constants
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1024))
SHARD_COUNT = int(os.getenv("SHARD_COUNT", 8))
NETWORK_OVERHEAD = float(os.getenv("NETWORK_OVERHEAD", 0.02))  # seconds
AES_KEY_SIZE = int(os.getenv("AES_KEY_SIZE", 256))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", 0.75))
MVCC_WRITE_LOCK_TIMEOUT = float(os.getenv("MVCC_WRITE_LOCK_TIMEOUT", 5.0))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 128))
BLOCKCHAIN_DIFFICULTY = int(os.getenv("BLOCKCHAIN_DIFFICULTY", 2))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", 30.0))
WRITE_TIMEOUT = float(os.getenv("WRITE_TIMEOUT", 60.0))

# Default paths
DEFAULT_DATA_DIR = os.getenv("WDBX_DATA_DIR", "./wdbx_data")
CACHE_DIR = os.getenv("WDBX_CACHE_DIR", os.path.join(DEFAULT_DATA_DIR, "cache"))
LOG_DIR = os.getenv("WDBX_LOG_DIR", os.path.join(DEFAULT_DATA_DIR, "logs"))
TEMP_DIR = os.getenv("WDBX_TEMP_DIR", os.path.join(DEFAULT_DATA_DIR, "temp"))

# Ensure directories exist
for directory in [DEFAULT_DATA_DIR, CACHE_DIR, LOG_DIR, TEMP_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# HTTP server settings
HTTP_HOST = os.getenv("WDBX_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("WDBX_HTTP_PORT", 8080))
HTTP_WORKERS = int(os.getenv("WDBX_HTTP_WORKERS", 4))
HTTP_TIMEOUT = int(os.getenv("WDBX_HTTP_TIMEOUT", 60))
HTTP_MAX_REQUEST_SIZE = int(os.getenv("WDBX_HTTP_MAX_REQUEST_SIZE", 10 * 1024 * 1024))  # 10MB

# Socket server settings
SOCKET_HOST = os.getenv("WDBX_SOCKET_HOST", "127.0.0.1")
SOCKET_PORT = int(os.getenv("WDBX_SOCKET_PORT", 9090))
SOCKET_BACKLOG = int(os.getenv("WDBX_SOCKET_BACKLOG", 100))
SOCKET_TIMEOUT = int(os.getenv("WDBX_SOCKET_TIMEOUT", 30))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT", 
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "wdbx.log"))
LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", 10 * 1024 * 1024))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

# Configure logging with rotating file handler
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT
)
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger("WDBX")

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
USE_JIT = os.getenv("WDBX_USE_JIT", "false").lower() == "true"
USE_COMPRESSION = os.getenv("WDBX_USE_COMPRESSION", "false").lower() == "true"
USE_MMAP = os.getenv("WDBX_USE_MMAP", "true").lower() == "true"
ENABLE_CACHING = os.getenv("WDBX_ENABLE_CACHING", "true").lower() == "true"
CACHE_TTL = int(os.getenv("WDBX_CACHE_TTL", 3600))  # seconds
PARALLELISM = int(os.getenv("WDBX_PARALLELISM", os.cpu_count() or 1))

# Security settings
AUTH_REQUIRED = os.getenv("WDBX_AUTH_REQUIRED", "false").lower() == "true"
JWT_SECRET = os.getenv("WDBX_JWT_SECRET", "change_me_in_production")
JWT_EXPIRATION = int(os.getenv("WDBX_JWT_EXPIRATION", 86400))  # 24 hours
ENCRYPTION_ENABLED = os.getenv("WDBX_ENCRYPTION_ENABLED", "false").lower() == "true"
SSL_CERT_FILE = os.getenv("WDBX_SSL_CERT", "")
SSL_KEY_FILE = os.getenv("WDBX_SSL_KEY", "")