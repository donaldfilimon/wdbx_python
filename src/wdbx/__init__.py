"""
WDBX - A powerful data processing and analysis framework.

This package provides tools for data processing, analysis, and visualization.
It includes features for data management, machine learning, and monitoring.

Example:
    >>> from wdbx import Client
    >>> client = Client()
    >>> data = client.get_data()
"""

from .cli import CLI
from .client import Client
from .config import Config
from .data_structures import DataStore
from .health import HealthMonitor
from .mvcc import MVCCManager
from .prometheus import PrometheusMetrics
from .version import __version__

__all__ = [
    "__version__",
    "Config",
    "Client",
    "CLI",
    "PrometheusMetrics",
    "HealthMonitor",
    "DataStore",
    "MVCCManager",
]

# Initialize logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
