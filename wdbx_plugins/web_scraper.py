"""
WDBX Web Scraper Plugin

This plugin provides functionality to scrape web content, process it, and store the results
in the WDBX database for search and retrieval.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

# Constants for magic numbers
MIN_CHUNK_SIZE = 100
RATE_LIMIT_WINDOW_SECONDS = 60
HTTP_STATUS_TOO_MANY_REQUESTS = 429
HTTP_STATUS_CLIENT_ERROR_MIN = 400
MAX_FILENAME_LENGTH = 100
MAX_DISPLAY_URL_LENGTH = 40
MAX_DISPLAY_TITLE_LENGTH = 30
MAX_DISPLAY_CHUNK_PREVIEW = 100
MAX_DISPLAY_RESULTS_PREVIEW = 10
MAX_RESULTS_BEFORE_MORE_NOTICE = 10
MAX_CHUNKS_BEFORE_MORE_NOTICE = 3
MIN_PARTS_PLUGIN_COMMAND = 2
DEFAULT_MAX_PAGES = 1000
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_FILE_SIZE_MB = 10

# Check for required dependencies
try:
    import requests
    from bs4 import BeautifulSoup
    from bs4.builder import ParserRejectedMarkup

    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

# Initialize logger
logger = logging.getLogger("WDBX.plugins.web_scraper")

# Global variables
scraper_config: Optional["ScraperConfig"] = None  # Will be initialized in register_commands
rate_limiter: Optional["RateLimiter"] = None  # Will be initialized in register_commands
robots_parser: Optional["RobotsParser"] = None  # Will be initialized in register_commands
domain_counts: Dict[str, int] = {}  # Track counts of pages scraped per domain
scraped_cache: Dict[str, Dict[str, Any]] = {}  # Cache of scraped content
active_scrapes: Set[str] = set()  # Set of active scrape IDs
scrape_status: Dict[str, Dict[str, Any]] = {}  # Status of scrapes


# Initialize embedding class for type hints
class EmbeddingVector:
    """Vector embedding for text with associated metadata."""

    def __init__(self, vector: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new embedding vector.

        Args:
            vector: The vector representation
            metadata: Associated metadata
        """
        self.vector = vector
        self.metadata = metadata if metadata is not None else {}


class WebScraperError(Exception):
    """Base exception for web scraper errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class FetchError(WebScraperError):
    """Exception raised when fetching a URL fails."""

    def __init__(self, url: str, message: str, status_code: Optional[int] = None):
        self.url = url
        self.status_code = status_code
        super().__init__(f"Failed to fetch {url}: {message}")


class FetchTimeoutError(FetchError):
    """Exception raised when fetching a URL times out."""


class FetchRateLimitError(FetchError):
    """Exception raised when rate limit is exceeded."""


class FetchRobotsError(FetchError):
    """Exception raised when robots.txt disallows access."""


class ScrapingError(WebScraperError):
    """Exception raised when scraping content fails."""


class StorageError(WebScraperError):
    """Exception raised when storing scraped content fails."""


class ConfigurationError(WebScraperError):
    """Exception raised when configuration is invalid."""


class ValidationError(WebScraperError):
    """Exception raised when validation fails."""


class RateLimitError(WebScraperError):
    """Exception raised when rate limit is exceeded."""


class RobotsTxtError(WebScraperError):
    """Exception raised when robots.txt parsing fails."""


class WebScraperConfig:
    """Configuration for the Web Scraper Plugin."""

    def __init__(self):
        """Initialize with default values."""
        self.user_agent = "WDBX Web Scraper/1.0"
        self.timeout = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.follow_redirects = True
        self.respect_robots_txt = True
        self.extract_links = True
        self.extract_metadata = True
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.allowed_content_types = ["text/html", "application/xhtml+xml"]
        self.concurrent_requests = 5
        self.requests_per_minute = 60
        self.save_html = False
        self.html_dir = "scraped_html"
        self.max_pages_per_domain = 10


def setup_logging() -> logging.Logger:
    """
    Configure structured logging for the web scraper.

    Returns:
        Logger instance
    """
    logger = logging.getLogger("WDBX.plugins.web_scraper")

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.expanduser("~"), ".wdbx", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create file handler
    log_file = os.path.join(log_dir, "web_scraper.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Set log level
    logger.setLevel(logging.INFO)

    return logger


# Initialize logger
logger = setup_logging()


@dataclass
class ScraperConfig:
    """Configuration for the web scraper."""

    # Network settings
    user_agent: str = "WDBX Web Scraper/1.0"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 2
    follow_redirects: bool = True

    # Scraping settings
    max_pages_per_domain: int = 100
    respect_robots_txt: bool = True
    extract_metadata: bool = True
    extract_links: bool = True

    # Content processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Storage settings
    save_html: bool = True
    html_dir: str = "./wdbx_scraped_html"
    max_file_size_mb: int = 10
    allowed_content_types: Optional[List[str]] = None
    default_embedding_model: str = "model:embed"

    # Rate limiting
    requests_per_minute: int = 60
    concurrent_requests: int = 5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate network settings
        if self.timeout < 1:
            raise ValidationError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValidationError("Max retries must be non-negative")
        if self.retry_delay < 0:
            raise ValidationError("Retry delay must be non-negative")

        # Validate scraping settings
        if self.max_pages_per_domain < 1:
            raise ValidationError("Max pages per domain must be positive")

        # Validate content processing
        if self.chunk_size < MIN_CHUNK_SIZE:
            raise ValidationError(f"Chunk size must be at least {MIN_CHUNK_SIZE} characters")
        if self.chunk_overlap < 0:
            raise ValidationError("Chunk overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError("Chunk overlap must be less than chunk size")

        # Validate storage settings
        if self.max_file_size_mb < 1:
            raise ValidationError("Max file size must be positive")

        # Set default allowed content types if not specified
        if self.allowed_content_types is None:
            self.allowed_content_types = ["text/html", "application/xhtml+xml"]

        # Validate rate limiting
        if self.requests_per_minute < 1:
            raise ValidationError("Requests per minute must be positive")
        if self.concurrent_requests < 1:
            raise ValidationError("Concurrent requests must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "user_agent": self.user_agent,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "follow_redirects": self.follow_redirects,
            "max_pages_per_domain": self.max_pages_per_domain,
            "respect_robots_txt": self.respect_robots_txt,
            "extract_metadata": self.extract_metadata,
            "extract_links": self.extract_links,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "save_html": self.save_html,
            "html_dir": self.html_dir,
            "max_file_size_mb": self.max_file_size_mb,
            "allowed_content_types": self.allowed_content_types,
            "default_embedding_model": self.default_embedding_model,
            "requests_per_minute": self.requests_per_minute,
            "concurrent_requests": self.concurrent_requests,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScraperConfig":
        """
        Create config from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            ScraperConfig instance
        """
        return cls(**data)


# Global configuration
scraper_config = ScraperConfig()

# Global storage for scraped data
scraped_cache = {}


class RateLimiter:
    """Rate limiter for web requests."""

    def __init__(self, requests_per_minute: int, concurrent_requests: int):
        """
        Initialize a new rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            concurrent_requests: Maximum concurrent requests
        """
        self.requests_per_minute = requests_per_minute
        self.concurrent_requests = concurrent_requests
        self.request_times: List[float] = []
        self.semaphore = threading.Semaphore(concurrent_requests)
        self.lock = threading.Lock()  # Add lock for thread safety

    def acquire(self) -> None:
        """Acquire permission to make a request."""
        self.semaphore.acquire()
        now = time.time()

        with self.lock:  # Use lock for thread-safe operations
            # Remove old request times
            self.request_times = [
                t for t in self.request_times if now - t < RATE_LIMIT_WINDOW_SECONDS
            ]

            # If we've hit the rate limit, wait
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.request_times.append(time.time())  # Use current time after potential sleep

    def release(self) -> None:
        """Release the semaphore after request is complete."""
        self.semaphore.release()


class RobotsParser:
    """Parser for robots.txt files."""

    def __init__(self):
        """Initialize the robots.txt parser."""
        self.rules: Dict[str, List[Tuple[str, bool]]] = {}
        self.sitemaps: Dict[str, List[str]] = {}
        self.cache: Dict[str, bool] = {}  # Cache for URL permission results
        self.parsed_domains: Set[str] = set()  # Track domains we've already parsed

    def parse(self, robots_url: str) -> None:
        """
        Parse robots.txt file from URL.

        Args:
            robots_url: URL to the robots.txt file
        """
        domain = _get_domain(robots_url)

        # Skip if already parsed this domain
        if domain in self.parsed_domains:
            return

        try:
            response = requests.get(robots_url, timeout=10)
            response.raise_for_status()
            self.parsed_domains.add(domain)

            current_agent = "*"
            for line in response.text.splitlines():
                line = line.strip().lower()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("user-agent:"):
                    current_agent = line.split(":", 1)[1].strip()
                    if current_agent not in self.rules:
                        self.rules[current_agent] = []
                elif line.startswith("allow:"):
                    path = line.split(":", 1)[1].strip()
                    self.rules.setdefault(current_agent, []).append((path, True))
                elif line.startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    self.rules.setdefault(current_agent, []).append((path, False))
                elif line.startswith("sitemap:"):
                    sitemap = line.split(":", 1)[1].strip()
                    self.sitemaps.setdefault(current_agent, []).append(sitemap)

        except Exception as e:
            logger.warning(f"Error parsing robots.txt from {robots_url}: {e}")

    def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """
        Check if URL is allowed by robots.txt rules.

        Args:
            url: URL to check
            user_agent: User agent to check permissions for

        Returns:
            True if allowed, False if disallowed
        """
        # Check cache first
        cache_key = f"{url}:{user_agent}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.rules:
            self.cache[cache_key] = True
            return True

        parsed_url = urlparse(url)
        path = parsed_url.path or "/"  # Default to root path if empty

        # Check specific user agent rules first
        if user_agent in self.rules:
            for rule_path, is_allowed in self.rules[user_agent]:
                if path.startswith(rule_path):
                    self.cache[cache_key] = is_allowed
                    return is_allowed

        # Check wildcard rules
        if "*" in self.rules:
            for rule_path, is_allowed in self.rules["*"]:
                if path.startswith(rule_path):
                    self.cache[cache_key] = is_allowed
                    return is_allowed

        # Default to allowed if no matching rules
        self.cache[cache_key] = True
        return True


# Global instances
rate_limiter = RateLimiter(scraper_config.requests_per_minute, scraper_config.concurrent_requests)
robots_parser = RobotsParser()


def _check_robots_txt(url: str) -> bool:
    """
    Check if URL is allowed by robots.txt.

    Args:
        url: URL to check

    Returns:
        True if allowed, False if disallowed
    """
    global robots_parser

    if not scraper_config.respect_robots_txt:
        return True

    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    # Parse robots.txt if not already done for this domain
    domain = _get_domain(url)
    if robots_parser and domain not in robots_parser.parsed_domains:
        robots_parser.parse(robots_url)

    # If robots_parser isn't initialized, initialize it now
    if not robots_parser:
        robots_parser = RobotsParser()
        robots_parser.parse(robots_url)

    if not robots_parser:
        logger.warning(f"No robots parser available, allowing URL: {url}")
        return True

    return robots_parser.is_allowed(url, scraper_config.user_agent)


def _get_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: URL to extract domain from

    Returns:
        Domain name
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc


def _ensure_html_dir() -> None:
    """Create HTML directory if it doesn't exist."""
    if scraper_config.save_html:
        html_dir = scraper_config.html_dir
        os.makedirs(html_dir, exist_ok=True)
        logger.info(f"Created HTML directory: {html_dir}")


def _save_scraper_config() -> None:
    """Save scraper configuration to JSON file."""
    try:
        # Create config directory if it doesn't exist
        config_dir = os.path.join(os.path.expanduser("~"), ".wdbx")
        os.makedirs(config_dir, exist_ok=True)

        # Save config to JSON file
        config_file = os.path.join(config_dir, "web_scraper_config.json")
        with open(config_file, "w") as f:
            json.dump(scraper_config.to_dict(), f, indent=4)

        logger.info(f"Saved scraper configuration to {config_file}")
    except Exception as e:
        logger.error(f"Error saving scraper configuration: {e}")


def _load_scraper_config() -> None:
    """Load scraper configuration from JSON file."""
    global scraper_config, rate_limiter

    try:
        config_file = os.path.join(os.path.expanduser("~"), ".wdbx", "web_scraper_config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config_data = json.load(f)

            # Update scraper config
            scraper_config = ScraperConfig.from_dict(config_data)

            # Reinitialize rate limiter with new settings
            rate_limiter = RateLimiter(
                scraper_config.requests_per_minute,
                scraper_config.concurrent_requests
            )

            logger.info(f"Loaded scraper configuration from {config_file}")
        else:
            logger.info("No configuration file found, using defaults")
    except Exception as e:
        logger.error(f"Error loading scraper configuration: {e}")


def register_commands(plugin_registry: Dict[str, Callable]) -> None:
    """
    Register web scraper commands with the CLI.

    Args:
        plugin_registry: Registry to add commands to
    """
    # Register commands
    plugin_registry["scrape:url"] = cmd_scrape_url
    plugin_registry["scrape:site"] = cmd_scrape_site
    plugin_registry["scrape:list"] = cmd_scrape_list
    plugin_registry["scrape:status"] = cmd_scrape_status
    plugin_registry["scrape:search"] = cmd_scrape_search
    plugin_registry["scrape:config"] = cmd_scrape_config
    plugin_registry["scrape"] = cmd_scrape_help

    logger.info(
        "Web Scraper commands registered: scrape:url, scrape:site, scrape:list, "
        "scrape:status, scrape:search, scrape:config"
    )

    # Load config if exists
    _load_scraper_config()

    # Create HTML directory if saving is enabled
    if scraper_config.save_html:
        _ensure_html_dir()
