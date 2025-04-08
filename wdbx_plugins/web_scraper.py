"""
WDBX Web Scraper Plugin

This plugin provides functionality to scrape web content, process it, and store the results
in the WDBX database for search and retrieval.
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
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
    import threading
    from urllib.parse import urljoin, urlparse

    import requests
    from bs4 import BeautifulSoup
    from bs4.builder import ParserRejectedMarkup

    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

# Initialize logger
logger = logging.getLogger("WDBX.plugins.web_scraper")

# Global variables
scraper_config = None  # Will be initialized in register_commands
rate_limiter = None  # Will be initialized in register_commands
robots_parser = None  # Will be initialized in register_commands
domain_counts = {}  # Track counts of pages scraped per domain
scraped_cache = {}  # Cache of scraped content
active_scrapes = set()  # Set of active scrape IDs
scrape_status = {}  # Status of scrapes


# Initialize embedding class for type hints
class EmbeddingVector:
    def __init__(self, vector, metadata=None):
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


def setup_logging():
    """Configure structured logging for the web scraper."""
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
    allowed_content_types: list[str] = None
    default_embedding_model: str = "model:embed"

    # Rate limiting
    requests_per_minute: int = 60
    concurrent_requests: int = 5

    def __post_init__(self):
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

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
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
    def from_dict(cls, data: dict[str, Any]) -> "ScraperConfig":
        """Create config from dictionary."""
        return cls(**data)


# Global configuration
scraper_config = ScraperConfig()

# Global storage for scraped data
scraped_cache = {}


class RateLimiter:
    """Rate limiter for web requests."""

    def __init__(self, requests_per_minute: int, concurrent_requests: int):
        self.requests_per_minute = requests_per_minute
        self.concurrent_requests = concurrent_requests
        self.request_times: list[float] = []
        self.semaphore = threading.Semaphore(concurrent_requests)
        self.lock = threading.Lock()  # Add lock for thread safety

    def acquire(self):
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

    def release(self):
        """Release the semaphore after request is complete."""
        self.semaphore.release()


class RobotsParser:
    """Parser for robots.txt files."""

    def __init__(self):
        self.rules: dict[str, list[tuple[str, bool]]] = {}
        self.sitemaps: dict[str, list[str]] = {}
        self.cache: dict[str, bool] = {}  # Cache for URL permission results
        self.parsed_domains: set[str] = set()  # Track domains we've already parsed

    def parse(self, robots_url: str) -> None:
        """Parse robots.txt file from URL."""
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
        """Check if URL is allowed by robots.txt rules."""
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
    """Check if URL is allowed by robots.txt."""
    if not scraper_config.respect_robots_txt:
        return True

    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    # Parse robots.txt if not already done for this domain
    domain = _get_domain(url)
    if domain not in robots_parser.parsed_domains:
        robots_parser.parse(robots_url)

    return robots_parser.is_allowed(url, scraper_config.user_agent)


def register_commands(plugin_registry: dict[str, Callable]) -> None:
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


def _ensure_html_dir() -> None:
    """Ensure the HTML directory exists."""
    try:
        dir_path = scraper_config.html_dir
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Web Scraper HTML directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating Web Scraper HTML directory: {e}")


def _get_config_path() -> str:
    """Get the full path to the config file."""
    return os.path.join(os.path.expanduser("~"), ".wdbx", "web_scraper_config.json")


def _load_scraper_config() -> None:
    """Load Web Scraper configuration from file."""
    global scraper_config
    config_path = _get_config_path()
    try:
        if os.path.exists(config_path):
            with open(config_path) as f:
                loaded_config_dict = json.load(f)

                # Create a temporary config instance to validate
                try:
                    # Filter dictionary to only include valid keys
                    valid_keys = scraper_config.to_dict().keys()
                    filtered_config_dict = {
                        k: v for k, v in loaded_config_dict.items() if k in valid_keys
                    }

                    # Create new config object from loaded data
                    loaded_config = ScraperConfig.from_dict(filtered_config_dict)
                    scraper_config = loaded_config  # Update global config
                    logger.info("Loaded and validated Web Scraper configuration from file.")
                except (ValidationError, TypeError) as e:
                    logger.error(f"Invalid configuration found in {config_path}: {e}")
                    raise ConfigurationError(f"Invalid config file: {e}") from e

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Web Scraper config file {config_path}: {e}")
        raise ConfigurationError(f"Corrupted config file: {e}") from e
    except ConfigurationError:  # Re-raise specific errors
        raise
    except Exception as e:
        logger.error(f"Error loading Web Scraper configuration from {config_path}: {e}")
        # Don't raise here, just log and use defaults


def _save_scraper_config() -> None:
    """Save Web Scraper configuration to file."""
    config_path = _get_config_path()
    config_dir = os.path.dirname(config_path)
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(scraper_config.to_dict(), f, indent=2)
        logger.info(f"Saved Web Scraper configuration to {config_path}.")
    except Exception as e:
        logger.error(f"Error saving Web Scraper configuration to {config_path}: {e}")


def _print_config():
    """Print the current configuration."""
    print("\n\033[1;34mWeb Scraper Configuration:\033[0m")
    for key, value in scraper_config.to_dict().items():
        print(f"  \033[1m{key}\033[0m = {value}")


def _update_config_value(key: str, value: str) -> bool:
    """
    Update a single config value with type conversion and validation.

    Args:
        key: Config key to update
        value: New value (as string)

    Returns:
        True if update was successful, False otherwise
    """
    if not hasattr(scraper_config, key):
        print(f"\033[1;31mError: Unknown configuration key: {key}\033[0m")
        return False

    # Get the expected type from the current config attribute
    expected_type = type(getattr(scraper_config, key))
    original_value = getattr(scraper_config, key)
    new_value = None

    try:
        if expected_type is bool:
            new_value = value.lower() in ("true", "yes", "1", "y")
        elif expected_type is int:
            new_value = int(value)
        elif expected_type is str:
            new_value = value
        elif expected_type is list:
            if value.startswith("[") and value.endswith("]"):
                new_value = json.loads(value)
            else:
                new_value = [x.strip() for x in value.split(",")]
            if not isinstance(new_value, list):
                raise ValueError("Invalid list format")
        else:
            error_msg = f"Cannot update configuration key '{key}' of type {expected_type.__name__}"
            print(f"\033[1;31m{error_msg}\033[0m")
            return False

        # Temporarily update the attribute
        setattr(scraper_config, key, new_value)

        # Re-validate the entire config object
        try:
            scraper_config.__post_init__()  # Trigger validation
            print(f"\033[1;32mSet {key} = {new_value}\033[0m")
            return True
        except ValidationError as e:
            # Revert the change if validation fails
            setattr(scraper_config, key, original_value)
            print(f"\033[1;31mError: Invalid value for {key}: {e}\033[0m")
            return False

    except ValueError:
        error_msg = f"Invalid value type for {key}. Expected {expected_type.__name__}"
        print(f"\033[1;31m{error_msg}\033[0m")
        return False
    except json.JSONDecodeError:
        print(f"\033[1;31mError: Invalid JSON format for list value for {key}\033[0m")
        return False


def _normalize_url(url: str) -> str:
    """
    Normalize URL by handling common variations.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL
    """
    # Add protocol if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Remove trailing slash for consistency
    if url.endswith("/"):
        url = url[:-1]

    # Handle URL encoding issues
    parsed = urlparse(url)
    # Ensure the path is properly encoded
    path = parsed.path

    # Reconstruct the URL with proper components
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.params:
        normalized += f";{parsed.params}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"

    return normalized


def _get_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: URL to extract domain from

    Returns:
        Domain name
    """
    parsed = urlparse(url)
    return parsed.netloc


def _safe_filename(url: str) -> str:
    """
    Convert URL to safe filename.

    Args:
        url: URL to convert

    Returns:
        Safe filename
    """
    # Remove protocol and replace unsafe chars with underscore
    safe_name = re.sub(r"^https?://", "", url)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_name)
    # Limit length
    if len(safe_name) > MAX_FILENAME_LENGTH:
        safe_name = safe_name[:MAX_FILENAME_LENGTH]
    return safe_name


def _save_html_content(url: str, html: str) -> Optional[str]:
    """
    Save HTML content to file.

    Args:
        url: Source URL
        html: HTML content

    Returns:
        Path to saved file or None on error
    """
    if not scraper_config.save_html:
        return None

    try:
        filename = f"{_safe_filename(url)}_{int(time.time())}.html"
        filepath = os.path.join(scraper_config.html_dir, filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return filepath
    except Exception as e:
        logger.error(f"Error saving HTML content for {url}: {e}")
        return None


@dataclass
class ScrapedContent:
    """Container for scraped content and metadata."""

    url: str
    title: Optional[str]
    description: Optional[str]
    content: str
    metadata: dict[str, Any]
    chunks: list[str]
    html_path: Optional[str]
    scrape_id: str
    timestamp: float = field(default_factory=time.time)


class ContentProcessor:
    """Process and chunk web content."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        return text.strip()

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # More robust sentence splitting
        # Look for period, question mark, or exclamation followed by space or newline
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> list[str]:
        """Create overlapping chunks of text."""
        if not text:
            return []

        chunks = []
        sentences = self.split_into_sentences(text)

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep some sentences for overlap
                    overlap_size = 0
                    overlap_chunks = []
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) > self.chunk_overlap:
                            break
                        overlap_chunks.insert(0, s)
                        overlap_size += len(s)
                    current_chunk = overlap_chunks
                    current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


def _extract_metadata(url: str, soup: BeautifulSoup) -> dict[str, Any]:
    """
    Extract metadata from HTML content.

    Args:
        url: Source URL
        soup: BeautifulSoup object

    Returns:
        Dictionary of metadata
    """
    metadata = {
        "url": url,
        "domain": _get_domain(url),
        "scraped_at": datetime.now().isoformat(),
    }

    # Extract title
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        metadata["title"] = title_tag.string.strip()

    # Extract meta description
    description = soup.find("meta", attrs={"name": "description"})
    if description and description.get("content"):
        metadata["description"] = description["content"].strip()

    # Extract meta keywords
    keywords = soup.find("meta", attrs={"name": "keywords"})
    if keywords and keywords.get("content"):
        metadata["keywords"] = [k.strip() for k in keywords["content"].split(",")]

    # Extract Open Graph metadata
    for og_prop in ["title", "description", "site_name", "type", "image", "url"]:
        og_tag = soup.find("meta", property=f"og:{og_prop}")
        if og_tag and og_tag.get("content"):
            metadata[f"og_{og_prop}"] = og_tag["content"].strip()

    # Extract author information
    author = soup.find("meta", attrs={"name": "author"})
    if author and author.get("content"):
        metadata["author"] = author["content"].strip()

    # Extract publication date
    pub_date = soup.find("meta", property="article:published_time")
    if pub_date and pub_date.get("content"):
        metadata["published_date"] = pub_date["content"].strip()

    # Extract structured data
    structured_data = soup.find_all("script", type="application/ld+json")
    if structured_data:
        metadata["structured_data"] = []
        for script in structured_data:
            try:
                if script.string:
                    data = json.loads(script.string)
                    metadata["structured_data"].append(data)
            except json.JSONDecodeError:
                continue

    # Extract language
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        metadata["language"] = html_tag.get("lang")

    # Extract canonical URL
    canonical = soup.find("link", attrs={"rel": "canonical"})
    if canonical and canonical.get("href"):
        metadata["canonical_url"] = canonical["href"]

    return metadata


def _extract_links(url: str, soup: BeautifulSoup) -> list[str]:
    """
    Extract links from HTML content.

    Args:
        url: Source URL
        soup: BeautifulSoup object

    Returns:
        List of normalized URLs
    """
    links = []
    base_url = url

    # Check for base tag
    base_tag = soup.find("base", href=True)
    if base_tag:
        base_url = base_tag["href"]

    # Extract all anchor tags with href attribute
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        # Skip empty, javascript, mailto and anchor links
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue

        # Convert relative URLs to absolute
        if not href.startswith(("http://", "https://")):
            href = urljoin(base_url, href)

        # Normalize URL
        normalized_url = _normalize_url(href)

        # Add to list if not already included
        if normalized_url not in links:
            links.append(normalized_url)

    return links


def _extract_text_content(soup: BeautifulSoup) -> str:
    """
    Extract and clean text content from BeautifulSoup object.

    Args:
        soup: BeautifulSoup object

    Returns:
        Cleaned text content
    """
    # Create a copy of the soup to avoid modifying the original
    soup_copy = BeautifulSoup(str(soup), "html.parser")

    # Remove script, style, and other non-content elements
    for element in soup_copy(
        ["script", "style", "header", "footer", "nav", "iframe", "noscript", "aside"]
    ):
        element.decompose()

    # Remove hidden elements
    for element in soup_copy.find_all(
        style=lambda value: value and "display:none" in value.replace(" ", "")
    ):
        element.decompose()

    # Get text with better spacing around block elements
    for tag in soup_copy.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        if tag.string:
            tag.string.replace_with(f"{tag.string} ")
        else:
            # Insert space after the tag
            tag.append(" ")

    # Get text
    text = soup_copy.get_text(separator=" ", strip=True)

    # Clean up text (remove extra whitespace)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    processor = ContentProcessor(chunk_size, chunk_overlap)
    return processor.create_chunks(text)


def _create_embedding(
    db: Any, text: str, metadata: dict[str, Any] = None
) -> Optional[Any]:
    """
    Create embedding from text using configured model.

    Args:
        db: WDBX database instance
        text: Text to create embedding from
        metadata: Optional metadata to include

    Returns:
        Embedding vector or None on error
    """
    # Skip empty text
    if not text or len(text.strip()) == 0:
        logger.warning("Attempted to create embedding from empty text")
        return None

    try:
        # We'll use the model:embed command if available, otherwise fall back to direct embedding
        embedding_model = scraper_config.default_embedding_model

        if embedding_model.startswith("model:"):
            # Use model repository plugin
            logger.debug(f"Creating embedding using {embedding_model}")
            return _dispatch_to_plugin(db, embedding_model, text, metadata or {})
        # Try direct embedding through available plugins
        logger.debug("Creating embedding using direct plugin call")
        # Here we would dispatch to specific plugins like OpenAI, HuggingFace, etc.
        logger.error("Direct embedding not implemented, configure a model:embed provider")
        return None
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return None


def _dispatch_to_plugin(db: Any, command: str, text: str, metadata: dict[str, Any] = None) -> Any:
    """
    Dispatch to another plugin command.

    Args:
        db: WDBX database instance
        command: Command to dispatch to
        text: Text argument
        metadata: Optional metadata

    Returns:
        Result from plugin or None on error
    """
    try:
        # Extract plugin and subcommand
        parts = command.split(":", 1)
        if len(parts) != MIN_PARTS_PLUGIN_COMMAND:
            logger.error(f"Invalid plugin command format: {command}")
            return None

        plugin, subcommand = parts

        # Build full command
        if plugin == "model" and subcommand == "embed":
            # Special case for model:embed which needs metadata
            from wdbx_plugins.model_repo import cmd_model_embed

            return cmd_model_embed(db, text, metadata=metadata)
        # Get the function from the plugin registry
        plugin_registry = getattr(db, "plugin_registry", {})
        if command in plugin_registry:
            plugin_func = plugin_registry[command]
            return plugin_func(db, text)
        logger.error(f"Plugin command not found: {command}")
        return None
    except ImportError:
        logger.error(f"Could not import plugin for: {command}")
        return None
    except Exception as e:
        logger.error(f"Error dispatching to plugin {command}: {e}")
        return None


def _store_scraped_data(db: Any, content: ScrapedContent, response_info: dict[str, Any]) -> str:
    """
    Store scraped data in database and cache.

    Args:
        db: WDBX database instance
        content: ScrapedContent object
        response_info: Response information dictionary

    Returns:
        ID of the scraped document
    """
    # Add response info to metadata
    full_metadata = content.metadata.copy()
    full_metadata.update(
        {
            "response_info": response_info,
            "chunk_count": len(content.chunks),
        }
    )

    if content.html_path:
        full_metadata["html_path"] = content.html_path

    # Store in cache
    scraped_cache[content.scrape_id] = {
        "url": content.url,
        "metadata": full_metadata,
        "chunks": content.chunks,
        "timestamp": content.timestamp,
    }

    # Store embeddings in database if available
    if db:
        try:
            # Create collection for web scrapes if it doesn't exist
            collection_name = "web_scrapes"
            if hasattr(db, "create_collection"):
                db.create_collection(collection_name, exist_ok=True)

            # Store each chunk with its metadata
            for i, chunk in enumerate(content.chunks):
                chunk_metadata = full_metadata.copy()
                chunk_metadata.update({"chunk_index": i, "chunk_total": len(content.chunks)})

                # Create embedding
                embedding = _create_embedding(db, chunk, chunk_metadata)

                if embedding:
                    # Store in database
                    if hasattr(db, "add_embedding"):
                        db.add_embedding(embedding, collection=collection_name)
                    elif hasattr(db, "store"):
                        db.store(embedding, collection=collection_name)
                    else:
                        logger.warning("Database does not support storing embeddings directly")
        except Exception as e:
            logger.error(f"Error storing scraped data in database: {e}")
            raise StorageError(f"Failed to store scraped data: {e}") from e

    return content.scrape_id


def _fetch_url(url: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    Fetch URL with retries, rate limiting, and error handling.

    Args:
        url: URL to fetch

    Returns:
        Tuple of (html_content, response_info) or (None, error_info) on failure
    """
    headers = {
        "User-Agent": scraper_config.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    normalized_url = _normalize_url(url)
    retries = scraper_config.max_retries
    delay = scraper_config.retry_delay

    response_info = {
        "url": normalized_url,
        "final_url": normalized_url,
        "status_code": None,
        "content_type": None,
        "headers": {},
        "size_bytes": 0,
        "fetch_time_sec": 0,
        "retries": 0,
        "error": None,
    }

    # Check robots.txt first
    if not _check_robots_txt(normalized_url):
        raise FetchRobotsError(normalized_url, "URL disallowed by robots.txt")

    for attempt in range(retries + 1):
        if attempt > 0:
            logger.info(f"Retry {attempt}/{retries} for {normalized_url} (waiting {delay}s)")
            time.sleep(delay)
            # Exponential backoff
            delay *= 1.5

        response_info["retries"] = attempt

        try:
            # Acquire rate limiter
            rate_limiter.acquire()

            start_time = time.time()

            response = requests.get(
                normalized_url,
                headers=headers,
                timeout=scraper_config.timeout,
                allow_redirects=scraper_config.follow_redirects,
            )

            fetch_time = time.time() - start_time
            response_info["fetch_time_sec"] = round(fetch_time, 2)
            response_info["status_code"] = response.status_code
            response_info["final_url"] = response.url
            response_info["headers"] = dict(response.headers)
            response_info["content_type"] = (
                response.headers.get("Content-Type", "").split(";")[0].strip()
            )

            # Check for rate limiting
            if response.status_code == HTTP_STATUS_TOO_MANY_REQUESTS:  # Too Many Requests
                raise FetchRateLimitError(
                    normalized_url, "Rate limit exceeded", HTTP_STATUS_TOO_MANY_REQUESTS
                ) from e

            # Check if status code indicates success
            if response.status_code >= HTTP_STATUS_CLIENT_ERROR_MIN:
                response_info["error"] = f"HTTP error: {response.status_code}"
                logger.warning(f"HTTP error {response.status_code} for {normalized_url}")
                continue

            # Check content type
            allowed_types = scraper_config.allowed_content_types
            current_type = response_info["content_type"]
            is_allowed = current_type in allowed_types or "text/html" in current_type
            if not is_allowed:
                content_type = response_info["content_type"]
                response_info["error"] = f"Unsupported content type: {content_type}"
                logger.warning(f"Unsupported content type for {normalized_url}: {content_type}")
                continue

            # Check file size
            content = response.text
            content_size = len(content.encode("utf-8"))
            response_info["size_bytes"] = content_size
            max_size = scraper_config.max_file_size_mb * 1024 * 1024

            if content_size > max_size:
                size_mb = content_size / (1024 * 1024)
                max_mb = scraper_config.max_file_size_mb
                response_info["error"] = f"Content too large: {size_mb:.2f}MB > {max_mb}MB"
                logger.warning(f"Content too large for {normalized_url}: {size_mb:.2f}MB")
                continue

            return content, response_info

        except requests.exceptions.Timeout as e:
            response_info["error"] = f"Request timed out after {scraper_config.timeout}s"
            logger.warning(f"Request timed out for {normalized_url}")
            timeout = scraper_config.timeout
            raise FetchTimeoutError(
                normalized_url, f"Request timed out after {timeout}s"
            ) from e
        except requests.exceptions.TooManyRedirects as e:
            response_info["error"] = "Too many redirects"
            logger.warning(f"Too many redirects for {normalized_url}")
            raise FetchError(normalized_url, "Too many redirects") from e
        except requests.exceptions.RequestException as e:
            response_info["error"] = f"Request error: {str(e)}"
            logger.warning(f"Request error for {normalized_url}: {e}")
            raise FetchError(normalized_url, str(e)) from e
        except Exception as e:
            response_info["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error fetching {normalized_url}: {e}")
            raise FetchError(normalized_url, str(e)) from e
        finally:
            # Always release rate limiter
            rate_limiter.release()

    # All retries failed
    error_message = response_info["error"]
    logger.error(f"Failed to fetch {normalized_url} after {retries} retries: {error_message}")
    return None, response_info


def scrape_url(db: Any, url: str) -> str:
    """
    Scrape single URL and store content.

    Args:
        db: WDBX database instance
        url: URL to scrape

    Returns:
        ID of scraped document

    Raises:
        WebScraperError: If any error occurs during scraping or processing.
    """
    normalized_url = _normalize_url(url)
    logger.info(f"Scraping URL: {normalized_url}")

    # Fetch content
    try:
        html, response_info = _fetch_url(normalized_url)
        if html is None:
            # _fetch_url should raise specific FetchError subtypes
            error_msg = response_info.get("error", "Unknown fetch error")
            logger.error(f"Failed to fetch {normalized_url}: {error_msg}")
            raise ScrapingError(f"Failed to fetch URL: {error_msg}")
    except FetchError as e:
        logger.error(f"Fetch error for {normalized_url}: {e}")
        raise ScrapingError(f"Failed to fetch URL: {e}") from e
    except Exception as e:  # Catch unexpected fetch errors
        logger.error(f"Unexpected fetch error for {normalized_url}: {e}", exc_info=True)
        raise ScrapingError(f"Unexpected error fetching URL: {e}") from e

    try:
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Extract metadata
        metadata = {
            # Include response info immediately
            "response_info": response_info
        }

        if scraper_config.extract_metadata:
            page_metadata = _extract_metadata(normalized_url, soup)
            metadata.update(page_metadata)

        # Extract links
        links = []
        if scraper_config.extract_links:
            links = _extract_links(normalized_url, soup)
            metadata["links"] = links
            logger.info(f"Extracted {len(links)} links from {normalized_url}")

        # Extract and clean text
        processor = ContentProcessor(scraper_config.chunk_size, scraper_config.chunk_overlap)
        raw_text = _extract_text_content(soup)
        text = processor.clean_text(raw_text)

        if not text:
            logger.warning(f"No text content extracted from {normalized_url}")
            metadata["warning"] = "No text content extracted"
            # We might still want to store metadata/links even if text is empty
            # Consider if this should be a hard error or just a warning.
            # For now, continue but log a warning.

        # Chunk text
        chunks = processor.create_chunks(text)
        logger.info(f"Created {len(chunks)} text chunks from {normalized_url}")

        # Save HTML content if enabled
        html_path = None
        if scraper_config.save_html:
            try:
                html_path = _save_html_content(normalized_url, html)
                if html_path:
                    logger.info(f"Saved HTML content to {html_path}")
                else:
                    logger.warning(f"Failed to save HTML for {normalized_url}")
            except Exception as e:
                logger.warning(f"Error saving HTML for {normalized_url}: {e}")

        # Prepare content object
        content = ScrapedContent(
            url=normalized_url,
            title=metadata.get("title"),
            description=metadata.get("description"),
            content=text,  # Store the cleaned text
            metadata=metadata,
            chunks=chunks,
            html_path=html_path,
            scrape_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )

        # Store data
        scrape_id = _store_scraped_data(db, content, response_info)
        logger.info(f"Stored scraped content with ID: {scrape_id}")

        return scrape_id

    except ParserRejectedMarkup as e:
        logger.error(f"HTML parsing error for {normalized_url}: {e}")
        raise ScrapingError(f"HTML parsing failed: {e}") from e
    except StorageError as e:  # Catch specific storage errors
        logger.error(f"Storage error for {normalized_url}: {e}")
        raise  # Re-raise storage errors
    except Exception as e:
        logger.error(f"Error processing {normalized_url}: {e}", exc_info=True)
        raise ScrapingError(f"Error processing content: {e}") from e


def scrape_site(db: Any, start_url: str, max_pages: Optional[int] = None) -> list[str]:
    """
    Crawl a site starting from URL and following links.

    Args:
        db: WDBX database instance
        start_url: URL to start crawling from
        max_pages: Maximum number of pages to scrape (default: from config)

    Returns:
        List of scraped document IDs

    Raises:
        WebScraperError: If errors occur during the crawl.
    """
    if max_pages is None:
        max_pages = scraper_config.max_pages_per_domain

    normalized_url = _normalize_url(start_url)
    base_domain = _get_domain(normalized_url)

    scraped_ids = []
    urls_to_scrape = [normalized_url]
    scraped_urls = set()
    visited_urls = set()  # Keep track of URLs attempted, even if failed
    crawl_errors = []  # Collect non-fatal errors during crawl

    logger.info(
        f"Starting site crawl from {normalized_url} (max {max_pages} pages, "
        f"domain: {base_domain})"
    )

    while urls_to_scrape and len(scraped_urls) < max_pages:
        # Get next URL to scrape
        url = urls_to_scrape.pop(0)

        # Skip if already visited or not in the same domain
        if url in visited_urls or _get_domain(url) != base_domain:
            continue

        visited_urls.add(url)

        try:
            # Scrape the URL
            scrape_id = scrape_url(db, url)

            # scrape_url now raises on error, so if we get here, it succeeded.
            scraped_ids.append(scrape_id)
            scraped_urls.add(url)

            # Get links from cache to add to queue
            if scrape_id in scraped_cache:
                cache_entry = scraped_cache.get(scrape_id, {})
                metadata = cache_entry.get("metadata", {})
                links = metadata.get("links", [])

                # Add new links from the same domain to the queue
                new_links_added = 0
                for link in links:
                    normalized_link = _normalize_url(link)
                    # Check domain and if not already visited/queued
                    if (
                        _get_domain(normalized_link) == base_domain
                        and normalized_link not in visited_urls
                        and normalized_link not in urls_to_scrape
                    ):
                        urls_to_scrape.append(normalized_link)
                        new_links_added += 1

                if new_links_added > 0:
                    logger.info(f"Added {new_links_added} new URLs to scrape queue from {url}")

            logger.info(
                f"Progress: {len(scraped_urls)}/{max_pages} pages scraped, "
                f"{len(urls_to_scrape)} in queue"
            )

        except WebScraperError as e:
            # Log the error but continue crawling other pages
            error_msg = f"Error scraping {url} during site crawl: {e}"
            logger.warning(error_msg)
            crawl_errors.append(error_msg)
            # Continue to the next URL in the queue
            continue
        except Exception as e:
            # Catch unexpected errors during scraping
            error_msg = f"Unexpected error scraping {url} during site crawl: {e}"
            logger.error(error_msg, exc_info=True)
            crawl_errors.append(error_msg)
            # Continue to the next URL
            continue

    logger.info(
        f"Completed site crawl for {base_domain}, successfully scraped "
        f"{len(scraped_urls)} pages."
    )
    if crawl_errors:
        logger.warning(f"Encountered {len(crawl_errors)} errors during crawl. First few errors:")
        for err in crawl_errors[:3]:
            logger.warning(f"  - {err}")
        # Consider raising an error or returning error info if needed
        # For now, just return the successfully scraped IDs

    return scraped_ids


def cmd_scrape_help(db: Any, args: str) -> None:
    """
    Display help information for the web scraper commands.

    Args:
        db: WDBX database instance (not used)
        args: Command arguments (not used)
    """
    print("\nWDBX Web Scraper Commands:")
    print("  scrape:url <url>     - Scrape a single URL and store the content")
    print("  scrape:site <url>    - Scrape multiple pages from a website")
    print("  scrape:list          - List recently scraped content")
    print("  scrape:status <id>   - Check the status of a scrape operation")
    print("  scrape:search <query> - Search scraped content")
    print("  scrape:config [key=value] - View or update scraper configuration")
    print("  scrape:help          - Show this help message")
    print("\nExamples:")
    print("  scrape:url https://example.com")
    print("  scrape:site https://example.com --limit 10")
    print('  scrape:config user_agent="My Custom User Agent"')

    print("\nConfiguration options:")
    print("  user_agent           - User agent string for requests")
    print("  timeout              - Request timeout in seconds")
    print("  max_retries          - Maximum number of retry attempts")
    print("  retry_delay          - Delay between retries in seconds")
    print("  follow_redirects     - Whether to follow redirects (true/false)")
    print("  respect_robots_txt   - Whether to respect robots.txt (true/false)")
    print("  extract_links        - Whether to extract links (true/false)")
    print("  extract_metadata     - Whether to extract metadata (true/false)")
    print("  chunk_size           - Size of text chunks for embeddings")
    print("  chunk_overlap        - Overlap between text chunks")
    print("  requests_per_minute  - Rate limit for requests per minute")
    print("  concurrent_requests  - Maximum concurrent requests")


def cmd_scrape_config(db: Any, args: str) -> None:
    """
    Configure web scraper settings.

    Args:
        db: WDBX database instance
        args: Configuration string in format key=value
    """
    print("\033[1;35mWDBX Web Scraper Configuration\033[0m")

    if not args:
        print("Current configuration:")
        _print_config()
        print("\nTo change a setting, use: scrape:config key=value")
        return

    # Parse key=value pairs
    updated_any = False
    parts = args.split()
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if _update_config_value(key, value):
                updated_any = True
        else:
            print(f"\033[1;31mInvalid format: {part}. Use key=value format.\033[0m")

    if updated_any:
        try:
            _save_scraper_config()

            # Ensure HTML directory exists if needed
            if scraper_config.save_html:
                _ensure_html_dir()

            # Update rate limiter instance if settings changed
            global rate_limiter
            rate_limiter = RateLimiter(
                scraper_config.requests_per_minute, scraper_config.concurrent_requests
            )
            logger.info("Updated rate limiter settings")

        except ConfigurationError as e:
            logger.error(f"Error saving configuration: {e}")
            print(f"\033[1;31mError saving configuration: {e}\033[0m")


def cmd_scrape_url(db: Any, args: str) -> None:
    """
    Scrape a single URL.

    Args:
        db: WDBX database instance
        args: URL to scrape
    """
    if not args:
        print("\033[1;31mError: URL required\033[0m")
        print("Usage: scrape:url <url>")
        return

    url = args.strip()
    print(f"\033[1;35mScraping URL: {url}\033[0m")

    try:
        # Normalize URL to ensure proper format
        normalized_url = _normalize_url(url)
        if normalized_url != url:
            print(f"Normalized URL: {normalized_url}")

        scrape_id = scrape_url(db, normalized_url)

        if scrape_id:
            print(f"\033[1;32mSuccessfully scraped URL: {normalized_url}\033[0m")
            print(f"Scrape ID: {scrape_id}")

            # Print basic stats
            if scrape_id in scraped_cache:
                cache_entry = scraped_cache[scrape_id]
                metadata = cache_entry.get("metadata", {})
                chunks = cache_entry.get("chunks", [])

                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"Description: {metadata.get('description', 'N/A')[:100]}...")
                print(f"Content chunks: {len(chunks)}")
                print(f"Content size: {sum(len(c) for c in chunks)} characters")

                if "links" in metadata:
                    print(f"Links: {len(metadata['links'])} extracted")
        else:
            print(f"\033[1;31mFailed to scrape URL: {normalized_url} (No ID returned)\033[0m")
            print("Check logs for more details. Use 'scrape:config' to adjust settings if needed")

    except FetchRobotsError as e:
        logger.error(f"Robots.txt disallowed: {e}")
        print(f"\033[1;31mCannot scrape URL: {e.url} - Blocked by robots.txt\033[0m")
        print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")

    except FetchRateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        print(f"\033[1;31mRate limit detected for {e.url}.\033[0m")
        print("Please try again later.")

    except FetchTimeoutError as e:
        logger.error(f"Timeout error: {e}")
        print(f"\033[1;31mTimeout error fetching {e.url}\033[0m")
        print("You can increase timeout with: scrape:config timeout=<higher_value>")

    except FetchError as e:
        logger.error(f"Fetch error: {e}")
        print(f"\033[1;31mError fetching {e.url}: {e.message}\033[0m")
        if hasattr(e, "status_code") and e.status_code:
            print(f"HTTP Status Code: {e.status_code}")

    except ScrapingError as e:
        logger.error(f"Scraping error: {e}")
        print(f"\033[1;31mError during scraping: {e}\033[0m")

    except StorageError as e:
        logger.error(f"Storage error: {e}")
        print(f"\033[1;31mError storing scraped content: {e}\033[0m")

    except WebScraperError as e:
        logger.error(f"Web scraper error: {e}")
        print(f"\033[1;31mWeb scraper error: {e}\033[0m")

    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}", exc_info=True)
        print(f"\033[1;31mAn unexpected error occurred: {e}\033[0m")
        print("Please check logs for more details.")


def cmd_scrape_site(db: Any, args: str) -> None:
    """
    Scrape a website starting from the given URL.

    Args:
        db: WDBX database instance
        args: Start URL and optional max_pages
    """
    if not args:
        print("\033[1;31mError: URL required\033[0m")
        print("Usage: scrape:site <url> [max_pages]")
        return

    parts = args.strip().split()
    url = parts[0]

    max_pages = scraper_config.max_pages_per_domain
    if len(parts) > 1:
        try:
            max_pages_arg = int(parts[1])
            if max_pages_arg > 0:
                max_pages = max_pages_arg
            else:
                print(
                    f"\033[1;33mWarning: max_pages must be positive,"
                    f" using default ({max_pages})\033[0m"
                )
        except ValueError:
            print(
                f"\033[1;31mError: max_pages must be an integer,"
                f" using default ({max_pages})\033[0m"
            )

    # Normalize URL to ensure proper format
    normalized_url = _normalize_url(url)
    if normalized_url != url:
        print(f"Normalized URL: {normalized_url}")

    print(f"\033[1;35mCrawling site: {normalized_url} (max {max_pages} pages)\033[0m")

    try:
        domain = _get_domain(normalized_url)
        print(f"Domain: {domain}")

        scraped_ids = scrape_site(db, normalized_url, max_pages)

        if scraped_ids:
            print(f"\033[1;32mSuccessfully crawled site: {normalized_url}\033[0m")
            print(f"Scraped {len(scraped_ids)} pages")

            # Print a summary of scraped pages
            if len(scraped_ids) > 0:
                print("\nScraped pages summary:")
                for i, scrape_id in enumerate(scraped_ids[:5]):  # Show first 5
                    if scrape_id in scraped_cache:
                        item = scraped_cache[scrape_id]
                        print(f"{i + 1}. {item.get('url', 'N/A')}")
                        print(f"   ({len(item.get('chunks', []))}) chunks")

                if len(scraped_ids) > 5:
                    print(f"...and {len(scraped_ids) - 5} more pages")

                print("\nUse 'scrape:list' to see all scraped URLs")
        else:
            print(f"\033[1;31mFailed to crawl site or scraped 0 pages: {normalized_url}\033[0m")
            print("Check logs for errors. Use 'scrape:config' to adjust settings if needed")

    except FetchRobotsError as e:
        logger.error(f"Robots.txt disallowed: {e}")
        print(f"\033[1;31mCannot crawl site: {e.url} - Blocked by robots.txt\033[0m")
        print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")

    except FetchRateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        print(f"\033[1;31mRate limit detected for {e.url}.\033[0m")
        print("Please try again later.")

    except FetchTimeoutError as e:
        logger.error(f"Timeout error: {e}")
        print(f"\033[1;31mTimeout error fetching {e.url}\033[0m")
        print("You can increase timeout with: scrape:config timeout=<higher_value>")

    except FetchError as e:
        logger.error(f"Fetch error: {e}")
        print(f"\033[1;31mError fetching {e.url}: {e.message}\033[0m")

    except ScrapingError as e:
        logger.error(f"Scraping error: {e}")
        print(f"\033[1;31mError during scraping: {e}\033[0m")

    except StorageError as e:
        logger.error(f"Storage error: {e}")
        print(f"\033[1;31mError storing scraped content: {e}\033[0m")

    except WebScraperError as e:
        logger.error(f"Web scraper error: {e}")
        print(f"\033[1;31mWeb scraper error: {e}\033[0m")

    except Exception as e:
        logger.error(f"Unexpected error crawling {normalized_url}: {e}", exc_info=True)
        print(f"\033[1;31mAn unexpected error occurred: {e}\033[0m")
        print("Please check logs for more details.")


def cmd_scrape_list(db: Any, args: str) -> None:
    """
    List scraped URLs.

    Args:
        db: WDBX database instance
        args: Optional domain filter
    """
    domain_filter = args.strip() if args else None

    print("\033[1;35mScraped URLs:\033[0m")

    # Get all entries from cache
    scraped_items = []
    for scrape_id, item in scraped_cache.items():
        url = item.get("url", "")
        metadata = item.get("metadata", {})

        if domain_filter and domain_filter not in url:
            continue

        scraped_items.append(
            {
                "id": scrape_id,
                "url": url,
                "title": metadata.get("title", "N/A"),
                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(item.get("timestamp", 0))
                ),
                "chunks": len(item.get("chunks", [])),
            }
        )

    # Sort by timestamp (newest first)
    scraped_items.sort(key=lambda x: x["time"], reverse=True)

    if not scraped_items:
        print("No scraped URLs found")
        return

    # Print as table
    if domain_filter:
        msg = f"Found {len(scraped_items)} scraped URLs for domain: {domain_filter}"
    else:
        msg = f"Found {len(scraped_items)} scraped URLs"
    print(msg)

    # Define table format
    fmt = "{:<8} {:<40} {:<30} {:<20} {:<8}"
    print(fmt.format("ID", "URL", "Title", "Scraped At", "Chunks"))
    print("-" * 110)

    for item in scraped_items:
        # Truncate long fields
        short_id = item["id"][:8]
        short_url = (
            item["url"][:MAX_DISPLAY_URL_LENGTH]
            + ("..." if len(item["url"]) > MAX_DISPLAY_URL_LENGTH else "")
        )
        short_title = (
            item["title"][:MAX_DISPLAY_TITLE_LENGTH]
            + ("..." if len(item["title"]) > MAX_DISPLAY_TITLE_LENGTH else "")
        )

        print(fmt.format(short_id, short_url, short_title, item["time"], item["chunks"]))


def cmd_scrape_status(db: Any, args: str) -> None:
    """
    Show detailed status of a scraped document.

    Args:
        db: WDBX database instance
        args: Scrape ID
    """
    if not args:
        print("\033[1;31mError: Scrape ID required\033[0m")
        print("Usage: scrape:status <id>")
        return

    scrape_id = args.strip()

    # Try partial match if exact ID not found
    if scrape_id not in scraped_cache:
        matches = [id for id in scraped_cache if id.startswith(scrape_id)]
        if len(matches) == 1:
            scrape_id = matches[0]
        elif len(matches) > 1:
            print(f"\033[1;31mMultiple matches found for ID: {scrape_id}\033[0m")
            for match in matches:
                print(f"  {match}")
            return

    if scrape_id not in scraped_cache:
        print(f"\033[1;31mNo scrape found with ID: {scrape_id}\033[0m")
        return

    item = scraped_cache[scrape_id]
    metadata = item.get("metadata", {})
    chunks = item.get("chunks", [])

    print(f"\033[1;35mScrape Status for ID: {scrape_id}\033[0m")
    print(f"\033[1mURL:\033[0m {item.get('url', 'N/A')}")
    print(f"\033[1mTitle:\033[0m {metadata.get('title', 'N/A')}")
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("timestamp", 0)))
    print(f"\033[1mScraped at:\033[0m {time_str}")
    print(f"\033[1mChunks:\033[0m {len(chunks)}")

    # Print description if available
    if "description" in metadata:
        print(f"\033[1mDescription:\033[0m {metadata['description']}")

    # Print response info
    if "response_info" in metadata:
        resp = metadata["response_info"]
        print("\n\033[1mResponse Info:\033[0m")
        print(f"  Status: {resp.get('status_code', 'N/A')}")
        print(f"  Content Type: {resp.get('content_type', 'N/A')}")
        print(f"  Size: {resp.get('size_bytes', 0) / 1024:.1f} KB")
        print(f"  Fetch Time: {resp.get('fetch_time_sec', 0):.2f} seconds")

    # Print extracted metadata
    print("\n\033[1mMetadata:\033[0m")
    for key, value in metadata.items():
        if key not in ["response_info", "links", "scrape_id", "chunk_count"]:
            if isinstance(value, str) and len(value) > MAX_DISPLAY_CHUNK_PREVIEW:
                value = value[:MAX_DISPLAY_CHUNK_PREVIEW] + "..."
            print(f"  {key}: {value}")

    # Print chunk previews
    print("\n\033[1mContent Chunks:\033[0m")
    for i, chunk in enumerate(chunks[:MAX_CHUNKS_BEFORE_MORE_NOTICE]):
        print(f"  Chunk {i + 1}/{len(chunks)}: {chunk[:MAX_DISPLAY_CHUNK_PREVIEW]}...")

    if len(chunks) > MAX_CHUNKS_BEFORE_MORE_NOTICE:
        print(f"  ... and {len(chunks) - MAX_CHUNKS_BEFORE_MORE_NOTICE} more chunks")
    print("-" * 40)

    # Print HTML path if available
    if "html_path" in metadata:
        print(f"\n\033[1mHTML File:\033[0m {metadata['html_path']}")

    # Print link stats if available
    if "links" in metadata:
        links = metadata["links"]
        print(f"\n\033[1mLinks:\033[0m {len(links)} extracted")

        # Group links by domain
        domains = {}
        for link in links:
            domain = _get_domain(link)
            domains[domain] = domains.get(domain, 0) + 1

        # Print top domains
        print("  Top domains:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5]:

            print(f"    {domain}: {count} links")


def cmd_scrape_search(db: Any, args: str) -> None:
    """
    Search scraped content.

    Args:
        db: WDBX database instance
        args: Search query
    """
    if not args:
        print("\033[1;31mError: Search query required\033[0m")
        print("Usage: scrape:search <query>")
        return

    query = args.strip()
    print(f"\033[1;35mSearching scraped content for: {query}\033[0m")

    # Simple text search in cache first
    results = []
    query_lower = query.lower()

    for scrape_id, item in scraped_cache.items():
        url = item.get("url", "")
        metadata = item.get("metadata", {})
        chunks = item.get("chunks", [])

        # Search in title and description
        title = metadata.get("title", "").lower()
        description = metadata.get("description", "").lower()

        title_match = query_lower in title
        desc_match = query_lower in description

        # Search in chunks
        matched_chunks = []
        for i, chunk in enumerate(chunks):
            if query_lower in chunk.lower():
                # Get context around the match
                index = chunk.lower().find(query_lower)
                start = max(0, index - 50)
                end = min(len(chunk), index + len(query) + 50)
                context = chunk[start:end]

                # Replace the query with highlighted version
                highlight = context.replace(
                    query, f"\033[1;33m{query}\033[0m", 1  # Only replace first occurrence
                )

                matched_chunks.append({"index": i, "highlight": highlight})

        if title_match or desc_match or matched_chunks:
            results.append(
                {
                    "id": scrape_id,
                    "url": url,
                    "title": metadata.get("title", "N/A"),
                    "title_match": title_match,
                    "desc_match": desc_match,
                    "chunks": matched_chunks,
                }
            )

    # Sort results: title matches first, then description matches, then by number of chunk matches
    results.sort(key=lambda x: (not x["title_match"], not x["desc_match"], -len(x["chunks"])))

    if not results:
        print("No matches found in scraped content")

        # Suggest using vector search if available
        print("\nFor semantic search, use the database search capabilities or model:embed")
        return

    # Print results
    print(f"Found {len(results)} matches:")

    for i, result in enumerate(results[:MAX_DISPLAY_RESULTS_PREVIEW]):  # Show top 10 results
        print(f"\n\033[1m{i + 1}. {result['title']}\033[0m")
        print(f"   URL: {result['url']}")
        print(f"   ID: {result['id']}")

        # Print chunk matches
        for j, chunk in enumerate(result["chunks"][:MAX_CHUNKS_BEFORE_MORE_NOTICE]):
            print(f"   Match {j + 1}: ...{chunk['highlight']}...")

        if len(result["chunks"]) > MAX_CHUNKS_BEFORE_MORE_NOTICE:
            print(f"   ... and {len(result['chunks']) - MAX_CHUNKS_BEFORE_MORE_NOTICE} more matches")

    if len(results) > MAX_RESULTS_BEFORE_MORE_NOTICE:
        print(f"\n... and {len(results) - MAX_RESULTS_BEFORE_MORE_NOTICE} more results")

    print("\nTo see details for a specific result, use: scrape:status <id>")
