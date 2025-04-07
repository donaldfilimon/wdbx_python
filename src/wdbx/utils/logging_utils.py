"""
Logging utilities for WDBX.

This module provides a unified logging interface with support for structured logging,
colored output, and integration with the configuration system.
"""

import atexit
import json
import logging
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

# Avoid circular imports - we'll load config lazily
# from ..config.config_manager import get_config


class LogLevel(Enum):
    """Log levels for WDBX logging."""
    
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """
        Convert a string to a LogLevel.
        
        Args:
            level_str: String representation of log level
            
        Returns:
            LogLevel enum value
        """
        try:
            return cls[level_str.upper()]
        except KeyError:
            # Default to INFO for unknown levels
            return cls.INFO


class LogFormat(Enum):
    """Log format options for WDBX logging."""
    
    TEXT = "text"
    JSON = "json"
    COLOR = "color"


@dataclass
class LogContext:
    """
    Context information for a log entry.
    
    Attributes:
        request_id: Identifier for the current request
        user_id: Identifier for the current user
        session_id: Identifier for the current session
        component: Component that generated the log
        tags: List of tags for categorizing logs
        extra: Additional context specific to the log entry
    """
    
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


# Thread-local storage for log context
_log_context = threading.local()


def get_log_context() -> LogContext:
    """
    Get the current log context.
        
    Returns:
        Current log context for this thread
    """
    if not hasattr(_log_context, "context"):
        _log_context.context = LogContext()
    return _log_context.context


def set_log_context(context: LogContext) -> None:
    """
    Set the current log context.
    
    Args:
        context: Log context to set
    """
    _log_context.context = context


def update_log_context(**kwargs) -> None:
    """
    Update the current log context with new values.
    
    Args:
        **kwargs: Key-value pairs to update in the context
    """
    context = get_log_context()
    for key, value in kwargs.items():
        if hasattr(context, key):
            setattr(context, key, value)
        else:
            context.extra[key] = value


def clear_log_context() -> None:
    """Clear the current log context."""
    if hasattr(_log_context, "context"):
        delattr(_log_context, "context")


@contextmanager
def log_context(**kwargs) -> None:
    """
    Context manager for temporarily setting log context.
    
    Args:
        **kwargs: Key-value pairs to update in the context
    """
    old_context = get_log_context()
    new_context = LogContext(
        request_id=old_context.request_id,
        user_id=old_context.user_id,
        session_id=old_context.session_id,
        component=old_context.component,
        tags=old_context.tags.copy(),
        extra=old_context.extra.copy()
    )
    
    for key, value in kwargs.items():
        if hasattr(new_context, key):
            setattr(new_context, key, value)
        else:
            new_context.extra[key] = value
    
    set_log_context(new_context)
    try:
        yield
    finally:
        set_log_context(old_context)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings for logging."""
    
    def __init__(self, include_context: bool = True):
        """
        Initialize a JSONFormatter.
        
        Args:
            include_context: Whether to include context in log entries
        """
        super().__init__()
        self.include_context = include_context
    
    def _serialize(self, obj: Any) -> Any:
        """
        Serialize an object for JSON logging.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Exception):
            return {
                "type": obj.__class__.__name__,
                "message": str(obj),
                "traceback": traceback.format_exc()
            }
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return self._serialize(obj.__dict__)
        return obj
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON representation of the log record
        """
        # Basic log record info
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
            "message": record.getMessage(),
            "source": {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }
        }
        
        # Add context if available
        if self.include_context:
            context = get_log_context()
            log_entry["context"] = self._serialize(context)
        
        # Add extra attributes from record
        if hasattr(record, "exc_info") and record.exc_info:
            exception = record.exc_info[1]
            log_entry["exception"] = self._serialize(exception)
        
        # Add any extra attributes from record.__dict__
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName"
            }:
                log_entry[key] = self._serialize(value)
        
        return json.dumps(log_entry)


class ColorFormatter(logging.Formatter):
    """
    Formatter that adds color to console log output.
    
    Example output:
    [2023-08-01 12:34:56] [INFO] [MainThread] [app.module]: Log message
    """
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m\033[37m",  # White on Red
        "RESET": "\033[0m",  # Reset
    }
    
    def __init__(
        self,
        include_context: bool = True,
        include_thread: bool = True,
        include_source: bool = False,
    ):
        """
        Initialize a ColorFormatter.
        
        Args:
            include_context: Whether to include context in log entries
            include_thread: Whether to include thread information
            include_source: Whether to include source file and line
        """
        self.include_context = include_context
        self.include_thread = include_thread
        self.include_source = include_source
        
        fmt = "[%(asctime)s] [%(levelname)s]"
        if include_thread:
            fmt += " [%(threadName)s]"
        fmt += " [%(name)s]: %(message)s"
        
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with color.
        
        Args:
            record: Log record to format
            
        Returns:
            Colored string representation of the log record
        """
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add color
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        # Color the levelname
        formatted = formatted.replace(f"[{levelname}]", f"[{color}{levelname}{reset}]")
        
        # Add context if available
        if self.include_context:
            context = get_log_context()
            context_parts = []
            
            if context.request_id:
                context_parts.append(f"request_id={context.request_id}")
            if context.user_id:
                context_parts.append(f"user_id={context.user_id}")
            if context.session_id:
                context_parts.append(f"session_id={context.session_id}")
            if context.tags:
                context_parts.append(f"tags={','.join(context.tags)}")
            
            if context_parts:
                formatted += f" ({' '.join(context_parts)})"
        
        # Add source information if requested
        if self.include_source:
            source = f"{record.pathname}:{record.lineno}"
            formatted += f" [{source}]"
        
        return formatted


class FileHandler:
    """Utility for configuring file logging."""
    
    @staticmethod
    def get_log_file_path(log_dir: Union[str, Path], prefix: str = "wdbx") -> Path:
        """
        Get the path for a log file.
        
        Args:
            log_dir: Directory for log files
            prefix: Prefix for log file name
            
        Returns:
            Path to log file
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        return log_dir / f"{prefix}_{date_str}.log"
    
    @staticmethod
    def create_rotating_handler(
        log_file: Union[str, Path],
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        formatter: Optional[logging.Formatter] = None
    ) -> logging.Handler:
        """
        Create a rotating file handler.
        
        Args:
            log_file: Path to log file
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            formatter: Formatter for log entries
            
        Returns:
            Configured RotatingFileHandler
        """
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        
        if formatter:
            handler.setFormatter(formatter)
        
        return handler
    
    @staticmethod
    def create_timed_rotating_handler(
        log_file: Union[str, Path],
        when: str = "midnight",
        interval: int = 1,
        backup_count: int = 7,
        formatter: Optional[logging.Formatter] = None
    ) -> logging.Handler:
        """
        Create a time-based rotating file handler.
        
        Args:
            log_file: Path to log file
            when: Time unit for rotation
            interval: Interval in the specified time unit
            backup_count: Number of backup files to keep
            formatter: Formatter for log entries
            
        Returns:
            Configured TimedRotatingFileHandler
        """
        handler = TimedRotatingFileHandler(
            log_file, when=when, interval=interval, backupCount=backup_count
        )
        
        if formatter:
            handler.setFormatter(formatter)
        
        return handler


class LogManager:
    """
    Manager for WDBX logging configuration.
    
    This class provides a unified interface for configuring loggers,
    handlers, formatters, and log destinations.
    """
    
    def __init__(
        self,
        default_level: Union[LogLevel, str, int] = LogLevel.INFO,
        log_format: LogFormat = LogFormat.COLOR,
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_dir: Optional[Union[str, Path]] = None,
        configure_root: bool = True
    ):
        """
        Initialize a LogManager.
        
        Args:
            default_level: Default log level
            log_format: Format for log entries
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_dir: Directory for log files
            configure_root: Whether to configure the root logger
        """
        # Convert string or int level to enum
        if isinstance(default_level, str):
            self.default_level = LogLevel.from_string(default_level)
        elif isinstance(default_level, int):
            # Find the closest log level
            levels = sorted([(level.value, level) for level in LogLevel], key=lambda x: x[0])
            closest = min(levels, key=lambda x: abs(x[0] - default_level))
            self.default_level = closest[1]
        else:
            self.default_level = default_level
        
        self.log_format = log_format
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        
        # Get log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            # Try to get from environment or use a default
            log_dir_env = os.environ.get("WDBX_LOG_DIR")
            if log_dir_env:
                self.log_dir = Path(log_dir_env)
            else:
                self.log_dir = Path("./logs")
        
        # Create formatters
        self.console_formatter = self._create_formatter(self.log_format)
        self.file_formatter = self._create_formatter(LogFormat.JSON)
        
        # Dictionary to track configured loggers
        self.configured_loggers: Set[str] = set()
        
        # Configure root logger if requested
        if configure_root:
            self.configure_logger(logging.getLogger())
    
    def _create_formatter(self, log_format: LogFormat) -> logging.Formatter:
        """
        Create a formatter based on the specified format.
        
        Args:
            log_format: Format for log entries
            
        Returns:
            Configured formatter
        """
        if log_format == LogFormat.JSON:
            return JSONFormatter()
        elif log_format == LogFormat.COLOR:
            return ColorFormatter()
        else:  # TEXT
            return logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
    
    def configure_logger(
        self,
        logger: Union[str, logging.Logger],
        level: Optional[Union[LogLevel, str, int]] = None,
        propagate: bool = False
    ) -> logging.Logger:
        """
        Configure a logger with the specified settings.
        
        Args:
            logger: Logger object or name
            level: Log level (defaults to manager's default_level)
            propagate: Whether to propagate logs to parent loggers
            
        Returns:
            Configured logger
        """
        # Get logger object if name was provided
        if isinstance(logger, str):
            logger_obj = logging.getLogger(logger)
        else:
            logger_obj = logger
        
        # Skip if already configured
        if logger_obj.name in self.configured_loggers:
            return logger_obj
        
        # Remove existing handlers
        for handler in list(logger_obj.handlers):
            logger_obj.removeHandler(handler)
        
        # Set propagation
        logger_obj.propagate = propagate
        
        # Set level
        if level is None:
            logger_obj.setLevel(self.default_level.value)
        elif isinstance(level, str):
            logger_obj.setLevel(LogLevel.from_string(level).value)
        elif isinstance(level, int):
            logger_obj.setLevel(level)
        else:
            logger_obj.setLevel(level.value)
        
        # Add console handler if enabled
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.console_formatter)
            logger_obj.addHandler(console_handler)
        
        # Add file handler if enabled
        if self.log_to_file:
            log_file = FileHandler.get_log_file_path(self.log_dir)
            file_handler = FileHandler.create_rotating_handler(
                log_file, formatter=self.file_formatter
            )
            logger_obj.addHandler(file_handler)
        
        # Mark as configured
        self.configured_loggers.add(logger_obj.name)
        
        return logger_obj
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger
        """
        return self.configure_logger(name)
    
    def set_level(self, level: Union[LogLevel, str, int], logger_name: Optional[str] = None) -> None:
        """
        Set the log level for a logger.
        
        Args:
            level: Log level
            logger_name: Logger name (None for all configured loggers)
        """
        # Convert level to int
        if isinstance(level, str):
            level_value = LogLevel.from_string(level).value
        elif isinstance(level, LogLevel):
            level_value = level.value
        else:
            level_value = level
        
        if logger_name:
            # Set level for specific logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(level_value)
        else:
            # Set level for all configured loggers
            for name in self.configured_loggers:
                logging.getLogger(name).setLevel(level_value)
    
    def shutdown(self) -> None:
        """Shut down logging and ensure all messages are flushed."""
        logging.shutdown()


# Singleton instance for reuse
_LOG_MANAGER_INSTANCE: Optional[LogManager] = None


def _init_from_config() -> LogManager:
    """
    Initialize LogManager from configuration.
    
    Returns:
        Configured LogManager
    """
    # Import here to avoid circular import
    from ..config.config_manager import get_config
    
    log_level = get_config("log_level", "INFO")
    log_format = get_config("log_format", "color")
    log_to_console = get_config("log_to_console", True)
    log_to_file = get_config("log_to_file", False)
    log_dir = get_config("log_dir", "./logs")
    
    return LogManager(
        default_level=log_level,
        log_format=LogFormat(log_format.lower()) if isinstance(log_format, str) else LogFormat.COLOR,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        log_dir=log_dir
    )


def get_log_manager() -> LogManager:
    """
    Get the log manager instance, initializing it if necessary.
    
    Returns:
        LogManager instance
    """
    global _LOG_MANAGER_INSTANCE
    
    if _LOG_MANAGER_INSTANCE is None:
        try:
            # Try to initialize from config
            _LOG_MANAGER_INSTANCE = _init_from_config()
        except ImportError:
            # Fall back to default if config not available
            _LOG_MANAGER_INSTANCE = LogManager()
    
    return _LOG_MANAGER_INSTANCE


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return get_log_manager().get_logger(name)


def set_log_level(level: Union[LogLevel, str, int], logger_name: Optional[str] = None) -> None:
    """
    Set the log level for a logger.
    
    Args:
        level: Log level
        logger_name: Logger name (None for all configured loggers)
    """
    get_log_manager().set_level(level, logger_name)


@contextmanager
def log_duration(
    logger: Union[str, logging.Logger],
    operation: str,
    level: Union[LogLevel, str, int] = LogLevel.INFO
) -> None:
    """
    Context manager to log the duration of an operation.
    
    Args:
        logger: Logger object or name
        operation: Name of the operation
        level: Log level for the duration message
    """
    # Get logger object if name was provided
    if isinstance(logger, str):
        logger_obj = get_logger(logger)
    else:
        logger_obj = logger
    
    # Convert level to int
    if isinstance(level, str):
        level_value = LogLevel.from_string(level).value
    elif isinstance(level, LogLevel):
        level_value = level.value
    else:
        level_value = level
    
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger_obj.log(level_value, f"{operation} completed in {duration:.3f} seconds")


# Register shutdown handler to ensure logs are flushed
def _shutdown_logging():
    """Ensure all logs are flushed on exit."""
    if _LOG_MANAGER_INSTANCE is not None:
        _LOG_MANAGER_INSTANCE.shutdown()


atexit.register(_shutdown_logging) 