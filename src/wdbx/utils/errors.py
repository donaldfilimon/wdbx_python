"""
Error handling utilities for WDBX.

This module provides a standardized approach to error handling in WDBX,
including custom exceptions, error codes, and helper functions.
"""

import enum
import inspect
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from .logging_utils import get_logger

# Initialize logger
logger = get_logger("wdbx.errors")


class ErrorCode(enum.Enum):
    """Error codes for WDBX exceptions."""
    
    # General errors (1-99)
    UNKNOWN_ERROR = 1
    NOT_IMPLEMENTED = 2
    INVALID_ARGUMENT = 3
    OPERATION_FAILED = 4
    TIMEOUT = 5
    
    # Configuration errors (100-199)
    CONFIG_ERROR = 100
    INVALID_CONFIG = 101
    CONFIG_NOT_FOUND = 102
    
    # Data errors (200-299)
    DATA_ERROR = 200
    INVALID_DATA = 201
    DATA_NOT_FOUND = 202
    DUPLICATE_DATA = 203
    DATA_CORRUPTION = 204
    
    # Vector errors (300-399)
    VECTOR_ERROR = 300
    VECTOR_NOT_FOUND = 301
    VECTOR_DIMENSION_MISMATCH = 302
    INVALID_VECTOR = 303
    
    # Block errors (400-499)
    BLOCK_ERROR = 400
    BLOCK_NOT_FOUND = 401
    INVALID_BLOCK = 402
    
    # ML backend errors (500-599)
    ML_BACKEND_ERROR = 500
    BACKEND_NOT_AVAILABLE = 501
    BACKEND_INITIALIZATION_FAILED = 502
    
    # Memory errors (600-699)
    MEMORY_ERROR = 600
    OUT_OF_MEMORY = 601
    MEMORY_OPTIMIZATION_FAILED = 602
    
    # I/O errors (700-799)
    IO_ERROR = 700
    FILE_NOT_FOUND = 701
    PERMISSION_DENIED = 702
    STORAGE_ERROR = 703
    
    # API errors (800-899)
    API_ERROR = 800
    INVALID_REQUEST = 801
    AUTHENTICATION_ERROR = 802
    AUTHORIZATION_ERROR = 803
    RATE_LIMIT_EXCEEDED = 804
    
    # Server errors (900-999)
    SERVER_ERROR = 900
    INTERNAL_SERVER_ERROR = 901
    SERVICE_UNAVAILABLE = 902


class WDBXError(Exception):
    """Base exception class for WDBX errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize a WDBX error.
        
        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
            cause: Original exception that caused this error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        
        # Construct the error message
        full_message = f"[{error_code.name} ({error_code.value})] {message}"
        if details:
            full_message += f"\nDetails: {details}"
        if cause:
            full_message += f"\nCaused by: {type(cause).__name__}: {str(cause)}"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        result = {
            "error": self.error_code.name,
            "code": self.error_code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        error_code: Optional[ErrorCode] = None,
        message: Optional[str] = None,
    ) -> "WDBXError":
        """
        Create a WDBX error from another exception.
        
        Args:
            exception: Original exception
            error_code: Error code (defaults to UNKNOWN_ERROR)
            message: Custom error message (defaults to the original exception message)
            
        Returns:
            WDBX error
        """
        if isinstance(exception, WDBXError):
            return exception
        
        error_message = message or str(exception)
        code = error_code or ErrorCode.UNKNOWN_ERROR
        
        return cls(error_message, code, cause=exception)


# Specific exception classes

class ConfigError(WDBXError):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CONFIG_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class DataError(WDBXError):
    """Data-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATA_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class VectorError(WDBXError):
    """Vector-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.VECTOR_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class BlockError(WDBXError):
    """Block-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.BLOCK_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class MLBackendError(WDBXError):
    """ML backend-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.ML_BACKEND_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class MemoryError(WDBXError):
    """Memory-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.MEMORY_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class IOError(WDBXError):
    """I/O-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.IO_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class APIError(WDBXError):
    """API-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.API_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class ServerError(WDBXError):
    """Server-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


# Error handling utilities

def error_context(
    error_code: ErrorCode,
    message: str = "Operation failed",
    log_level: str = "error",
    include_stack_trace: bool = True,
) -> Callable:
    """
    Decorator for handling errors in functions.
    
    Args:
        error_code: Error code to use for exceptions
        message: Base error message
        log_level: Logging level to use
        include_stack_trace: Whether to include stack trace in logs
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except WDBXError as e:
                # Pass through existing WDBX errors
                log_method = getattr(logger, log_level)
                log_method(
                    f"Error in {func.__name__}: {str(e)}", 
                    exc_info=include_stack_trace
                )
                raise
            except Exception as e:
                # Convert other exceptions to WDBX errors
                func_args = inspect.signature(func).bind(*args, **kwargs)
                func_args.apply_defaults()
                details = {
                    "function": func.__name__,
                    "args": str(func_args.arguments),
                }
                
                # Build appropriate error message
                error_msg = f"{message} in {func.__name__}: {str(e)}"
                
                # Create WDBX error
                wdbx_error = WDBXError(
                    error_msg,
                    error_code=error_code,
                    details=details,
                    cause=e,
                )
                
                # Log the error
                log_method = getattr(logger, log_level)
                log_method(str(wdbx_error), exc_info=include_stack_trace)
                
                # Raise the WDBX error
                raise wdbx_error
        
        # Preserve the original function's metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    return decorator


def handle_error(
    exception: Exception,
    error_code: Optional[ErrorCode] = None,
    message: Optional[str] = None,
    log_level: str = "error",
) -> WDBXError:
    """
    Handle an exception by converting it to a WDBX error.
    
    Args:
        exception: Original exception
        error_code: Error code to use (defaults to UNKNOWN_ERROR)
        message: Custom error message (defaults to the original exception message)
        log_level: Logging level to use
        
    Returns:
        WDBX error
    """
    wdbx_error = WDBXError.from_exception(exception, error_code, message)
    
    # Log the error
    log_method = getattr(logger, log_level)
    log_method(str(wdbx_error), exc_info=True)
    
    return wdbx_error


def get_error_info(exc_info: Optional[Any] = None) -> Dict[str, Any]:
    """
    Get information about an exception.
    
    Args:
        exc_info: Exception info from sys.exc_info() (defaults to current exception)
        
    Returns:
        Dictionary with exception information
    """
    if exc_info is None:
        exc_info = sys.exc_info()
    
    exc_type, exc_value, exc_tb = exc_info
    
    # Get traceback frames
    tb_frames = []
    if exc_tb:
        for frame, lineno in traceback.walk_tb(exc_tb):
            co = frame.f_code
            tb_frames.append({
                "filename": co.co_filename,
                "name": co.co_name,
                "lineno": lineno,
            })
    
    # Build error info dictionary
    error_info = {
        "type": exc_type.__name__ if exc_type else None,
        "message": str(exc_value) if exc_value else None,
        "traceback": tb_frames,
    }
    
    # Add additional details for WDBX errors
    if isinstance(exc_value, WDBXError):
        error_info.update({
            "code": exc_value.error_code.value,
            "name": exc_value.error_code.name,
            "details": exc_value.details,
        })
    
    return error_info


def is_retriable_error(error: Union[Exception, ErrorCode]) -> bool:
    """
    Check if an error is retriable.
    
    Args:
        error: Exception or error code
        
    Returns:
        True if the error is retriable, False otherwise
    """
    # Convert exception to error code if needed
    if isinstance(error, Exception):
        if isinstance(error, WDBXError):
            code = error.error_code
        else:
            # Non-WDBX exceptions are generally not retriable
            return False
    else:
        code = error
    
    # List of retriable error codes
    retriable_codes = [
        ErrorCode.TIMEOUT,
        ErrorCode.MEMORY_ERROR,
        ErrorCode.OUT_OF_MEMORY,
        ErrorCode.IO_ERROR,
        ErrorCode.STORAGE_ERROR,
        ErrorCode.SERVER_ERROR,
        ErrorCode.SERVICE_UNAVAILABLE,
    ]
    
    return code in retriable_codes 