"""
Rate limiting utilities for WDBX.

This module provides rate limiting functionality to protect
the API from excessive usage and abuse.
"""

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, cast

from .logging_utils import get_logger

# Initialize logger
logger = get_logger("wdbx.rate_limit")

# Type variable for generic function
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class RateLimitBucket:
    """Rate limiting bucket with token bucket algorithm."""
    
    max_tokens: int
    tokens: int
    refill_rate: float  # Tokens per second
    last_refill: float  # Timestamp of last refill
    blocked_until: float = 0.0  # Timestamp when blocking expires
    
    def __post_init__(self) -> None:
        """Initialize bucket with current time."""
        self.last_refill = time.time()
    
    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        
        # Skip if blocked
        if now < self.blocked_until:
            return
        
        # Calculate elapsed time since last refill
        elapsed = now - self.last_refill
        
        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate
        
        # Add tokens
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        
        # Update last refill time
        self.last_refill = now
    
    def consume(self, amount: int = 1) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            amount: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        # Refill tokens first
        self.refill()
        
        # Check if we have enough tokens
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        
        return False
    
    def block(self, duration: float) -> None:
        """
        Block bucket for a specified duration.
        
        Args:
            duration: Blocking duration in seconds
        """
        self.blocked_until = time.time() + duration
    
    def is_blocked(self) -> bool:
        """
        Check if bucket is blocked.
        
        Returns:
            True if bucket is blocked, False otherwise
        """
        return time.time() < self.blocked_until
    
    def get_wait_time(self, amount: int = 1) -> float:
        """
        Get time to wait for enough tokens.
        
        Args:
            amount: Number of tokens needed
            
        Returns:
            Time to wait in seconds
        """
        # Check if blocked
        now = time.time()
        if now < self.blocked_until:
            return self.blocked_until - now
        
        # Refill tokens first
        self.refill()
        
        # Calculate tokens needed
        tokens_needed = amount - self.tokens
        
        # If we have enough tokens, no need to wait
        if tokens_needed <= 0:
            return 0.0
        
        # Calculate time to wait
        return tokens_needed / self.refill_rate


@dataclass
class RateLimiter:
    """Rate limiter implementation."""
    
    # Default settings
    default_max_tokens: int = 10
    default_refill_rate: float = 1.0  # Tokens per second
    block_duration: float = 60.0  # Block for 60 seconds after exceeding limits
    
    # Buckets for different keys
    buckets: Dict[str, RateLimitBucket] = field(default_factory=dict)
    
    # Blocked keys
    blocked_keys: Set[str] = field(default_factory=set)
    
    def get_bucket(self, key: str) -> RateLimitBucket:
        """
        Get or create a bucket for a key.
        
        Args:
            key: Bucket key
            
        Returns:
            Rate limit bucket
        """
        if key not in self.buckets:
            self.buckets[key] = RateLimitBucket(
                max_tokens=self.default_max_tokens,
                tokens=self.default_max_tokens,
                refill_rate=self.default_refill_rate,
                last_refill=time.time(),
            )
        
        return self.buckets[key]
    
    def allow_request(self, key: str, amount: int = 1) -> bool:
        """
        Check if a request is allowed.
        
        Args:
            key: Request key
            amount: Token amount to consume
            
        Returns:
            True if request is allowed, False otherwise
        """
        bucket = self.get_bucket(key)
        
        # Check if key is blocked
        if bucket.is_blocked():
            logger.warning(f"Request blocked for key: {key}")
            return False
        
        # Try to consume tokens
        if bucket.consume(amount):
            return True
        
        # Block the key if it exceeds limit
        logger.warning(f"Rate limit exceeded for key: {key}")
        bucket.block(self.block_duration)
        self.blocked_keys.add(key)
        
        return False
    
    def get_wait_time(self, key: str, amount: int = 1) -> float:
        """
        Get time to wait before allowing a request.
        
        Args:
            key: Request key
            amount: Token amount needed
            
        Returns:
            Time to wait in seconds
        """
        bucket = self.get_bucket(key)
        return bucket.get_wait_time(amount)
    
    def reset(self, key: str) -> None:
        """
        Reset the rate limiter for a key.
        
        Args:
            key: Key to reset
        """
        if key in self.buckets:
            bucket = self.buckets[key]
            bucket.tokens = bucket.max_tokens
            bucket.last_refill = time.time()
            bucket.blocked_until = 0.0
            
            if key in self.blocked_keys:
                self.blocked_keys.remove(key)
    
    def clear(self) -> None:
        """Clear all buckets."""
        self.buckets.clear()
        self.blocked_keys.clear()


# Global rate limiter instance
_RATE_LIMITER = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance.
    
    Returns:
        Global rate limiter instance
    """
    return _RATE_LIMITER


def rate_limit(
    key_func: Callable[..., str],
    tokens: int = 1,
    max_tokens: Optional[int] = None,
    refill_rate: Optional[float] = None,
    block_duration: Optional[float] = None,
) -> Callable[[F], F]:
    """
    Decorator for rate limiting functions.
    
    Args:
        key_func: Function to generate a key from function arguments
        tokens: Number of tokens to consume per request
        max_tokens: Maximum tokens in the bucket (defaults to global setting)
        refill_rate: Token refill rate (defaults to global setting)
        block_duration: Duration to block after exceeding limit (defaults to global setting)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate key
            key = key_func(*args, **kwargs)
            
            # Get rate limiter
            limiter = get_rate_limiter()
            
            # Configure bucket if needed
            if max_tokens is not None or refill_rate is not None:
                bucket = limiter.get_bucket(key)
                if max_tokens is not None:
                    bucket.max_tokens = max_tokens
                if refill_rate is not None:
                    bucket.refill_rate = refill_rate
                if block_duration is not None:
                    limiter.block_duration = block_duration
            
            # Check if request is allowed
            if not limiter.allow_request(key, tokens):
                wait_time = limiter.get_wait_time(key, tokens)
                logger.warning(f"Rate limit exceeded for key: {key}, need to wait {wait_time:.2f} seconds")
                raise RateLimitExceeded(key=key, wait_time=wait_time)
            
            # Execute function
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, key: str, wait_time: float):
        """
        Initialize exception.
        
        Args:
            key: Rate limiting key
            wait_time: Time to wait before retrying
        """
        self.key = key
        self.wait_time = wait_time
        super().__init__(f"Rate limit exceeded for key: {key}, need to wait {wait_time:.2f} seconds")


class IPRateLimiter:
    """Rate limiter for IP addresses."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        block_duration: int = 300,
    ):
        """
        Initialize IP rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
            block_duration: Block duration in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.block_duration = block_duration
        
        # Use the global rate limiter
        self.limiter = get_rate_limiter()
        
        # Configure default settings
        self.limiter.default_max_tokens = max_requests
        self.limiter.default_refill_rate = max_requests / window_seconds
        self.limiter.block_duration = block_duration
    
    def is_allowed(self, ip_address: str) -> bool:
        """
        Check if a request from an IP is allowed.
        
        Args:
            ip_address: IP address
            
        Returns:
            True if the request is allowed, False otherwise
        """
        key = f"ip:{ip_address}"
        return self.limiter.allow_request(key)
    
    def get_wait_time(self, ip_address: str) -> float:
        """
        Get wait time for an IP address.
        
        Args:
            ip_address: IP address
            
        Returns:
            Wait time in seconds
        """
        key = f"ip:{ip_address}"
        return self.limiter.get_wait_time(key)
    
    def reset(self, ip_address: str) -> None:
        """
        Reset rate limiting for an IP address.
        
        Args:
            ip_address: IP address
        """
        key = f"ip:{ip_address}"
        self.limiter.reset(key)


class APIRateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(
        self,
        default_max_requests: int = 100,
        default_window_seconds: int = 60,
        default_block_duration: int = 300,
    ):
        """
        Initialize API rate limiter.
        
        Args:
            default_max_requests: Default maximum requests per window
            default_window_seconds: Default window size in seconds
            default_block_duration: Default block duration in seconds
        """
        self.default_max_requests = default_max_requests
        self.default_window_seconds = default_window_seconds
        self.default_block_duration = default_block_duration
        
        # Use the global rate limiter
        self.limiter = get_rate_limiter()
        
        # Endpoint-specific limits
        self.endpoint_limits: Dict[str, Dict[str, int]] = {}
    
    def set_endpoint_limit(
        self,
        endpoint: str,
        max_requests: int,
        window_seconds: int,
        block_duration: int = 300,
    ) -> None:
        """
        Set rate limit for an endpoint.
        
        Args:
            endpoint: API endpoint
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
            block_duration: Block duration in seconds
        """
        self.endpoint_limits[endpoint] = {
            "max_requests": max_requests,
            "window_seconds": window_seconds,
            "block_duration": block_duration,
        }
    
    def is_allowed(self, ip_address: str, endpoint: str) -> bool:
        """
        Check if a request to an endpoint is allowed.
        
        Args:
            ip_address: IP address
            endpoint: API endpoint
            
        Returns:
            True if the request is allowed, False otherwise
        """
        # Generate key
        key = f"api:{ip_address}:{endpoint}"
        
        # Configure bucket if endpoint has specific limits
        if endpoint in self.endpoint_limits:
            limits = self.endpoint_limits[endpoint]
            bucket = self.limiter.get_bucket(key)
            bucket.max_tokens = limits["max_requests"]
            bucket.refill_rate = limits["max_requests"] / limits["window_seconds"]
            self.limiter.block_duration = limits["block_duration"]
        else:
            # Use default settings
            bucket = self.limiter.get_bucket(key)
            bucket.max_tokens = self.default_max_requests
            bucket.refill_rate = self.default_max_requests / self.default_window_seconds
            self.limiter.block_duration = self.default_block_duration
        
        # Check if request is allowed
        return self.limiter.allow_request(key)
    
    def get_wait_time(self, ip_address: str, endpoint: str) -> float:
        """
        Get wait time for an IP and endpoint.
        
        Args:
            ip_address: IP address
            endpoint: API endpoint
            
        Returns:
            Wait time in seconds
        """
        key = f"api:{ip_address}:{endpoint}"
        return self.limiter.get_wait_time(key)
    
    def reset(self, ip_address: str, endpoint: str) -> None:
        """
        Reset rate limiting for an IP and endpoint.
        
        Args:
            ip_address: IP address
            endpoint: API endpoint
        """
        key = f"api:{ip_address}:{endpoint}"
        self.limiter.reset(key)


# HTTP middleware for rate limiting

def create_rate_limit_middleware(rate_limiter: Optional[APIRateLimiter] = None) -> Callable:
    """
    Create a middleware function for rate limiting HTTP requests.
    
    Args:
        rate_limiter: Rate limiter instance (defaults to a new instance)
        
    Returns:
        Middleware function
    """
    # Create rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = APIRateLimiter()
    
    async def rate_limit_middleware(request: Any, handler: Callable) -> Any:
        """
        Middleware function for rate limiting HTTP requests.
        
        Args:
            request: HTTP request
            handler: Next handler function
            
        Returns:
            HTTP response
        """
        # Extract IP address
        try:
            ip_address = request.remote
        except AttributeError:
            try:
                ip_address = request.client.host
            except (AttributeError, TypeError):
                # If we can't get the IP, don't rate limit
                return await handler(request)
        
        # Extract endpoint
        try:
            endpoint = request.path
        except AttributeError:
            endpoint = "/"
        
        # Check if request is allowed
        if not rate_limiter.is_allowed(ip_address, endpoint):
            # Get wait time
            wait_time = rate_limiter.get_wait_time(ip_address, endpoint)
            
            # Return 429 Too Many Requests
            return create_rate_limit_response(wait_time)
        
        # Call next handler
        return await handler(request)
    
    return rate_limit_middleware


def create_rate_limit_response(wait_time: float) -> Any:
    """
    Create a rate limit exceeded HTTP response.
    
    Args:
        wait_time: Time to wait before retrying
        
    Returns:
        HTTP response
    """
    # Try to import aiohttp
    try:
        from aiohttp import web
        return web.json_response(
            {
                "error": "Rate limit exceeded",
                "wait_time": wait_time,
                "message": f"Too many requests, please try again in {wait_time:.2f} seconds",
            },
            status=429,
            headers={"Retry-After": str(int(wait_time))},
        )
    except ImportError:
        # Try to import Flask
        try:
            from flask import jsonify
            response = jsonify({
                "error": "Rate limit exceeded",
                "wait_time": wait_time,
                "message": f"Too many requests, please try again in {wait_time:.2f} seconds",
            })
            response.status_code = 429
            response.headers["Retry-After"] = str(int(wait_time))
            return response
        except ImportError:
            # Return a generic response
            logger.warning("Unable to import aiohttp or flask, returning generic response")
            return {
                "error": "Rate limit exceeded",
                "wait_time": wait_time,
                "message": f"Too many requests, please try again in {wait_time:.2f} seconds",
                "status": 429,
                "headers": {"Retry-After": str(int(wait_time))},
            } 