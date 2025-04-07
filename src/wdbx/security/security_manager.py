# wdbx/security.py
"""
Security and authentication system for WDBX.

This module provides robust security features for WDBX, including:
- API key authentication
- Role-based access control
- JWT token support
- Rate limiting
- Audit logging
"""
import base64
import hashlib
import ipaddress
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, cast

# Import the logger first to avoid circular imports
from ..core.constants import logger

# Attempt to import JWT library
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not available. JWT authentication will not be available.")

# Attempt to import OAuth library
try:
    from authlib.integrations.starlette_client import OAuth  # Example using authlib
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False

# Secret key for signing tokens - use environment variable or generate a secure random key
SECRET_KEY = os.environ.get("WDBX_SECRET_KEY", os.urandom(32).hex())

# Default expiration time for tokens (24 hours)
TOKEN_EXPIRATION = int(os.environ.get("WDBX_TOKEN_EXPIRATION", 86400))

# Default rate limits
RATE_LIMIT_WINDOW = int(os.environ.get("WDBX_RATE_LIMIT_WINDOW", 60))  # 1 minute
RATE_LIMIT_MAX_REQUESTS = int(
    os.environ.get(
        "WDBX_RATE_LIMIT_MAX_REQUESTS",
        100))  # 100 requests per minute

# Default roles and permissions
DEFAULT_ROLES = {
    "admin": ["read", "write", "delete", "manage"],
    "writer": ["read", "write"],
    "reader": ["read"]
}

# Audit log configuration
AUDIT_LOG_FILE = os.environ.get("WDBX_AUDIT_LOG", "wdbx_audit.log")


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""
    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate user based on credentials. Return user info or None."""
        pass

    @abstractmethod
    def generate_token(self, user_info: Dict[str, Any]) -> Optional[str]:
        """Generate an authentication token for the user."""
        pass

    @abstractmethod
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token and return user info or None."""
        pass


class JWTAuthProvider(AuthProvider):
    """JWT-based authentication provider."""
    def __init__(self, secret_key: str, algorithm: str = "HS256", expires_delta_seconds: int = 3600):
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT library is required for JWTAuthProvider. Install with: pip install pyjwt")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expires_delta_seconds = expires_delta_seconds
        logger.info(f"JWTAuthProvider initialized with algorithm {self.algorithm}")

    def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate using username/password (placeholder)."""
        # In a real scenario, verify credentials against a user database
        username = credentials.get("username")
        password = credentials.get("password")
        if username == "testuser" and password == "password123":
            logger.info(f"Authenticated user: {username}")
            return {"user_id": "123", "username": username, "roles": ["user"]}
        logger.warning(f"Authentication failed for username: {username}")
        return None

    def generate_token(self, user_info: Dict[str, Any]) -> Optional[str]:
        """Generate a JWT token."""
        try:
            payload = {
                "sub": user_info["user_id"],
                "username": user_info["username"],
                "roles": user_info.get("roles", []),
                "exp": time.time() + self.expires_delta_seconds,
                "iat": time.time()
            }
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"Generated JWT token for user {user_info['username']}")
            return token
        except Exception as e:
            logger.error(f"Error generating JWT token: {e}")
            return None

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            # Basic validation successful, return payload containing user info
            logger.debug(f"Validated JWT token for user {payload.get('username')}")
            return payload 
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token validation failed: Expired signature")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"JWT token validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error validating JWT token: {e}")
            return None


class APIKeyAuthProvider(AuthProvider):
    """Simple API Key based authentication (placeholder)."""
    def __init__(self, valid_keys: Dict[str, Dict[str, Any]]):
        # valid_keys format: {"api_key_value": {"user_id": "...", "roles": [...]}} 
        self.valid_keys = valid_keys
        logger.info(f"APIKeyAuthProvider initialized with {len(valid_keys)} keys.")

    def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """API Keys are typically validated directly, not authenticated via credentials."""
        return None # Direct validation via validate_token

    def generate_token(self, user_info: Dict[str, Any]) -> Optional[str]:
        """Generate a secure API key (example)."""
        # In production, use a more robust generation method and store securely
        key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        # Store the key and associate it with user_info (outside this simple example)
        self.valid_keys[key] = user_info
        logger.info(f"Generated API key for user {user_info.get('user_id')}")
        return key

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        user_info = self.valid_keys.get(token)
        if user_info:
            logger.debug(f"Validated API key for user {user_info.get('user_id')}")
            return user_info
        logger.warning("API key validation failed: Invalid key")
        return None


class ApiKey:
    """
    Represents an API key with associated metadata and permissions.
    """

    def __init__(self, key_id: str, key_secret: str, name: str, role: str,
                 owner: str = "", expires_at: Optional[int] = None,
                 enabled: bool = True, ip_whitelist: Optional[List[str]] = None):
        """
        Initialize an API key.

        Args:
            key_id: Unique identifier for the key
            key_secret: Secret value for the key (hashed before storage)
            name: Human-readable name for the key
            role: Role associated with the key (determines permissions)
            owner: Owner of the key
            expires_at: Timestamp when the key expires (None for no expiration)
            enabled: Whether the key is currently enabled
            ip_whitelist: List of allowed IP addresses (None for no restriction)
        """
        self.key_id = key_id
        self.key_hash = self._hash_secret(key_secret) if key_secret else ""
        self.name = name
        self.role = role
        self.owner = owner
        self.expires_at = expires_at
        self.enabled = enabled
        self.ip_whitelist = ip_whitelist or []
        self.created_at = int(time.time())
        self.last_used_at: Optional[int] = None

    def _hash_secret(self, secret: str) -> str:
        """
        Securely hash the key secret using PBKDF2-HMAC-SHA256.

        Args:
            secret: Secret to hash

        Returns:
            Hashed secret with salt
        """
        salt = hashlib.sha256(os.urandom(60)).hexdigest().encode("utf-8")
        hash_obj = hashlib.pbkdf2_hmac(
            "sha256",
            secret.encode("utf-8"),
            salt,
            100000  # High iteration count for security
        )
        return salt.decode("utf-8") + ":" + base64.b64encode(hash_obj).decode("utf-8")

    def verify_secret(self, secret: str) -> bool:
        """
        Verify a secret against the stored hash.

        Args:
            secret: Secret to verify

        Returns:
            True if the secret is valid, False otherwise
        """
        if not self.key_hash or ":" not in self.key_hash:
            return False

        salt, stored_hash = self.key_hash.split(":")
        hash_obj = hashlib.pbkdf2_hmac(
            "sha256",
            secret.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        )
        return base64.b64encode(hash_obj).decode("utf-8") == stored_hash

    def is_valid(self, ip_address: Optional[str] = None) -> bool:
        """
        Check if the key is valid.

        Args:
            ip_address: IP address to check against whitelist

        Returns:
            True if the key is valid, False otherwise
        """
        # Check if key is enabled
        if not self.enabled:
            return False

        # Check expiration
        if self.expires_at and time.time() > self.expires_at:
            return False

        # Check IP whitelist
        if ip_address and self.ip_whitelist:
            try:
                client_ip = ipaddress.ip_address(ip_address)
                allowed = False
                for allowed_ip in self.ip_whitelist:
                    # Handle IP networks (CIDR notation)
                    if "/" in allowed_ip:
                        if client_ip in ipaddress.ip_network(allowed_ip, strict=False):
                            allowed = True
                            break
                    # Handle individual IPs
                    elif client_ip == ipaddress.ip_address(allowed_ip):
                        allowed = True
                        break

                if not allowed:
                    return False
            except ValueError:
                # Invalid IP address
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the API key
        """
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "name": self.name,
            "role": self.role,
            "owner": self.owner,
            "expires_at": self.expires_at,
            "enabled": self.enabled,
            "ip_whitelist": self.ip_whitelist,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiKey":
        """
        Create an API key from a dictionary.

        Args:
            data: Dictionary representation of the API key

        Returns:
            ApiKey object
        """
        key = cls(
            key_id=data["key_id"],
            key_secret="",  # Secret is already hashed
            name=data["name"],
            role=data["role"],
            owner=data.get("owner", ""),
            expires_at=data.get("expires_at"),
            enabled=data.get("enabled", True),
            ip_whitelist=data.get("ip_whitelist", [])
        )
        key.key_hash = data["key_hash"]
        key.created_at = data.get("created_at", int(time.time()))
        key.last_used_at = data.get("last_used_at")
        return key


class RateLimiter:
    """
    Implements a sliding window rate limiter.
    """

    def __init__(self, window_size: int = RATE_LIMIT_WINDOW,
                 max_requests: int = RATE_LIMIT_MAX_REQUESTS):
        """
        Initialize the rate limiter.

        Args:
            window_size: Size of the sliding window in seconds
            max_requests: Maximum number of requests allowed in the window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: Dict[str, List[float]] = {}  # key -> list of timestamps
        self.lock = threading.RLock()

    def check_rate_limit(self, key: str) -> Tuple[bool, int]:
        """
        Check if a key has exceeded its rate limit.

        Args:
            key: Key to check

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        with self.lock:
            now = time.time()

            # Initialize if this is the first request for this key
            if key not in self.requests:
                self.requests[key] = []

            # Remove timestamps outside the window
            self.requests[key] = [ts for ts in self.requests[key] if ts > now - self.window_size]

            # Check if the key has exceeded the rate limit
            if len(self.requests[key]) >= self.max_requests:
                return False, 0

            # Record the current request
            self.requests[key].append(now)

            # Return the number of remaining requests
            return True, self.max_requests - len(self.requests[key])

    def clear(self, key: Optional[str] = None):
        """
        Clear rate limit history.

        Args:
            key: Key to clear, or None to clear all
        """
        with self.lock:
            if key:
                self.requests.pop(key, None)
            else:
                self.requests.clear()


class SecurityManager:
    """
    Manages authentication and authorization for WDBX.

    Coordinates different authentication providers and enforces access control.
    """
    def __init__(self, config=None): # Accept optional config object
        self.providers: Dict[str, AuthProvider] = {}
        self.default_provider: Optional[str] = None
        self.config = config # Store config if provided
        logger.info("SecurityManager initialized.")
        self._configure_providers(config) # Configure based on config
        self.api_keys: Dict[str, ApiKey] = {}
        self.roles: Dict[str, List[str]] = DEFAULT_ROLES.copy()
        self.rate_limiter = RateLimiter()
        self.lock = threading.RLock()

        # Set up audit logging
        self.audit_logger = self._setup_audit_logger()

        # Load existing data if available
        if config and config.get("storage_path"):
            self._load_data()

    def _setup_audit_logger(self) -> logging.Logger:
        """
        Set up the audit logger.

        Returns:
            Configured logger
        """
        audit_logger = logging.getLogger("WDBX.Audit")
        audit_logger.setLevel(logging.INFO)

        # Create file handler
        try:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(AUDIT_LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            handler = logging.FileHandler(AUDIT_LOG_FILE)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
        except Exception as e:
            logger.warning(f"Failed to set up audit log file: {e}")
            # Use console handler as fallback
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "AUDIT: %(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)

        return audit_logger

    def _configure_providers(self, config) -> None:
        """Configure auth providers based on WDBXConfig (if available)."""
        if not config:
            logger.warning("No configuration provided to SecurityManager. Auth providers may not be configured.")
            return
            
        # Example: Configure JWT provider if secret is set
        if hasattr(config, "jwt_secret") and config.jwt_secret and JWT_AVAILABLE:
            try:
                jwt_provider = JWTAuthProvider(secret_key=config.jwt_secret)
                self.register_provider("jwt", jwt_provider)
                if not self.default_provider:
                    self.set_default_provider("jwt")
            except Exception as e:
                logger.error(f"Failed to initialize JWTAuthProvider: {e}")
                
        # Example: Configure API Key provider (needs keys in config)
        # if hasattr(config, 'api_keys') and config.api_keys:
        #     api_key_provider = APIKeyAuthProvider(valid_keys=config.api_keys)
        #     self.register_provider("apikey", api_key_provider)

    def register_provider(self, name: str, provider: AuthProvider) -> None:
        """Register an authentication provider."""
        self.providers[name] = provider
        logger.info(f"Registered authentication provider: {name} ({type(provider).__name__})")
        if self.default_provider is None:
            self.set_default_provider(name)

    def set_default_provider(self, name: str) -> None:
        """Set the default authentication provider."""
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not registered.")
        self.default_provider = name
        logger.info(f"Set default authentication provider to: {name}")

    def authenticate(self, credentials: Dict[str, Any], provider_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Authenticate using a specific or default provider."""
        name = provider_name or self.default_provider
        if not name or name not in self.providers:
            logger.error(f"Authentication failed: Provider '{name}' not available.")
            return None
        
        provider = self.providers[name]
        try:
            user_info = provider.authenticate(credentials)
            if user_info:
                logger.info(f"Authentication successful via provider '{name}' for user {user_info.get('username') or user_info.get('user_id')}")
            return user_info
        except Exception as e:
            logger.error(f"Error during authentication with provider '{name}': {e}", exc_info=True)
            return None
            
    def generate_token(self, user_info: Dict[str, Any], provider_name: Optional[str] = None) -> Optional[str]:
        """Generate a token using a specific or default provider."""
        name = provider_name or self.default_provider
        if not name or name not in self.providers:
            logger.error(f"Token generation failed: Provider '{name}' not available.")
            return None
        
        provider = self.providers[name]
        try:
            return provider.generate_token(user_info)
        except Exception as e:
            logger.error(f"Error generating token with provider '{name}': {e}", exc_info=True)
            return None

    def validate_token(self, token: str, provider_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Validate a token using a specific or all available providers."""
        providers_to_try = []
        if provider_name:
            if provider_name in self.providers:
                providers_to_try.append(self.providers[provider_name])
            else:
                 logger.error(f"Token validation failed: Provider '{provider_name}' not available.")
                 return None
        else:
            # Try all registered providers if none is specified
            providers_to_try.extend(self.providers.values())

        if not providers_to_try:
             logger.error("Token validation failed: No authentication providers registered.")
             return None
             
        for provider in providers_to_try:
            try:
                user_info = provider.validate_token(token)
                if user_info:
                    logger.info(f"Token validated successfully via provider {type(provider).__name__} for user {user_info.get('username') or user_info.get('user_id')}")
                    return user_info # Return on first successful validation
            except Exception as e:
                 logger.error(f"Error validating token with provider {type(provider).__name__}: {e}", exc_info=True)
                 # Continue to try other providers

        logger.warning("Token validation failed: Invalid or expired token, or no provider could validate it.")
        return None
        
    def authorize(self, user_info: Dict[str, Any], required_role: Optional[str] = None, required_permission: Optional[str] = None) -> bool:
        """Check if the authenticated user has the required roles/permissions."""
        if not user_info: # Not authenticated
            logger.warning("Authorization check failed: User not authenticated.")
            return False
        
        user_roles = user_info.get("roles", [])
        user_permissions = user_info.get("permissions", []) # Assuming permissions might exist

        # Role check
        if required_role:
            if required_role not in user_roles:
                logger.warning(f"Authorization failed for user {user_info.get('username')}: Missing required role '{required_role}'. User roles: {user_roles}")
                return False

        # Permission check (example)
        if required_permission:
            if required_permission not in user_permissions:
                # Check if any role grants the permission (more complex logic needed)
                # For now, simple check against direct permissions
                logger.warning(f"Authorization failed for user {user_info.get('username')}: Missing required permission '{required_permission}'. User permissions: {user_permissions}")
                return False

        logger.debug(f"Authorization successful for user {user_info.get('username')} (Required role: {required_role}, Perm: {required_permission})")
        return True

    def _load_data(self):
        """Load security data from storage."""
        if not self.config or not self.config.get("storage_path"):
            return

        try:
            os.makedirs(self.config["storage_path"], exist_ok=True)

            # Load API keys
            api_keys_path = os.path.join(self.config["storage_path"], "api_keys.json")
            if os.path.exists(api_keys_path):
                with open(api_keys_path, encoding="utf-8") as f:
                    keys_data = json.load(f)
                    for key_data in keys_data:
                        try:
                            key = ApiKey.from_dict(key_data)
                            self.api_keys[key.key_id] = key
                        except KeyError as e:
                            logger.warning(f"Skipping API key due to missing key: {e}")

            # Load roles
            roles_path = os.path.join(self.config["storage_path"], "roles.json")
            if os.path.exists(roles_path):
                with open(roles_path, encoding="utf-8") as f:
                    self.roles = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load security data: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error loading security data: {e}")

    def _save_data(self):
        """Save security data to storage."""
        if not self.config or not self.config.get("storage_path"):
            return

        try:
            os.makedirs(self.config["storage_path"], exist_ok=True)

            # Save API keys
            api_keys_path = os.path.join(self.config["storage_path"], "api_keys.json")
            temp_path = f"{api_keys_path}.tmp"

            # Write to temporary file first to prevent corruption if write fails
            with open(temp_path, "w", encoding="utf-8") as f:
                keys_data = [key.to_dict() for key in self.api_keys.values()]
                json.dump(keys_data, f, indent=2)

            # Rename to actual file
            os.replace(temp_path, api_keys_path)

            # Save roles
            roles_path = os.path.join(self.config["storage_path"], "roles.json")
            temp_path = f"{roles_path}.tmp"

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.roles, f, indent=2)

            os.replace(temp_path, roles_path)

        except (TypeError, FileNotFoundError) as e:
            logger.error(f"Failed to save security data: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error saving security data: {e}")

    def create_api_key(self,
                       name: str,
                       role: str,
                       owner: str = "",
                       expires_in: Optional[int] = None,
                       ip_whitelist: Optional[List[str]] = None) -> Tuple[str,
                                                                          str]:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            role: Role associated with the key
            owner: Owner of the key
            expires_in: Number of seconds until the key expires, or None for no expiration
            ip_whitelist: List of allowed IP addresses, or None for no restriction

        Returns:
            Tuple of (key_id, key_secret)

        Raises:
            ValueError: If the role is invalid
        """
        with self.lock:
            # Check if the role exists
            if role not in self.roles:
                raise ValueError(f"Invalid role: {role}")

            # Generate key ID and secret
            key_id = str(uuid.uuid4())
            key_secret = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8")

            # Calculate expiration timestamp
            expires_at = None
            if expires_in is not None:
                expires_at = int(time.time()) + expires_in

            # Create the API key
            api_key = ApiKey(
                key_id=key_id,
                key_secret=key_secret,
                name=name,
                role=role,
                owner=owner,
                expires_at=expires_at,
                ip_whitelist=ip_whitelist
            )

            # Store the key
            self.api_keys[key_id] = api_key

            # Save to storage
            self._save_data()

            # Log the action
            self.audit_logger.info(
                f"API key created: id={key_id}, name={name}, role={role}, "
                f"owner={owner}, expires_at={expires_at}"
            )

            # Return the key credentials
            return key_id, key_secret

    def validate_api_key(self, key_id: str, key_secret: str,
                         ip_address: Optional[str] = None) -> bool:
        """
        Validate an API key.

        Args:
            key_id: Key ID to validate
            key_secret: Key secret to validate
            ip_address: IP address to check against whitelist

        Returns:
            True if the key is valid, False otherwise
        """
        with self.lock:
            # Check if the key exists
            api_key = self.api_keys.get(key_id)
            if not api_key:
                return False

            # Verify the secret
            if not api_key.verify_secret(key_secret):
                return False

            # Check if the key is valid
            if not api_key.is_valid(ip_address):
                return False

            # Update last used timestamp
            api_key.last_used_at = int(time.time())

            # Apply rate limiting
            allowed, remaining = self.rate_limiter.check_rate_limit(key_id)
            if not allowed:
                self.audit_logger.warning(f"Rate limit exceeded for API key: {key_id}")
                return False

            return True

    def get_permissions(self, key_id: str) -> List[str]:
        """
        Get permissions for an API key.

        Args:
            key_id: Key ID to get permissions for

        Returns:
            List of permissions, or empty list if the key doesn't exist
        """
        with self.lock:
            api_key = self.api_keys.get(key_id)
            if not api_key:
                return []

            # Check if the key is valid
            if not api_key.is_valid():
                return []

            # Get permissions for the key's role
            return self.roles.get(api_key.role, [])

    def check_permission(self, key_id: str, permission: str) -> bool:
        """
        Check if an API key has a specific permission.

        Args:
            key_id: Key ID to check
            permission: Permission to check for

        Returns:
            True if the key has the permission, False otherwise
        """
        permissions = self.get_permissions(key_id)
        return permission in permissions

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: Key ID to revoke

        Returns:
            True if the key was revoked, False otherwise
        """
        with self.lock:
            api_key = self.api_keys.get(key_id)
            if not api_key:
                return False

            # Disable the key
            api_key.enabled = False

            # Save to storage
            self._save_data()

            # Log the action
            self.audit_logger.info(f"API key revoked: {key_id}")

            return True

    def delete_api_key(self, key_id: str) -> bool:
        """
        Delete an API key.

        Args:
            key_id: Key ID to delete

        Returns:
            True if the key was deleted, False otherwise
        """
        with self.lock:
            if key_id not in self.api_keys:
                return False

            # Delete the key
            del self.api_keys[key_id]

            # Save to storage
            self._save_data()

            # Log the action
            self.audit_logger.info(f"API key deleted: {key_id}")

            # Clear rate limit history for this key
            self.rate_limiter.clear(key_id)

            return True

    def create_role(self, role_name: str, permissions: List[str]) -> bool:
        """
        Create or update a role.

        Args:
            role_name: Name of the role
            permissions: List of permissions for the role

        Returns:
            True if the role was created/updated, False otherwise
        """
        with self.lock:
            # Validate role name
            if not role_name:
                logger.error("Role name cannot be empty")
                return False

            # Check for invalid characters in role name
            if not role_name.isalnum() and role_name.replace("_", "").isalnum():
                logger.warning(
                    f"Role name '{role_name}' contains underscores. Using underscores is allowed but alphanumeric names are preferred.")
            elif not role_name.isalnum():
                logger.warning(
                    f"Role name '{role_name}' contains non-alphanumeric characters. This is discouraged.")

            # Store the role
            self.roles[role_name] = list(set(permissions))  # Remove duplicates

            # Save to storage
            self._save_data()

            # Log the action
            self.audit_logger.info(f"Role created/updated: {role_name}, permissions={permissions}")

            return True

    def delete_role(self, role_name: str) -> bool:
        """
        Delete a role.

        Args:
            role_name: Name of the role to delete

        Returns:
            True if the role was deleted, False otherwise
        """
        with self.lock:
            if role_name not in self.roles:
                return False

            # Check if any API keys are using this role
            key_ids_with_role = [
                key.key_id for key in self.api_keys.values() if key.role == role_name]
            if key_ids_with_role:
                logger.warning(
                    f"Cannot delete role '{role_name}'. API keys still using it: {key_ids_with_role}")
                return False

            # Delete the role
            del self.roles[role_name]

            # Save to storage
            self._save_data()

            # Log the action
            self.audit_logger.info(f"Role deleted: {role_name}")

            return True

    def generate_jwt_token(self, key_id: str, expiration: int = TOKEN_EXPIRATION) -> Optional[str]:
        """
        Generate a JWT token for an API key.

        Args:
            key_id: Key ID to generate token for
            expiration: Token expiration time in seconds

        Returns:
            JWT token, or None if JWT is not available or key is invalid
        """
        if not JWT_AVAILABLE:
            logger.warning("JWT authentication not available")
            return None

        with self.lock:
            api_key = self.api_keys.get(key_id)
            if not api_key:
                return None

            # Check if the key is valid
            if not api_key.is_valid():
                return None

            # Prepare token payload
            now = datetime.utcnow()
            payload = {
                "sub": key_id,
                "name": api_key.name,
                "role": api_key.role,
                "iat": now,
                "exp": now + timedelta(seconds=expiration)
            }

            # Generate token
            try:
                if JWT_AVAILABLE:
                    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
                    # In PyJWT >=2.0.0, encode returns a string, but in <2.0.0 it returns bytes
                    if isinstance(token, bytes):
                        token = token.decode("utf-8")
                    return token
                return None
            except Exception as e:
                logger.error(f"Failed to generate JWT token: {e}")
                return None

            # Log the action
            self.audit_logger.info(f"JWT token generated for API key: {key_id}")

            return token

    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token.

        Args:
            token: JWT token to validate

        Returns:
            Token payload if valid, None otherwise
        """
        if not JWT_AVAILABLE:
            logger.warning("JWT authentication not available")
            return None

        try:
            # Decode and verify the token
            if JWT_AVAILABLE:
                payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

                # Check if the key exists and is valid
                key_id = payload.get("sub")
                if not key_id or key_id not in self.api_keys:
                    return None

                api_key = self.api_keys[key_id]
                if not api_key.is_valid():
                    return None

                # Apply rate limiting
                allowed, remaining = self.rate_limiter.check_rate_limit(key_id)
                if not allowed:
                    self.audit_logger.warning(f"Rate limit exceeded for API key: {key_id}")
                    return None

                # Update last used timestamp
                api_key.last_used_at = int(time.time())

                return payload
            return None
        except Exception as e:
            if JWT_AVAILABLE:
                if isinstance(e, jwt.ExpiredSignatureError):
                    logger.warning("JWT token has expired")
                elif isinstance(e, jwt.InvalidTokenError):
                    logger.warning("Invalid JWT token")
                else:
                    logger.error(f"Failed to validate JWT token: {e}")
            return None

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                           level: str = "INFO") -> None:
        """
        Log a security event to the audit log.

        Args:
            event_type: Type of event
            details: Event details
            level: Log level
        """
        # Sanitize sensitive data from logs
        sanitized_details = details.copy()
        if "token" in sanitized_details:
            sanitized_details["token"] = "***REDACTED***"
        if "key_secret" in sanitized_details:
            sanitized_details["key_secret"] = "***REDACTED***"

        message = f"{event_type}: {json.dumps(sanitized_details)}"

        if level == "DEBUG":
            self.audit_logger.debug(message)
        elif level == "INFO":
            self.audit_logger.info(message)
        elif level == "WARNING":
            self.audit_logger.warning(message)
        elif level == "ERROR":
            self.audit_logger.error(message)
        elif level == "CRITICAL":
            self.audit_logger.critical(message)


def require_auth(permission: Optional[str] = None):
    """
    Decorator for requiring API key authentication and permission check.
    For use with Flask or other web frameworks.

    Args:
        permission: Required permission, or None for just authentication
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Import Flask here to avoid circular imports
            from flask import current_app, jsonify, request

            # Get security manager
            app = current_app
            security_manager = cast(SecurityManager, getattr(app, "security_manager", None))
            if not security_manager:
                return jsonify({"error": "Security manager not available"}), 500

            # Check API key authentication
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                # Check JWT token
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    if token:
                        payload = security_manager.validate_jwt_token(token)
                        if payload:
                            key_id = payload.get("sub")
                            if not key_id:
                                security_manager.log_security_event(
                                    "Invalid JWT payload",
                                    {
                                        "endpoint": request.path,
                                        "method": request.method,
                                        "ip": request.remote_addr
                                    },
                                    "WARNING"
                                )
                                return jsonify({"error": "Invalid JWT payload"}), 401

                            # Check permission if required
                            if permission and not security_manager.check_permission(
                                    key_id, permission):
                                security_manager.log_security_event(
                                    "Permission denied",
                                    {
                                        "key_id": key_id,
                                        "permission": permission,
                                        "endpoint": request.path,
                                        "method": request.method,
                                        "ip": request.remote_addr
                                    },
                                    "WARNING"
                                )
                                return jsonify({"error": "Permission denied"}), 403

                            # Log the access
                            security_manager.log_security_event(
                                "API access",
                                {
                                    "key_id": key_id,
                                    "endpoint": request.path,
                                    "method": request.method,
                                    "ip": request.remote_addr
                                }
                            )

                            return f(*args, **kwargs)

                return jsonify({"error": "Authentication required"}), 401

            # Split API key into ID and secret
            key_parts = api_key.split(":")
            if len(key_parts) != 2:
                return jsonify({"error": "Invalid API key format"}), 401

            key_id, key_secret = key_parts

            # Validate API key
            if not security_manager.validate_api_key(key_id, key_secret, request.remote_addr):
                security_manager.log_security_event(
                    "Authentication failed",
                    {
                        "key_id": key_id,
                        "endpoint": request.path,
                        "method": request.method,
                        "ip": request.remote_addr
                    },
                    "WARNING"
                )
                return jsonify({"error": "Invalid API key"}), 401

            # Check permission if required
            if permission and not security_manager.check_permission(key_id, permission):
                security_manager.log_security_event(
                    "Permission denied",
                    {
                        "key_id": key_id,
                        "permission": permission,
                        "endpoint": request.path,
                        "method": request.method,
                        "ip": request.remote_addr
                    },
                    "WARNING"
                )
                return jsonify({"error": "Permission denied"}), 403

            # Log the access
            security_manager.log_security_event(
                "API access",
                {
                    "key_id": key_id,
                    "endpoint": request.path,
                    "method": request.method,
                    "ip": request.remote_addr
                }
            )

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def init_security(app, storage_path: Optional[str] = None) -> SecurityManager:
    """
    Initialize security for a Flask application.

    Args:
        app: Flask application
        storage_path: Path to store security data, or None for in-memory only

    Returns:
        SecurityManager instance
    """
    # Try to make storage path absolute if it's relative
    if storage_path and not os.path.isabs(storage_path):
        try:
            app_root = os.path.abspath(os.path.dirname(app.root_path))
            storage_path = os.path.join(app_root, storage_path)
        except AttributeError:
            # Not a Flask app or no root_path
            pass

    security_manager = SecurityManager(storage_path)
    app.security_manager = security_manager

    # Create default admin key if no keys exist
    if not security_manager.api_keys:
        key_id, key_secret = security_manager.create_api_key(
            name="Default Admin Key",
            role="admin",
            owner="system"
        )
        logger.info(f"Created default admin API key: {key_id}:{key_secret}")
        logger.info("IMPORTANT: Save this key, it will not be shown again!")

    # Add security-related routes
    @app.route("/security/health", methods=["GET"])
    def security_health():
        """Check if security is healthy."""
        from flask import jsonify
        return jsonify({"status": "healthy", "timestamp": time.time()})

    @app.route("/security/keys", methods=["GET"])
    @require_auth("manage")
    def list_keys():
        """List all API keys."""
        from flask import jsonify
        keys = []
        for key in security_manager.api_keys.values():
            # Don't include the key hash
            key_data = key.to_dict()
            del key_data["key_hash"]
            keys.append(key_data)
        return jsonify(keys)

    @app.route("/security/roles", methods=["GET"])
    @require_auth("manage")
    def list_roles():
        """List all roles."""
        from flask import jsonify
        return jsonify(security_manager.roles)

    logger.info("Security initialized")
    return security_manager


# Unit tests
def test_security():
    """Run unit tests for the security module."""
    import unittest

    class SecurityTest(unittest.TestCase):
        def setUp(self):
            self.security = SecurityManager()

        def test_api_key_creation(self):
            key_id, key_secret = self.security.create_api_key("Test Key", "reader")
            self.assertIsNotNone(key_id)
            self.assertIsNotNone(key_secret)
            self.assertTrue(self.security.validate_api_key(key_id, key_secret))

        def test_permission_check(self):
            key_id, _ = self.security.create_api_key("Test Key", "reader")
            self.assertTrue(self.security.check_permission(key_id, "read"))
            self.assertFalse(self.security.check_permission(key_id, "write"))

        def test_role_management(self):
            self.security.create_role("custom", ["read", "custom_action"])
            key_id, _ = self.security.create_api_key("Custom Key", "custom")
            self.assertTrue(self.security.check_permission(key_id, "custom_action"))

        def test_key_revocation(self):
            key_id, key_secret = self.security.create_api_key("Test Key", "reader")
            self.security.revoke_api_key(key_id)
            self.assertFalse(self.security.validate_api_key(key_id, key_secret))

        def test_jwt_token(self):
            if not JWT_AVAILABLE:
                self.skipTest("JWT not available")

            key_id, _ = self.security.create_api_key("JWT Key", "reader")
            token = self.security.generate_jwt_token(key_id)
            self.assertIsNotNone(token)

            payload = self.security.validate_jwt_token(token)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["sub"], key_id)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)


if __name__ == "__main__":
    test_security()
