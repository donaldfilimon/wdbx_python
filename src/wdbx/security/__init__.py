"""
Security components for WDBX.

This module provides security-related components including:
- Authentication and authorization
- Content filtering
- Persona management
- Security monitoring

Most components are enhanced with ML capabilities when available.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import ML components to check availability
try:
    from ..ml.backend import MLBackend

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Import from security submodules
try:
    from .security_manager import ApiKey, AuthProvider, JWTAuthProvider, SecurityManager

    # Create default instances using JWTAuthProvider
    default_auth_provider = JWTAuthProvider(
        secret_key=os.environ.get("WDBX_SECRET_KEY", "default_secret_key"),
        algorithm="HS256",
        expires_delta_seconds=3600,
    )
except ImportError:
    # Define fallback classes if imports fail
    class AuthProvider:
        """Fallback AuthProvider when real implementation is unavailable."""

        def authenticate(self, credentials):
            return None

        def generate_token(self, user_info):
            return None

        def validate_token(self, token):
            return None

    class SecurityManager:
        """Fallback SecurityManager when real implementation is unavailable."""

        def __init__(self):
            pass

        def authenticate(self, credentials):
            return None

        def generate_token(self, user_info):
            return None

        def validate_token(self, token):
            return None

    # Create default instances using fallback classes
    default_auth_provider = AuthProvider()

# Import content filtering
try:
    from .content_filter import ContentFilter, ContentSafetyLevel, ContentTopic
except ImportError:
    # Define fallback content filter if import fails
    class ContentSafetyLevel:
        NONE = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    class ContentTopic:
        VIOLENCE = "violence"
        HATE_SPEECH = "hate_speech"
        SEXUAL = "sexual"

    class ContentFilter:
        """Fallback ContentFilter when real implementation is unavailable."""

        def __init__(self, safety_level=None):
            pass

        def filter_content(self, content, safety_level=None):
            return content

        def check_safety(self, content):
            return {"safe": True, "categories": {}}


# Import persona management
try:
    from .persona import BiasDetector, PersonaAttributes, PersonaManager, PersonaTokenManager
except ImportError:
    # Define fallback classes if import fails
    class PersonaAttributes:
        FORMALITY = "formality"
        VERBOSITY = "verbosity"

    class PersonaTokenManager:
        """Fallback PersonaTokenManager when real implementation is unavailable."""

        def create_token(self, persona_id, attributes=None):
            return None

        def validate_token(self, token):
            return None

    class PersonaManager:
        """Fallback PersonaManager when real implementation is unavailable."""

        def __init__(self):
            self.token_manager = PersonaTokenManager()

        def create_persona(self, name, description, attributes):
            return None

        def apply_persona(self, token, content, safe_mode=True):
            return {"content": content, "modified": False}

    class BiasDetector:
        """Fallback BiasDetector when real implementation is unavailable."""

        def detect_bias(self, content):
            return {"detected": False}


# Create default instances of commonly used classes
default_security_manager = SecurityManager()
default_content_filter = ContentFilter()
default_persona_manager = PersonaManager()
default_bias_detector = BiasDetector()

# Export all relevant components
__all__ = [
    # Security manager components
    "SecurityManager",
    "AuthProvider",
    "JWTAuthProvider",
    "ApiKey",
    # Content filtering
    "ContentFilter",
    "ContentSafetyLevel",
    "ContentTopic",
    # Persona management
    "PersonaManager",
    "PersonaTokenManager",
    "PersonaAttributes",
    "BiasDetector",
    # Default instances
    "default_auth_provider",
    "default_security_manager",
    "default_content_filter",
    "default_persona_manager",
    "default_bias_detector",
    # ML availability flag
    "ML_AVAILABLE",
]
