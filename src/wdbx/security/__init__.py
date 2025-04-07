"""
Security components for WDBX.

Includes authentication, authorization, content filtering, and persona management.
"""

from .content_filter import BiasDetector, ContentFilter
from .persona import PersonaManager, PersonaTokenManager

# from .encryption import encrypt_data, decrypt_data  # If encryption module exists
# from .access_control import check_permission         # If access control module exists
from .security_manager import AuthProvider, SecurityManager

# Import security manager components conditionally
try:
    from wdbx.security.security_manager import AuthProvider, SecurityManager
except ImportError:
    # Create dummy classes as fallback
    class SecurityManager:
        """Fallback for when security manager isn't available."""

    class AuthProvider:
        """Fallback for when auth provider isn't available."""

__all__ = [
    # Content filtering
    "BiasDetector",
    "ContentFilter",
    
    # Persona management
    "PersonaManager",
    "PersonaTokenManager",
    
    # Security manager
    "SecurityManager",
    "AuthProvider",
    
    # Add other security components if they exist
    # 'encrypt_data',
    # 'decrypt_data',
    # 'check_permission',
]
