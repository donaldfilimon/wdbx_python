"""
Persona management for WDBX interactions.

This module provides classes for managing personas and persona tokens,
ensuring secure and consistent interactions with different personality profiles.
"""

import base64
import hashlib
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

# Import the logger from core
from ..core.constants import logger

# Try to import ML components for enhanced personality detection
try:
    from ..ml import VectorLike
    from ..ml.backend import MLBackend

    ML_AVAILABLE = True
    ml_backend = MLBackend()
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML components not available for enhanced persona management")

# Try to import monitoring
try:
    from ..monitoring.performance import PerformanceMonitor

    MONITORING_AVAILABLE = True
    performance_monitor = PerformanceMonitor()
except ImportError:
    MONITORING_AVAILABLE = False

# Import content filter to apply safety limits to persona-generated content
from .content_filter import ContentFilter, ContentSafetyLevel


class PersonaTokenManager:
    """
    Manager for persona tokens ensuring secure persona authentication.

    This class handles the creation, validation, and management of
    tokens that authenticate specific personas for interactions.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the token manager.

        Args:
            secret_key: Secret key for token signing, uses env var or generates one if None
        """
        self.secret_key = secret_key or os.environ.get("WDBX_PERSONA_SECRET", os.urandom(32).hex())

        if secret_key is None and "WDBX_PERSONA_SECRET" not in os.environ:
            logger.warning(
                "No WDBX_PERSONA_SECRET found. Generated temporary key. "
                "For production, set a strong WDBX_PERSONA_SECRET environment variable."
            )

        # Token expiration in seconds (default: 24 hours)
        self.token_expiration = int(os.environ.get("WDBX_PERSONA_TOKEN_EXPIRATION", 86400))

        # Store a cache of valid tokens for quick validation
        self._token_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("PersonaTokenManager initialized")

    def create_token(self, persona_id: str, attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a secure token for a persona.

        Args:
            persona_id: Unique identifier for the persona
            attributes: Additional attributes to include in the token

        Returns:
            Secure token string
        """
        if MONITORING_AVAILABLE:
            with performance_monitor.profile("create_persona_token"):
                return self._create_token_internal(persona_id, attributes)
        else:
            return self._create_token_internal(persona_id, attributes)

    def _create_token_internal(
        self, persona_id: str, attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """Internal method to create a persona token."""
        # Create a token payload
        now = int(time.time())
        token_id = str(uuid.uuid4())

        payload = {
            "pid": persona_id,
            "jti": token_id,
            "iat": now,
            "exp": now + self.token_expiration,
        }

        # Add additional attributes if provided
        if attributes:
            # Ensure we don't override reserved fields
            for key, value in attributes.items():
                if key not in payload:
                    payload[key] = value

        # Convert payload to JSON and encode
        payload_json = json.dumps(payload)
        payload_bytes = payload_json.encode("utf-8")

        # Sign the payload with HMAC-SHA256
        signature = self._sign_payload(payload_bytes)

        # Encode payload as base64
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")

        # Combine into a token
        token = f"{payload_b64}.{signature}"

        # Cache the token for quick validation
        self._token_cache[token_id] = payload

        return token

    def _sign_payload(self, payload: bytes) -> str:
        """
        Sign a payload using HMAC-SHA256.

        Args:
            payload: Bytes to sign

        Returns:
            Base64-encoded signature
        """
        h = hashlib.hmac(key=self.secret_key.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
        return base64.urlsafe_b64encode(h.digest()).decode("utf-8").rstrip("=")

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a persona token.

        Args:
            token: Token to validate

        Returns:
            Payload dictionary if valid, None otherwise
        """
        if MONITORING_AVAILABLE:
            with performance_monitor.profile("validate_persona_token"):
                return self._validate_token_internal(token)
        else:
            return self._validate_token_internal(token)

    def _validate_token_internal(self, token: str) -> Optional[Dict[str, Any]]:
        """Internal method to validate a persona token."""
        try:
            # Split token into parts
            if "." not in token:
                logger.warning("Invalid token format")
                return None

            payload_b64, signature = token.split(".")

            # Add padding if needed
            padding = "=" * (4 - (len(payload_b64) % 4))
            payload_b64_padded = payload_b64 + padding

            # Decode payload
            payload_bytes = base64.urlsafe_b64decode(payload_b64_padded)

            # Verify signature
            expected_signature = self._sign_payload(payload_bytes)
            if signature != expected_signature:
                logger.warning("Invalid token signature")
                return None

            # Parse payload
            payload = json.loads(payload_bytes.decode("utf-8"))

            # Check expiration
            now = int(time.time())
            if payload.get("exp", 0) < now:
                logger.warning("Token expired")
                return None

            # Check token ID in cache for revoked tokens
            token_id = payload.get("jti")
            if token_id and self._is_token_revoked(token_id):
                logger.warning(f"Token {token_id} has been revoked")
                return None

            return payload
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None

    def _is_token_revoked(self, token_id: str) -> bool:
        """
        Check if a token has been revoked.

        Args:
            token_id: Token ID to check

        Returns:
            True if revoked, False otherwise
        """
        # This would normally check against a database or cache of revoked tokens
        # For now, it's a placeholder
        return False

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token so it can no longer be used.

        Args:
            token: Token to revoke

        Returns:
            True if successfully revoked, False otherwise
        """
        payload = self.validate_token(token)
        if not payload:
            return False

        token_id = payload.get("jti")
        if not token_id:
            return False

        # Remove from cache
        if token_id in self._token_cache:
            del self._token_cache[token_id]

        # In a real implementation, add to a revocation list/database
        logger.info(f"Revoked token {token_id}")
        return True

    def get_persona_id(self, token: str) -> Optional[str]:
        """
        Extract persona ID from a token.

        Args:
            token: Token to extract from

        Returns:
            Persona ID if valid, None otherwise
        """
        payload = self.validate_token(token)
        if not payload:
            return None

        return payload.get("pid")


class PersonaAttributes:
    """Attributes that can be configured for a persona."""

    FORMALITY = "formality"  # Formal vs. casual tone
    VERBOSITY = "verbosity"  # Concise vs. detailed responses
    CREATIVITY = "creativity"  # Conservative vs. creative in responses
    EMPATHY = "empathy"  # Level of emotional understanding shown
    ASSERTIVENESS = "assertiveness"  # Passive vs. assertive communication
    OPTIMISM = "optimism"  # Pessimistic vs. optimistic outlook
    HUMOR = "humor"  # Serious vs. humorous tone


class PersonaManager:
    """
    Manage personas for WDBX interactions.

    This class handles the creation, management, and application of personas
    with different characteristics, ensuring consistent interactions.
    """

    def __init__(
        self,
        wdbx_instance=None,
        default_safety_level: ContentSafetyLevel = ContentSafetyLevel.MEDIUM,
        enable_ml: bool = True,
    ):
        """
        Initialize the persona manager.

        Args:
            wdbx_instance: The WDBX instance that owns this manager
            default_safety_level: Default safety level for content filtering
            enable_ml: Whether to use ML for enhanced persona management
        """
        self.wdbx = wdbx_instance
        self.token_manager = PersonaTokenManager()
        self.content_filter = ContentFilter(safety_level=default_safety_level)
        self.personas: Dict[str, Dict[str, Any]] = {}
        self.default_persona_id = "default"
        self.enable_ml = enable_ml and ML_AVAILABLE

        # Create a default persona
        self._create_default_persona()

        # Embeddings cache for ML-based persona matching
        self._embedding_cache: Dict[str, VectorLike] = {}

        logger.info("PersonaManager initialized")

    def _create_default_persona(self) -> None:
        """Create a default persona with balanced attributes."""
        self.personas[self.default_persona_id] = {
            "name": "Default",
            "description": "A balanced, helpful assistant persona",
            "attributes": {
                PersonaAttributes.FORMALITY: 0.5,  # Middle of formal vs casual
                PersonaAttributes.VERBOSITY: 0.5,  # Balanced verbosity
                PersonaAttributes.CREATIVITY: 0.5,  # Balanced creativity
                PersonaAttributes.EMPATHY: 0.7,  # Slightly more empathetic
                PersonaAttributes.ASSERTIVENESS: 0.5,  # Balanced assertiveness
                PersonaAttributes.OPTIMISM: 0.6,  # Slightly optimistic
                PersonaAttributes.HUMOR: 0.3,  # Slightly serious
            },
            "safety_level": ContentSafetyLevel.MEDIUM,
            "created_at": int(time.time()),
            "modified_at": int(time.time()),
        }

    def create_persona(
        self,
        name: str,
        description: str,
        attributes: Dict[str, float],
        safety_level: ContentSafetyLevel = ContentSafetyLevel.MEDIUM,
    ) -> str:
        """
        Create a new persona with the given attributes.

        Args:
            name: Human-readable name for the persona
            description: Description of the persona
            attributes: Dictionary of persona attributes (values 0.0-1.0)
            safety_level: Content safety level for this persona

        Returns:
            Persona ID
        """
        # Generate a unique ID
        persona_id = str(uuid.uuid4())

        # Validate and normalize attributes
        normalized_attributes = {}
        for attr, value in attributes.items():
            if not hasattr(PersonaAttributes, attr.upper()):
                logger.warning(f"Unknown persona attribute: {attr}")
                continue
            # Ensure value is in range 0.0-1.0
            normalized_attributes[attr] = max(0.0, min(1.0, float(value)))

        # Create persona
        self.personas[persona_id] = {
            "name": name,
            "description": description,
            "attributes": normalized_attributes,
            "safety_level": safety_level,
            "created_at": int(time.time()),
            "modified_at": int(time.time()),
        }

        # Generate embedding if ML is enabled
        if self.enable_ml:
            self._generate_persona_embedding(persona_id)

        logger.info(f"Created persona: {name} (ID: {persona_id})")
        return persona_id

    def _generate_persona_embedding(self, persona_id: str) -> None:
        """
        Generate an embedding vector for a persona.

        Args:
            persona_id: ID of the persona to generate embedding for
        """
        if not self.enable_ml:
            return

        try:
            persona = self.personas.get(persona_id)
            if not persona:
                return

            # Combine persona attributes into a text representation
            text = f"{persona['name']}. {persona['description']}. "
            for attr, value in persona["attributes"].items():
                # Convert numeric value to descriptive text
                level = (
                    "very low"
                    if value < 0.2
                    else (
                        "low"
                        if value < 0.4
                        else "moderate" if value < 0.6 else "high" if value < 0.8 else "very high"
                    )
                )
                text += f"{attr}: {level}. "

            # Generate embedding using ML backend
            if ML_AVAILABLE:
                embedding = ml_backend.get_embedding(text)
                self._embedding_cache[persona_id] = embedding
        except Exception as e:
            logger.error(f"Error generating persona embedding: {e}")

    def get_persona(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a persona by ID.

        Args:
            persona_id: Persona ID

        Returns:
            Persona dictionary if found, None otherwise
        """
        return self.personas.get(persona_id)

    def get_personas(self) -> List[Dict[str, Any]]:
        """
        Get all personas.

        Returns:
            List of persona dictionaries with IDs
        """
        return [{"id": pid, **persona} for pid, persona in self.personas.items()]

    def update_persona(
        self,
        persona_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        attributes: Optional[Dict[str, float]] = None,
        safety_level: Optional[ContentSafetyLevel] = None,
    ) -> bool:
        """
        Update a persona.

        Args:
            persona_id: ID of persona to update
            name: New name (or None to keep current)
            description: New description (or None to keep current)
            attributes: New attributes (or None to keep current)
            safety_level: New safety level (or None to keep current)

        Returns:
            True if updated successfully, False otherwise
        """
        persona = self.personas.get(persona_id)
        if not persona:
            logger.warning(f"Persona not found: {persona_id}")
            return False

        # Update fields if provided
        if name is not None:
            persona["name"] = name

        if description is not None:
            persona["description"] = description

        if attributes is not None:
            # Validate and normalize attributes
            for attr, value in attributes.items():
                if not hasattr(PersonaAttributes, attr.upper()):
                    logger.warning(f"Unknown persona attribute: {attr}")
                    continue
                # Ensure value is in range 0.0-1.0
                persona["attributes"][attr] = max(0.0, min(1.0, float(value)))

        if safety_level is not None:
            persona["safety_level"] = safety_level

        # Update modification timestamp
        persona["modified_at"] = int(time.time())

        # Regenerate embedding if ML is enabled
        if self.enable_ml:
            self._generate_persona_embedding(persona_id)

        logger.info(f"Updated persona: {persona['name']} (ID: {persona_id})")
        return True

    def delete_persona(self, persona_id: str) -> bool:
        """
        Delete a persona.

        Args:
            persona_id: ID of persona to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        if persona_id == self.default_persona_id:
            logger.warning("Cannot delete the default persona")
            return False

        if persona_id in self.personas:
            del self.personas[persona_id]

            # Remove from embedding cache if exists
            if persona_id in self._embedding_cache:
                del self._embedding_cache[persona_id]

            logger.info(f"Deleted persona: {persona_id}")
            return True
        else:
            logger.warning(f"Persona not found: {persona_id}")
            return False

    def create_token(
        self, persona_id: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a token for a persona.

        Args:
            persona_id: ID of persona to create token for
            attributes: Additional attributes to include in token

        Returns:
            Token string if successful, None otherwise
        """
        if persona_id not in self.personas:
            logger.warning(f"Persona not found: {persona_id}")
            return None

        return self.token_manager.create_token(persona_id, attributes)

    def apply_persona(self, token: str, content: str, safe_mode: bool = True) -> Dict[str, Any]:
        """
        Apply persona characteristics to content.

        Args:
            token: Persona token
            content: Content to process
            safe_mode: Whether to apply content filtering

        Returns:
            Dictionary with processed content and metadata
        """
        if MONITORING_AVAILABLE:
            with performance_monitor.profile("apply_persona"):
                return self._apply_persona_internal(token, content, safe_mode)
        else:
            return self._apply_persona_internal(token, content, safe_mode)

    def _apply_persona_internal(
        self, token: str, content: str, safe_mode: bool = True
    ) -> Dict[str, Any]:
        """Internal method to apply persona to content."""
        # Validate token
        payload = self.token_manager.validate_token(token)
        if not payload:
            logger.warning("Invalid persona token")
            return {
                "error": "Invalid token",
                "content": content,
                "modified": False,
                "persona_id": None,
            }

        # Get persona ID from token
        persona_id = payload.get("pid")
        if not persona_id or persona_id not in self.personas:
            logger.warning(f"Persona not found: {persona_id}")
            return {
                "error": "Persona not found",
                "content": content,
                "modified": False,
                "persona_id": persona_id,
            }

        # Get persona
        persona = self.personas[persona_id]

        # Apply content filtering if in safe mode
        if safe_mode:
            safety_level = persona.get("safety_level", ContentSafetyLevel.MEDIUM)
            filtered_content = self.content_filter.filter_content(
                content, safety_level=safety_level
            )
        else:
            filtered_content = content

        # Get persona attributes
        attributes = persona.get("attributes", {})

        # Apply persona characteristics to content
        modified_content = self._modify_content_with_persona(filtered_content, attributes)

        # Check safety again if modified
        if safe_mode and modified_content != filtered_content:
            safety_level = persona.get("safety_level", ContentSafetyLevel.MEDIUM)
            modified_content = self.content_filter.filter_content(
                modified_content, safety_level=safety_level
            )

        return {
            "content": modified_content,
            "modified": modified_content != content,
            "persona_id": persona_id,
            "persona_name": persona.get("name"),
            "attributes": attributes,
        }

    def _modify_content_with_persona(self, content: str, attributes: Dict[str, float]) -> str:
        """
        Modify content based on persona attributes.

        Args:
            content: Content to modify
            attributes: Persona attributes

        Returns:
            Modified content
        """
        # This is a mock implementation
        # In a real system, this would use more sophisticated NLP and ML techniques

        if self.enable_ml and ML_AVAILABLE:
            try:
                # Use ML backend for more sophisticated content modification
                return self._modify_content_with_ml(content, attributes)
            except Exception as e:
                logger.error(f"Error using ML for content modification: {e}")
                # Fall back to rule-based approach

        # Rule-based approach (simplified)
        modified = content

        # Apply formality
        formality = attributes.get(PersonaAttributes.FORMALITY, 0.5)
        if formality > 0.7:  # More formal
            modified = re.sub(r"\bdon\'t\b", "do not", modified)
            modified = re.sub(r"\bcan\'t\b", "cannot", modified)
            modified = re.sub(r"\bwon\'t\b", "will not", modified)
            modified = re.sub(r"\bI\'m\b", "I am", modified)
        elif formality < 0.3:  # Less formal
            modified = re.sub(r"\bdo not\b", "don't", modified)
            modified = re.sub(r"\bcannot\b", "can't", modified)
            modified = re.sub(r"\bwill not\b", "won't", modified)
            modified = re.sub(r"\bI am\b", "I'm", modified)

        # Apply verbosity
        verbosity = attributes.get(PersonaAttributes.VERBOSITY, 0.5)
        if verbosity < 0.3:  # More concise
            # Remove filler words
            modified = re.sub(r"\b(just|actually|basically|essentially)\b", "", modified)

        # Apply optimism
        optimism = attributes.get(PersonaAttributes.OPTIMISM, 0.5)
        if optimism > 0.7:  # More optimistic
            modified = re.sub(r"\b(unfortunately|sadly|regrettably)\b", "interestingly", modified)
        elif optimism < 0.3:  # Less optimistic
            modified = re.sub(r"\b(fortunately|luckily|happily)\b", "as it happens", modified)

        return modified

    def _modify_content_with_ml(self, content: str, attributes: Dict[str, float]) -> str:
        """
        Use ML to modify content based on persona attributes.

        Args:
            content: Content to modify
            attributes: Persona attributes

        Returns:
            Modified content
        """
        # This is a placeholder for actual ML-based content modification
        # In a real implementation, this would use sophisticated NLP techniques
        return content

    def find_similar_persona(self, text_description: str, threshold: float = 0.7) -> Optional[str]:
        """
        Find a persona similar to the given description using ML embeddings.

        Args:
            text_description: Text describing desired persona
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            ID of most similar persona if above threshold, None otherwise
        """
        if not self.enable_ml or not ML_AVAILABLE:
            logger.warning("ML capabilities required for persona similarity matching")
            return None

        try:
            # Generate embedding for the description
            query_embedding = ml_backend.get_embedding(text_description)

            best_match = None
            best_score = 0.0

            # Compare with all persona embeddings
            for persona_id, persona_embedding in self._embedding_cache.items():
                similarity = ml_backend.cosine_similarity(query_embedding, persona_embedding)
                if similarity > best_score:
                    best_score = similarity
                    best_match = persona_id

            # Return match if above threshold
            if best_match and best_score >= threshold:
                return best_match
            return None
        except Exception as e:
            logger.error(f"Error finding similar persona: {e}")
            return None


# Detector for bias in content
class BiasDetector:
    """
    Detect various types of bias in content.

    This class uses ML models when available to detect cognitive biases,
    social biases, and other forms of unfairness in content.
    """

    def __init__(self, enable_ml: bool = True):
        """
        Initialize the bias detector.

        Args:
            enable_ml: Whether to use ML for enhanced bias detection
        """
        self.enable_ml = enable_ml and ML_AVAILABLE

        logger.info("BiasDetector initialized")
        if not self.enable_ml:
            logger.warning("ML capabilities not available for advanced bias detection")

    def detect_bias(self, content: str) -> Dict[str, Any]:
        """
        Detect bias in content.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with bias analysis results
        """
        if MONITORING_AVAILABLE:
            with performance_monitor.profile("detect_bias"):
                return self._detect_bias_internal(content)
        else:
            return self._detect_bias_internal(content)

    def _detect_bias_internal(self, content: str) -> Dict[str, Any]:
        """Internal method to detect bias in content."""
        # Use ML-based detection if available
        if self.enable_ml and ML_AVAILABLE:
            try:
                return self._detect_bias_with_ml(content)
            except Exception as e:
                logger.error(f"Error using ML for bias detection: {e}")
                # Fall back to rule-based approach

        # Simple rule-based approach as fallback
        results = {}

        # Check for gender bias (very simplified)
        gender_patterns = [
            (r"\b(he|him|his)\b", "male"),
            (r"\b(she|her|hers)\b", "female"),
            (r"\b(they|them|theirs)\b", "neutral"),
        ]

        gender_counts = {}
        for pattern, gender in gender_patterns:
            count = len(re.findall(pattern, content, re.IGNORECASE))
            gender_counts[gender] = count

        total = sum(gender_counts.values())
        if total > 0:
            # Calculate proportions
            proportions = {gender: count / total for gender, count in gender_counts.items()}

            # Check for significant imbalance
            results["gender_bias"] = {
                "detected": max(proportions.values()) > 0.7,  # Simple threshold
                "proportions": proportions,
                "confidence": 0.5,  # Low confidence for rule-based
            }
        else:
            results["gender_bias"] = {
                "detected": False,
                "proportions": {},
                "confidence": 0.5,
            }

        # Add placeholder results for other bias types
        for bias_type in ["racial_bias", "age_bias", "political_bias", "confirmation_bias"]:
            results[bias_type] = {
                "detected": False,
                "confidence": 0.0,
                "notes": "Analysis not available without ML capabilities",
            }

        return results

    def _detect_bias_with_ml(self, content: str) -> Dict[str, Any]:
        """
        Use ML to detect bias in content.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with bias analysis results
        """
        # This is a placeholder for actual ML-based bias detection
        # In a real implementation, this would use bias detection models

        # Mock results - would come from ML models in a real implementation
        return {
            "gender_bias": {
                "detected": False,
                "confidence": 0.8,
                "score": 0.2,  # Lower is better
            },
            "racial_bias": {
                "detected": False,
                "confidence": 0.7,
                "score": 0.1,
            },
            "age_bias": {
                "detected": False,
                "confidence": 0.6,
                "score": 0.15,
            },
            "political_bias": {
                "detected": False,
                "confidence": 0.7,
                "score": 0.3,
            },
            "confirmation_bias": {
                "detected": False,
                "confidence": 0.5,
                "score": 0.25,
            },
        }
