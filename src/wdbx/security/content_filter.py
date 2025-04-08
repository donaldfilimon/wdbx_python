# wdbx/content_filter.py
import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Set up logger
logger = logging.getLogger("wdbx.security.content_filter")

# Try to import ML components
try:
    from ..ml import VectorLike
    from ..ml.backend import MLBackend

    ML_AVAILABLE = True
    ml_backend = MLBackend()
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML components not available for advanced content filtering")

# Try to import monitoring
try:
    from ..monitoring.performance import PerformanceMonitor

    MONITORING_AVAILABLE = True
    performance_monitor = PerformanceMonitor()
except ImportError:
    MONITORING_AVAILABLE = False


class ContentSafetyLevel(Enum):
    """Safety levels for content filtering."""

    HIGH = 3  # Strict filtering, suitable for all audiences
    MEDIUM = 2  # Moderate filtering, may allow mild content
    LOW = 1  # Minimal filtering, allows most content except harmful
    NONE = 0  # No filtering, raw content


class ContentTopic(Enum):
    """Topics that may be filtered based on sensitivity."""

    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL = "sexual"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    PERSONAL_INFO = "personal_info"
    POLITICAL = "political"
    RELIGIOUS = "religious"
    PROFANITY = "profanity"
    WEAPONS = "weapons"
    GAMBLING = "gambling"
    DRUGS = "drugs"
    ALCOHOL = "alcohol"
    TOBACCO = "tobacco"


class ContentFilter:
    """
    Filter content for safety by detecting sensitive topics and offensive language.

    This class provides methods to filter content, check safety levels, and
    measure bias. It uses pattern matching combined with ML classification when
    available for more accurate results.
    """

    def __init__(
        self,
        safety_level: ContentSafetyLevel = ContentSafetyLevel.MEDIUM,
        enable_ml: bool = True,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        blocked_topics: Optional[List[ContentTopic]] = None,
    ):
        """
        Initialize the content filter with specified safety level and patterns.

        Args:
            safety_level: The default safety level to apply
            enable_ml: Whether to use ML-based filtering if available
            custom_patterns: Dictionary of custom regex patterns by category
            blocked_topics: List of topics to always block regardless of safety level
        """
        # If safety_level is not a ContentSafetyLevel enum, use MEDIUM as default
        if not isinstance(safety_level, ContentSafetyLevel):
            logger.warning(
                f"Invalid safety_level type: {type(safety_level).__name__}, using MEDIUM as default"
            )
            safety_level = ContentSafetyLevel.MEDIUM

        self.safety_level = safety_level
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.custom_patterns = custom_patterns or {}
        self.blocked_topics = set(blocked_topics or [])

        # Load built-in patterns
        self.patterns = self._load_default_patterns()

        # Add custom patterns if provided
        if custom_patterns:
            for category, patterns in custom_patterns.items():
                if category in self.patterns:
                    self.patterns[category].extend(patterns)
                else:
                    self.patterns[category] = patterns

        # Compile regex patterns for efficiency
        self.compiled_patterns = self._compile_patterns()

        # Cache for already analyzed content to improve performance
        self._content_cache = {}
        self._cache_size_limit = 1000  # Limit cache size to prevent memory issues

        logger.info(f"ContentFilter initialized with safety level: {self.safety_level.name}")
        if self.enable_ml:
            logger.info("ML-based content filtering enabled")

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        """
        Load default patterns for each content topic.

        Returns:
            Dictionary mapping topic names to lists of regex patterns
        """
        # Default patterns path - can be overridden with env var
        patterns_path = os.environ.get(
            "WDBX_PATTERNS_PATH", os.path.join(os.path.dirname(__file__), "patterns.json")
        )

        # Start with basic patterns
        default_patterns = {
            "profanity": [
                r"\b(f+[^\w]*u+[^\w]*c+[^\w]*k+)\b",
                r"\b(s+[^\w]*h+[^\w]*i+[^\w]*t+)\b",
                r"\b(a+[^\w]*s+[^\w]*s+[^\w]*h+[^\w]*o+[^\w]*l+[^\w]*e+)\b",
                r"\b(b+[^\w]*i+[^\w]*t+[^\w]*c+[^\w]*h+)\b",
                r"\b(d+[^\w]*a+[^\w]*m+[^\w]*n+)\b",
            ],
            "violence": [
                r"\b(k+[^\w]*i+[^\w]*l+[^\w]*l+)\b",
                r"\b(m+[^\w]*u+[^\w]*r+[^\w]*d+[^\w]*e+[^\w]*r+)\b",
                r"\b(s+[^\w]*l+[^\w]*a+[^\w]*u+[^\w]*g+[^\w]*h+[^\w]*t+[^\w]*e+[^\w]*r+)\b",
                r"\b(a+[^\w]*s+[^\w]*s+[^\w]*a+[^\w]*u+[^\w]*l+[^\w]*t+)\b",
                r"\b(b+[^\w]*e+[^\w]*a+[^\w]*t+[^\w]*i+[^\w]*n+[^\w]*g+)\b",
            ],
            "hate_speech": [
                r"\b(n+[^\w]*i+[^\w]*g+[^\w]*g+[^\w]*[ae]+[^\w]*r+)\b",
                r"\b(f+[^\w]*a+[^\w]*g+[^\w]*g*[^\w]*o+[^\w]*t+)\b",
                r"\b(k+[^\w]*i+[^\w]*k+[^\w]*e+)\b",
                r"\b(s+[^\w]*p+[^\w]*i+[^\w]*c+)\b",
                r"\b(w+[^\w]*e+[^\w]*t+[^\w]*b+[^\w]*a+[^\w]*c+[^\w]*k+)\b",
                r"\b(g+[^\w]*o+[^\w]*o+[^\w]*k+)\b",
            ],
            "sexual": [
                r"\b(p+[^\w]*o+[^\w]*r+[^\w]*n+)\b",
                r"\b(s+[^\w]*e+[^\w]*x+)\b",
                r"\b(a+[^\w]*n+[^\w]*a+[^\w]*l+)\b",
                r"\b(v+[^\w]*a+[^\w]*g+[^\w]*i+[^\w]*n+[^\w]*a+)\b",
                r"\b(p+[^\w]*e+[^\w]*n+[^\w]*i+[^\w]*s+)\b",
                r"\b(d+[^\w]*i+[^\w]*c+[^\w]*k+)\b",
                r"\b(c+[^\w]*o+[^\w]*c+[^\w]*k+)\b",
            ],
            "personal_info": [
                r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
                r"\b\d{16}\b",  # Credit card (basic)
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",  # Phone
            ],
        }

        # Try to load patterns from file
        try:
            if os.path.exists(patterns_path):
                with open(patterns_path) as f:
                    file_patterns = json.load(f)
                    # Merge with default patterns
                    for category, patterns in file_patterns.items():
                        if category in default_patterns:
                            default_patterns[category].extend(patterns)
                        else:
                            default_patterns[category] = patterns
                    logger.info(f"Loaded patterns from {patterns_path}")
        except Exception as e:
            logger.warning(f"Could not load patterns from {patterns_path}: {e}")

        return default_patterns

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Compile all regex patterns for efficient matching.

        Returns:
            Dictionary mapping categories to lists of compiled regex patterns
        """
        compiled = {}
        for category, patterns in self.patterns.items():
            compiled[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled

    def filter_content(
        self,
        content: Union[str, Dict[str, Any]],
        safety_level: Optional[ContentSafetyLevel] = None,
        content_field: str = "content",
    ) -> Union[str, Dict[str, Any]]:
        """
        Filter content according to the specified safety level.

        Args:
            content: String or dictionary containing content to filter
            safety_level: Override the default safety level
            content_field: Field name to filter if content is a dictionary

        Returns:
            Filtered content with potentially harmful content removed or redacted
        """
        if MONITORING_AVAILABLE:
            with performance_monitor.profile("content_filter"):
                return self._perform_content_filtering(content, safety_level, content_field)
        else:
            return self._perform_content_filtering(content, safety_level, content_field)

    def _perform_content_filtering(
        self,
        content: Union[str, Dict[str, Any]],
        safety_level: Optional[ContentSafetyLevel] = None,
        content_field: str = "content",
    ) -> Union[str, Dict[str, Any]]:
        """Internal method to perform the actual content filtering."""
        level = safety_level or self.safety_level

        # If level is NONE, return content unchanged
        if level == ContentSafetyLevel.NONE:
            return content

        # Process dictionary content
        if isinstance(content, dict):
            if content_field in content and isinstance(content[content_field], str):
                filtered_content = self._filter_text(content[content_field], level)
                result = content.copy()
                result[content_field] = filtered_content
                return result
            else:
                # Process all string fields in the dictionary
                result = {}
                for key, value in content.items():
                    if isinstance(value, str):
                        result[key] = self._filter_text(value, level)
                    elif isinstance(value, dict):
                        result[key] = self.filter_content(value, level)
                    elif isinstance(value, list):
                        result[key] = [
                            (
                                self.filter_content(item, level)
                                if isinstance(item, (dict, str))
                                else item
                            )
                            for item in value
                        ]
                    else:
                        result[key] = value
                return result
        # Process string content
        elif isinstance(content, str):
            return self._filter_text(content, level)
        else:
            # Return unchanged if type is unsupported
            logger.warning(f"Unsupported content type for filtering: {type(content)}")
            return content

    def _filter_text(self, text: str, level: ContentSafetyLevel) -> str:
        """
        Filter text content based on safety level.

        Args:
            text: Text content to filter
            level: Safety level to apply

        Returns:
            Filtered text with potentially harmful content redacted
        """
        # Check cache first
        cache_key = (text, level.value)
        if cache_key in self._content_cache:
            return self._content_cache[cache_key]

        # Classify content with ML if available and enabled
        ml_classifications = {}
        if self.enable_ml and ML_AVAILABLE:
            try:
                ml_classifications = self._classify_with_ml(text)
            except Exception as e:
                logger.error(f"ML classification error: {e}")

        # Apply regex pattern matching
        filtered = text
        categories_to_check = self._get_categories_for_level(level)

        for category in categories_to_check:
            if category in self.compiled_patterns:
                for pattern in self.compiled_patterns[category]:
                    # Combine ML confidence with regex
                    ml_confidence = ml_classifications.get(category, 0.0)

                    # Adjust threshold based on ML confidence
                    if ml_confidence > 0.8:
                        # High confidence from ML, aggressively redact
                        filtered = pattern.sub("[REDACTED]", filtered)
                    elif ml_confidence > 0.5:
                        # Medium confidence, look for matches
                        filtered = pattern.sub("[REDACTED]", filtered)
                    else:
                        # Low/no ML confidence, only redact clear matches
                        filtered = pattern.sub("[REDACTED]", filtered)

        # Store in cache (with size limit)
        if len(self._content_cache) >= self._cache_size_limit:
            # Clear half the cache when limit is reached
            keys_to_remove = list(self._content_cache.keys())[: self._cache_size_limit // 2]
            for key in keys_to_remove:
                del self._content_cache[key]

        self._content_cache[cache_key] = filtered
        return filtered

    def _classify_with_ml(self, text: str) -> Dict[str, float]:
        """
        Use ML models to classify content into categories.

        Args:
            text: Text to classify

        Returns:
            Dictionary mapping category names to confidence scores (0.0-1.0)
        """
        # This is a simplified placeholder for ML classification
        # In a real implementation, this would call actual ML models

        if not ML_AVAILABLE:
            return {}

        # Mock implementation - in a real system, this would use
        # embedding models and classifiers
        results = {}

        # Check if text is long enough to classify
        if len(text) < 10:
            return results

        # Simple keyword-based classification for demonstration
        # In a real system, this would use proper ML models
        keywords = {
            "profanity": ["fuck", "shit", "damn", "ass", "bitch"],
            "violence": ["kill", "murder", "attack", "hurt", "weapon"],
            "hate_speech": ["racist", "bigot", "hate", "slur"],
            "sexual": ["sex", "porn", "nude", "explicit"],
        }

        for category, words in keywords.items():
            text_lower = text.lower()
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                # Calculate confidence based on keyword matches
                confidence = min(1.0, count / len(words))
                results[category] = confidence

        return results

    def _get_categories_for_level(self, level: ContentSafetyLevel) -> List[str]:
        """
        Get content categories to check based on safety level.

        Args:
            level: Content safety level

        Returns:
            List of category names to check
        """
        # Base categories that should always be checked
        base_categories = [topic.value for topic in self.blocked_topics]

        # Add categories based on safety level
        if level == ContentSafetyLevel.HIGH:
            # Most strict - check all categories
            return list(self.patterns.keys())
        elif level == ContentSafetyLevel.MEDIUM:
            # Medium - check all except mild categories
            categories = [
                "violence",
                "hate_speech",
                "sexual",
                "harassment",
                "self_harm",
                "illegal_activity",
                "personal_info",
                "profanity",
            ]
        elif level == ContentSafetyLevel.LOW:
            # Low - check only the most harmful categories
            categories = [
                "violence",
                "hate_speech",
                "self_harm",
                "illegal_activity",
                "personal_info",
            ]
        else:
            # NONE - check only explicitly blocked topics
            return base_categories

        # Add any explicitly blocked topics
        for category in base_categories:
            if category not in categories:
                categories.append(category)

        # Return only categories that we have patterns for
        return [c for c in categories if c in self.patterns]

    def check_safety(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check content for safety concerns without filtering.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with safety analysis results
        """
        if MONITORING_AVAILABLE:
            with performance_monitor.profile("content_safety_check"):
                return self._perform_safety_check(content)
        else:
            return self._perform_safety_check(content)

    def _perform_safety_check(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Internal method to perform content safety analysis."""
        # Extract text content from string or dictionary
        if isinstance(content, dict):
            if "content" in content and isinstance(content["content"], str):
                text = content["content"]
            else:
                # Concatenate all string values
                text = " ".join(
                    [str(value) for value in content.values() if isinstance(value, str)]
                )
        elif isinstance(content, str):
            text = content
        else:
            logger.warning(f"Unsupported content type for safety check: {type(content)}")
            return {"safe": True, "categories": {}}

        # Get ML classifications if available
        ml_results = {}
        if self.enable_ml and ML_AVAILABLE:
            try:
                ml_results = self._classify_with_ml(text)
            except Exception as e:
                logger.error(f"ML classification error in safety check: {e}")

        # Check each category for matches
        category_results = {}
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    matches.extend(found)

            # Combine regex and ML results
            ml_score = ml_results.get(category, 0.0)
            regex_score = min(1.0, len(matches) / 10)  # Normalize to 0-1

            # Calculate combined score with ML given more weight if available
            if ml_score > 0:
                combined_score = (ml_score * 0.7) + (regex_score * 0.3)
            else:
                combined_score = regex_score

            category_results[category] = {
                "score": combined_score,
                "matches": matches[:5],  # Limit to first 5 matches
                "match_count": len(matches),
            }

        # Determine overall safety
        unsafe_categories = {
            category: details
            for category, details in category_results.items()
            if details["score"] > 0.5  # Threshold for marking as unsafe
        }

        blocked_categories = [topic.value for topic in self.blocked_topics]
        has_blocked_content = any(category in blocked_categories for category in unsafe_categories)

        return {
            "safe": len(unsafe_categories) == 0,
            "blocked": has_blocked_content,
            "categories": category_results,
            "overall_score": max(
                [details["score"] for details in category_results.values()], default=0.0
            ),
        }

    def measure_bias(self, content: Union[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Measure potential bias in content across various dimensions.

        Args:
            content: Content to analyze

        Returns:
            Dictionary mapping bias dimensions to scores (-1.0 to 1.0)
        """
        # This requires ML capabilities
        if not self.enable_ml or not ML_AVAILABLE:
            logger.warning("ML capabilities required for bias measurement")
            return {}

        # Extract text content
        if isinstance(content, dict):
            if "content" in content and isinstance(content["content"], str):
                text = content["content"]
            else:
                text = " ".join(
                    [str(value) for value in content.values() if isinstance(value, str)]
                )
        elif isinstance(content, str):
            text = content
        else:
            logger.warning(f"Unsupported content type for bias measurement: {type(content)}")
            return {}

        # This is a placeholder for actual ML-based bias detection
        # In a real implementation, this would use embeddings and bias detection models

        # Mock implementation - in a real system this would use proper ML models
        bias_dimensions = [
            "gender",
            "race",
            "religion",
            "age",
            "disability",
            "socioeconomic",
            "political",
        ]

        results = {}
        for dimension in bias_dimensions:
            # Generate a mock score between -1.0 and 1.0
            # In a real system, this would come from ML models
            results[dimension] = 0.0

        return results

    def update_patterns(self, patterns: Dict[str, List[str]]) -> None:
        """
        Update regex patterns used for content filtering.

        Args:
            patterns: Dictionary mapping categories to lists of regex patterns
        """
        # Update patterns dictionary
        for category, new_patterns in patterns.items():
            if category in self.patterns:
                self.patterns[category].extend(new_patterns)
            else:
                self.patterns[category] = new_patterns

        # Recompile patterns
        self.compiled_patterns = self._compile_patterns()

        # Clear cache since patterns have changed
        self._content_cache.clear()

        logger.info(f"Updated content filter patterns: {len(patterns)} categories")


class BiasDetector:
    """
    Detects and mitigates bias in content based on specified attributes.
    """

    def __init__(
        self, bias_attributes: List[str], bias_thresholds: Optional[Dict[str, float]] = None
    ) -> None:
        self.bias_attributes = bias_attributes
        self.bias_thresholds = bias_thresholds or dict.fromkeys(bias_attributes, 0.7)
        self.bias_patterns = self._compile_bias_patterns()

    def _compile_bias_patterns(self) -> Dict[str, List[str]]:
        """Create common patterns for each bias attribute to improve detection."""
        patterns = {}
        for attr in self.bias_attributes:
            # Add common patterns for each attribute - this would be expanded in a
            # real implementation
            patterns[attr] = [f"{attr}", f"all {attr}", f"typical {attr}", f"those {attr}"]
        return patterns

    def measure_bias(self, content: Any) -> Dict[str, float]:
        """
        Measures bias in content for each attribute.

        In a real implementation, this would use NLP techniques or ML models.

        Args:
            content: Text content to analyze (string or dictionary)

        Returns:
            Dictionary of bias scores by attribute
        """
        # Initialize empty bias scores
        bias_scores = {}

        # Handle non-string content
        if not isinstance(content, str):
            return bias_scores

        content_lower = content.lower()

        for attr, patterns in self.bias_patterns.items():
            # Count occurrences of patterns
            matches = sum(1 for pattern in patterns if pattern.lower() in content_lower)
            # Calculate normalized score (simple implementation)
            bias_scores[attr] = min(matches * 0.2, 1.0)

            # Add some randomness for demo purposes
            # In a real implementation, this would be a more sophisticated algorithm
            if np.random.random() < 0.3:
                bias_scores[attr] += np.random.random() * 0.3
                bias_scores[attr] = min(bias_scores[attr], 1.0)

        return bias_scores

    def calculate_overall_bias(self, bias_scores: Dict[str, float]) -> float:
        """Calculate overall bias score as weighted average of individual scores."""
        if not bias_scores:
            return 0.0

        # Calculate weighted average - give higher weight to higher biases
        weights = {attr: score + 0.5 for attr, score in bias_scores.items()}
        total_weight = sum(weights.values())

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(score * weights[attr] for attr, score in bias_scores.items())
        return weighted_sum / total_weight

    def mitigate_bias(self, content: str, bias_scores: Dict[str, float]) -> str:
        """
        Add disclaimers or modify content to mitigate detected bias.

        Args:
            content: Original content
            bias_scores: Dictionary of bias scores by attribute

        Returns:
            Modified content with bias mitigation applied
        """
        high_bias_attrs = [
            attr
            for attr, score in bias_scores.items()
            if score > self.bias_thresholds.get(attr, 0.7)
        ]

        if high_bias_attrs:
            bias_list = ", ".join(high_bias_attrs)
            disclaimer = (
                f"Note: I've tried to present a balanced perspective on this topic. "
                f"The content may have some bias related to {bias_list}. "
                f"If you notice any bias, please let me know."
            )
            return content + "\n\n" + disclaimer

        return content

    def detect_bias(self, content: Any) -> Dict[str, float]:
        """
        Analyze content to detect bias related to the configured attributes.

        Args:
            content: Text content to analyze (string or dictionary)

        Returns:
            Dictionary of bias scores by attribute (only includes scores above threshold)
        """
        # Early return for None
        if content is None:
            return {}

        # Handle dictionary inputs
        if isinstance(content, dict):
            # Combine all string values from the dictionary
            combined_text = ""
            for key, value in content.items():
                if isinstance(value, str):
                    combined_text += value + " "
                elif isinstance(value, (list, tuple)):
                    # Handle lists/tuples of strings
                    for item in value:
                        if isinstance(item, str):
                            combined_text += item + " "
            content = combined_text.strip()
            if not content:
                return {}  # Empty result for empty combined text

        # Handle list inputs (e.g., list of strings)
        elif isinstance(content, (list, tuple)):
            combined_text = ""
            for item in content:
                if isinstance(item, str):
                    combined_text += item + " "
            content = combined_text.strip()
            if not content:
                return {}

        # Handle non-string inputs
        if not isinstance(content, str):
            return {}  # Return empty result for non-string content

        # Empty string check
        if not content.strip():
            return {}

        try:
            # Measure bias using existing method
            all_bias_scores = self.measure_bias(content)

            # Filter to only include scores above threshold
            significant_bias = {
                attr: score
                for attr, score in all_bias_scores.items()
                if score > self.bias_thresholds.get(attr, 0.7)
            }

            return significant_bias
        except Exception as e:
            # Log error and return empty dict on exception
            logger.error(f"Error in bias detection: {e}", exc_info=True)
            return {}
