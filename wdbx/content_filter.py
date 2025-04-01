# wdbx/content_filter.py
from typing import List, Dict, Tuple, Optional
import numpy as np
import re

class ContentFilter:
    """
    Filters content to ensure safety by detecting and addressing sensitive topics,
    offensive language, and harmful content.
    """
    def __init__(self, sensitive_topics: List[str], offensive_patterns: List[str],
                 safety_thresholds: Optional[Dict[str, float]] = None) -> None:
        self.sensitive_topics = sensitive_topics
        self.offensive_patterns = offensive_patterns
        self.safety_thresholds = safety_thresholds or {
            "toxicity": 0.7,
            "hate": 0.8,
            "sexual": 0.8,
            "violence": 0.7,
            "self_harm": 0.5
        }
        # Precompile patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in offensive_patterns]

    def filter_content(self, content: str) -> Tuple[str, Dict[str, float], bool]:
        """
        Filters content based on safety criteria.

        Args:
            content: The text content to filter

        Returns:
            Tuple containing:
                - filtered content (original or redacted)
                - dictionary of safety scores
                - boolean indicating if content was filtered
        """
        scores = {
            "toxicity": 0.0,
            "hate": 0.0,
            "sexual": 0.0,
            "violence": 0.0,
            "self_harm": 0.0
        }

        # Check for sensitive topics
        for topic in self.sensitive_topics:
            if topic.lower() in content.lower():
                topic_lower = topic.lower()
                if any(kw in topic_lower for kw in ["harm", "suicide", "self-harm"]):
                    scores["self_harm"] += 0.3
                elif any(kw in topic_lower for kw in ["sex", "nude", "porn"]):
                    scores["sexual"] += 0.3
                elif any(kw in topic_lower for kw in ["kill", "attack", "weapon"]):
                    scores["violence"] += 0.3
                scores["toxicity"] += 0.2

        # Check for offensive patterns with compiled regex for efficiency
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                scores["hate"] += 0.3
                scores["toxicity"] += 0.3

        # Ensure scores don't exceed 1.0
        for key in scores:
            scores[key] = min(scores[key], 1.0)

        should_filter = any(scores[k] > self.safety_thresholds[k] for k in scores)

        if should_filter:
            # Identify which categories triggered the filter
            triggered_categories = [k for k in scores if scores[k] > self.safety_thresholds[k]]

            filtered_content = (
                "I apologize, but I cannot provide that information as requested. "
                f"The content may contain inappropriate material related to {', '.join(triggered_categories)}. "
                "Could you please rephrase your query?"
            )
        else:
            filtered_content = content

        return filtered_content, scores, should_filter


class BiasDetector:
    """
    Detects and mitigates bias in content based on specified attributes.
    """
    def __init__(self, bias_attributes: List[str], bias_thresholds: Optional[Dict[str, float]] = None) -> None:
        self.bias_attributes = bias_attributes
        self.bias_thresholds = bias_thresholds or {attr: 0.7 for attr in bias_attributes}
        self.bias_patterns = self._compile_bias_patterns()

    def _compile_bias_patterns(self) -> Dict[str, List[str]]:
        """Create common patterns for each bias attribute to improve detection."""
        patterns = {}
        for attr in self.bias_attributes:
            # Add common patterns for each attribute - this would be expanded in a real implementation
            patterns[attr] = [f"{attr}", f"all {attr}", f"typical {attr}", f"those {attr}"]
        return patterns

    def measure_bias(self, content: str) -> Dict[str, float]:
        """
        Measures bias in content for each attribute.

        In a real implementation, this would use NLP techniques or ML models.
        """
        bias_scores = {}
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
        high_bias_attrs = [attr for attr, score in bias_scores.items() 
                          if score > self.bias_thresholds.get(attr, 0.7)]

        if high_bias_attrs:
            bias_list = ", ".join(high_bias_attrs)
            disclaimer = (
                f"Note: I've tried to present a balanced perspective on this topic. "
                f"The content may have some bias related to {bias_list}. "
                f"If you notice any bias, please let me know."
            )
            return content + "\n\n" + disclaimer

        return content
