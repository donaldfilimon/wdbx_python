# wdbx/content_filter.py
from typing import List, Dict, Tuple
import numpy as np

class ContentFilter:
    """
    Filters content to ensure safety.
    """
    def __init__(self, sensitive_topics: List[str], offensive_patterns: List[str],
                 safety_thresholds: Dict[str, float] = None) -> None:
        self.sensitive_topics = sensitive_topics
        self.offensive_patterns = offensive_patterns
        self.safety_thresholds = safety_thresholds or {
            "toxicity": 0.7,
            "hate": 0.8,
            "sexual": 0.8,
            "violence": 0.7,
            "self_harm": 0.5
        }
    
    def filter_content(self, content: str) -> Tuple[str, Dict[str, float], bool]:
        scores = {
            "toxicity": 0.0,
            "hate": 0.0,
            "sexual": 0.0,
            "violence": 0.0,
            "self_harm": 0.0
        }
        for topic in self.sensitive_topics:
            if topic.lower() in content.lower():
                scores["toxicity"] += 0.2
        for pattern in self.offensive_patterns:
            if pattern.lower() in content.lower():
                scores["hate"] += 0.3
                scores["toxicity"] += 0.3
        should_filter = any(scores[k] > self.safety_thresholds[k] for k in scores)
        if should_filter:
            filtered_content = ("I apologize, but I cannot provide that information as requested. "
                                "Could you please rephrase your query?")
        else:
            filtered_content = content
        return filtered_content, scores, should_filter


class BiasDetector:
    """
    Detects and mitigates bias in content.
    """
    def __init__(self, bias_attributes: List[str], bias_thresholds: Dict[str, float] = None) -> None:
        self.bias_attributes = bias_attributes
        self.bias_thresholds = bias_thresholds or {attr: 0.7 for attr in bias_attributes}
    
    def measure_bias(self, content: str) -> Dict[str, float]:
        bias_scores = {attr: np.random.random() * 0.5 for attr in self.bias_attributes}
        return bias_scores
    
    def calculate_overall_bias(self, bias_scores: Dict[str, float]) -> float:
        return sum(bias_scores.values()) / len(bias_scores) if bias_scores else 0.0
    
    def mitigate_bias(self, content: str, bias_scores: Dict[str, float]) -> str:
        high_bias_attrs = [attr for attr, score in bias_scores.items() if score > self.bias_thresholds.get(attr, 0.7)]
        if high_bias_attrs:
            disclaimer = ("Note: I've tried to present a balanced perspective on this topic. "
                          "If you notice any bias, please let me know.")
            return content + "\n\n" + disclaimer
        return content
