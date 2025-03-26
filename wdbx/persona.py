# wdbx/persona.py
import uuid
import time
import numpy as np
from typing import Union, Dict, Tuple
from wdbx.data_structures import EmbeddingVector
from wdbx.blockchain import BlockChainManager
from wdbx.mvcc import MVCCManager
from wdbx.constants import logger
from wdbx.vector_store import VectorStore
from wdbx.neural_backtracking import NeuralBacktracker

class PersonaTokenManager:
    """
    Handles persona token injection into input embeddings.
    """
    def __init__(self, persona_embeddings: Dict[str, np.ndarray]) -> None:
        self.persona_embeddings = persona_embeddings
    
    def inject_persona_token(self, persona_id: str, user_input_embedding: np.ndarray) -> np.ndarray:
        if persona_id not in self.persona_embeddings:
            raise ValueError(f"Unknown persona: {persona_id}")
        persona_embedding = self.persona_embeddings[persona_id]
        if persona_embedding.shape != user_input_embedding.shape:
            raise ValueError("Dimension mismatch between persona and user input embeddings.")
        return 0.2 * persona_embedding + 0.8 * user_input_embedding
    
    def create_blended_embedding(self, persona_weights: Dict[str, float], user_input_embedding: np.ndarray) -> np.ndarray:
        total_weight = sum(persona_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            raise ValueError(f"Persona weights must sum to 1.0, got {total_weight}")
        result = np.zeros_like(user_input_embedding)
        for persona_id, weight in persona_weights.items():
            if persona_id not in self.persona_embeddings:
                raise ValueError(f"Unknown persona: {persona_id}")
            result += weight * self.persona_embeddings[persona_id]
        user_weight = 0.8
        return (1 - user_weight) * result + user_weight * user_input_embedding


class PersonaManager:
    """
    Manages multi-persona interactions including response generation.
    """
    def __init__(self, wdbx: "WDBX") -> None:
        self.wdbx = wdbx
        self.persona_embeddings = {
            "Abbey": np.random.randn(self.wdbx.vector_dimension).astype(np.float32),
            "Aviva": np.random.randn(self.wdbx.vector_dimension).astype(np.float32),
            "Abi": np.random.randn(self.wdbx.vector_dimension).astype(np.float32)
        }
        for persona_id, embedding in self.persona_embeddings.items():
            norm = np.linalg.norm(embedding)
            if norm > 0:
                self.persona_embeddings[persona_id] = embedding / norm
        self.persona_token_manager = PersonaTokenManager(self.persona_embeddings)
        from wdbx.content_filter import ContentFilter, BiasDetector
        self.content_filter = ContentFilter(
            sensitive_topics=["illegal activities", "dangerous substances", "explicit content"],
            offensive_patterns=["offensive language", "hate speech", "discriminatory language"]
        )
        self.bias_detector = BiasDetector(
            bias_attributes=["gender", "race", "age", "religion", "nationality"]
        )
        logger.info("PersonaManager initialized with Abbey, Aviva, and Abi.")
    
    def determine_optimal_persona(self, user_input: str, context: Dict[str, any] = None) -> Union[str, Dict[str, float]]:
        technical_keywords = ["how", "what", "explain", "code", "function", "algorithm"]
        emotional_keywords = ["feel", "sad", "happy", "worried", "anxious", "excited"]
        technical_score = sum(1 for kw in technical_keywords if kw.lower() in user_input.lower())
        emotional_score = sum(1 for kw in emotional_keywords if kw.lower() in user_input.lower())
        if technical_score > emotional_score * 2:
            return "Aviva"
        elif emotional_score > technical_score * 2:
            return "Abbey"
        else:
            total = technical_score + emotional_score
            if total == 0:
                return {"Abbey": 0.5, "Aviva": 0.5}
            aviva_ratio = technical_score / total
            abbey_ratio = emotional_score / total
            return {"Abbey": abbey_ratio, "Aviva": aviva_ratio}
    
    def generate_response(self, user_input: str, persona_id: Union[str, Dict[str, float]], 
                          context: Dict[str, any] = None) -> Tuple[str, str]:
        user_input_embedding = np.random.randn(self.wdbx.vector_dimension).astype(np.float32)
        norm = np.linalg.norm(user_input_embedding)
        if norm > 0:
            user_input_embedding = user_input_embedding / norm
        if isinstance(persona_id, str):
            combined_embedding = self.persona_token_manager.inject_persona_token(persona_id, user_input_embedding)
            active_persona = persona_id
        else:
            combined_embedding = self.persona_token_manager.create_blended_embedding(persona_id, user_input_embedding)
            active_persona = "Blended"
        embedding_vector = EmbeddingVector(
            vector=combined_embedding,
            metadata={
                "user_input": user_input,
                "active_persona": active_persona,
                "timestamp": time.time()
            }
        )
        if active_persona == "Abbey":
            response = (f"I understand you're asking about '{user_input}'. "
                        "Let me help you with this in a supportive way...")
        elif active_persona == "Aviva":
            response = (f"Regarding '{user_input}': "
                        "Here is the direct answer without extra fluff...")
        else:
            response = (f"In response to '{user_input}': "
                        "I have balanced factual information with support...")
        filtered_response, safety_scores, was_filtered = self.content_filter.filter_content(response)
        bias_scores = self.bias_detector.measure_bias(filtered_response)
        bias_mitigated_response = self.bias_detector.mitigate_bias(filtered_response, bias_scores)
        response_embedding = EmbeddingVector(
            vector=combined_embedding,
            metadata={
                "user_input": user_input,
                "response": bias_mitigated_response,
                "active_persona": active_persona,
                "safety_scores": safety_scores,
                "bias_scores": bias_scores,
                "timestamp": time.time()
            }
        )
        data = {
            "user_input": user_input,
            "response": bias_mitigated_response,
            "active_persona": active_persona,
            "timestamp": time.time()
        }
        chain_id = context.get("chain_id") if context else None
        context_references = context.get("block_ids") if context else []
        block_id = self.wdbx.create_conversation_block(
            data=data,
            embeddings=[embedding_vector, response_embedding],
            chain_id=chain_id,
            context_references=context_references
        )
        return bias_mitigated_response, block_id
    
    def process_user_input(self, user_input: str, context: Dict[str, any] = None) -> Tuple[str, str]:
        persona = self.determine_optimal_persona(user_input, context)
        response, block_id = self.generate_response(user_input, persona, context)
        return response, block_id
