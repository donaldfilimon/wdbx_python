from ast import Mod
from typing import Dict, Tuple, Any, Optional, Union, List
import time
import numpy as np
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
        # Store original dimensions for validation
        if persona_embeddings:
            first_key = next(iter(persona_embeddings))
            self.embedding_dim = persona_embeddings[first_key].shape[0]

    def inject_persona_token(self, persona_id: str, user_input_embedding: np.ndarray,
                            persona_weight: float = 0.2) -> np.ndarray:
        """
        Injects a persona token into a user input embedding.

        Args:
            persona_id: The ID of the persona to inject
            user_input_embedding: The user's input embedding
            persona_weight: Weight of the persona embedding (default: 0.2)

        Returns:
            Combined embedding of persona and user input

        Raises:
            ValueError: If persona_id is unknown or embeddings have mismatched dimensions
        """
        if persona_id not in self.persona_embeddings:
            raise ValueError(f"Unknown persona: {persona_id}")

        persona_embedding = self.persona_embeddings[persona_id]
        if persona_embedding.shape != user_input_embedding.shape:
            raise ValueError(f"Dimension mismatch between persona ({persona_embedding.shape}) and user input embeddings ({user_input_embedding.shape}).")

        # Ensure weights are valid
        if not 0.0 <= persona_weight <= 1.0:
            raise ValueError(f"Persona weight must be between 0.0 and 1.0, got {persona_weight}")

        user_weight = 1.0 - persona_weight
        return persona_weight * persona_embedding + user_weight * user_input_embedding

    def create_blended_embedding(self, persona_weights: Dict[str, float], user_input_embedding: np.ndarray,
                                user_weight: float = 0.8) -> np.ndarray:
        """
        Creates a blended embedding from multiple personas with weights.

        Args:
            persona_weights: Dictionary mapping persona IDs to their weights
            user_input_embedding: The user's input embedding
            user_weight: Fixed weight for user input (default: 0.8)

        Returns:
            Blended embedding combining multiple personas and user input

        Raises:
            ValueError: If weights don't sum to 1.0 or an unknown persona is referenced
        """
        if not persona_weights:
            raise ValueError("No personas provided for blending")

        total_weight = sum(persona_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            # Normalize weights if they don't sum to 1.0
            logger.warning(f"Persona weights sum to {total_weight}, normalizing to 1.0")
            factor = 1.0 / total_weight
            persona_weights = {k: v * factor for k, v in persona_weights.items()}

        # Validate user weight
        if not 0.0 <= user_weight <= 1.0:
            raise ValueError(f"User weight must be between 0.0 and 1.0, got {user_weight}")

        result = np.zeros_like(user_input_embedding)
        for persona_id, weight in persona_weights.items():
            if persona_id not in self.persona_embeddings:
                raise ValueError(f"Unknown persona: {persona_id}")
            result += weight * self.persona_embeddings[persona_id]

        # Scale persona blend and combine with user input
        return (1 - user_weight) * result + user_weight * user_input_embedding

    def add_new_persona(self, persona_id: str, embedding: np.ndarray) -> None:
        """
        Adds a new persona embedding to the collection.

        Args:
            persona_id: The ID for the new persona
            embedding: The embedding vector for the new persona

        Raises:
            ValueError: If persona already exists or embedding has wrong dimensions
        """
        if persona_id in self.persona_embeddings:
            raise ValueError(f"Persona {persona_id} already exists")

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {embedding.shape[0]}")

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self.persona_embeddings[persona_id] = embedding
        logger.info(f"Added new persona: {persona_id}")


class PersonaManager:
    """
    Manages multi-persona interactions including response generation.
    """
    def __init__(self, wdbx: Any) -> None:
        """
        Initialize the PersonaManager with predefined personas.

        Args:
            wdbx: The WDBX instance that will store conversation data
        """
        self.wdbx = wdbx
        # Initialize persona embeddings
        self.persona_embeddings = {
            "Abbey": np.random.randn(self.wdbx.vector_dimension).astype(np.float32),
            "Aviva": np.random.randn(self.wdbx.vector_dimension).astype(np.float32),
            "Abi": np.random.randn(self.wdbx.vector_dimension).astype(np.float32),
            "Alex": np.random.randn(self.wdbx.vector_dimension).astype(np.float32)  # Added a new persona
        }

        # Persona descriptions for better response generation
        self.persona_descriptions = {
            "Abbey": "empathetic and supportive, focuses on emotional connection",
            "Aviva": "analytical and direct, focuses on technical accuracy",
            "Abi": "balanced and methodical, provides nuanced analysis",
            "Alex": "creative and innovative, offers unique perspectives"
        }

        # Normalize all embeddings
        for persona_id, embedding in self.persona_embeddings.items():
            norm = np.linalg.norm(embedding)
            if norm > 0:
                self.persona_embeddings[persona_id] = embedding / norm

        self.persona_token_manager = PersonaTokenManager(self.persona_embeddings)

        # Import and initialize content safety components
        from wdbx.content_filter import ContentFilter, BiasDetector
        self.content_filter = ContentFilter(
            sensitive_topics=["illegal activities", "dangerous substances", "explicit content"],
            offensive_patterns=["offensive language", "hate speech", "discriminatory language"]
        )
        self.bias_detector = BiasDetector(
            bias_attributes=["gender", "race", "age", "religion", "nationality", "disability", "sexual orientation"]
        )

        # Cache for storing recent responses to improve performance
        self.response_cache = {}
        # Track conversation history for better context awareness
        self.conversation_history = {}

        # Initialize neural backtracker for complex query resolution
        self.backtracker = wdbx.NeuralBacktracker(self.wdbx.vector_dimension)

        logger.info("PersonaManager initialized with Abbey, Aviva, Abi, and Alex.")

    def determine_optimal_persona(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, float]]:
        """
        Determines the optimal persona or persona blend based on user input.

        Args:
            user_input: The user's input text
            context: Optional conversation context

        Returns:
            Either a single persona ID or a dictionary of persona IDs with weights
        """
        # Enhanced keyword lists for better persona matching
        technical_keywords = ["how", "what", "explain", "code", "function", "algorithm",
                            "technical", "implementation", "syntax", "programming", "logic",
                            "data", "process", "compute", "calculate", "analyze", "define"]

        emotional_keywords = ["feel", "sad", "happy", "worried", "anxious", "excited",
                            "upset", "emotional", "concerned", "distressed", "afraid",
                            "disappointed", "frustrated", "hopeful", "stressed", "overwhelmed"]

        creative_keywords = ["imagine", "create", "design", "innovate", "brainstorm",
                           "alternative", "creative", "new", "unique", "different",
                           "original", "artistic", "inspiration", "visualize"]

        # Calculate scores with word boundary check for more accurate matching
        technical_score = sum(1 for kw in technical_keywords
                            if f" {kw.lower()} " in f" {user_input.lower()} "
                            or user_input.lower().startswith(f"{kw.lower()} ")
                            or user_input.lower().endswith(f" {kw.lower()}"))

        emotional_score = sum(1 for kw in emotional_keywords
                            if f" {kw.lower()} " in f" {user_input.lower()} "
                            or user_input.lower().startswith(f"{kw.lower()} ")
                            or user_input.lower().endswith(f" {kw.lower()}"))

        creative_score = sum(1 for kw in creative_keywords
                           if f" {kw.lower()} " in f" {user_input.lower()} "
                           or user_input.lower().startswith(f"{kw.lower()} ")
                           or user_input.lower().endswith(f" {kw.lower()}"))

        # Consider context if provided
        if context and "previous_persona" in context:
            # Add bias toward previously used persona for conversation continuity
            prev_persona = context["previous_persona"]
            if prev_persona == "Aviva":
                technical_score += 0.7
            elif prev_persona == "Abbey":
                emotional_score += 0.7
            elif prev_persona == "Alex":
                creative_score += 0.7
            elif prev_persona == "Abi":
                # Abi is balanced, so add a small amount to all scores
                technical_score += 0.3
                emotional_score += 0.3
                creative_score += 0.3

        # Check conversation length for continuity
        conversation_length = len(context.get("block_ids", [])) if context else 0
        if conversation_length > 3:
            # Increase continuity bias for longer conversations
            continuity_factor = min(1.0, 0.3 + (conversation_length * 0.05))
            logger.debug(f"Applying continuity factor of {continuity_factor} for conversation length {conversation_length}")
            if context and "previous_persona" in context:
                prev_persona = context["previous_persona"]
                if prev_persona == "Aviva":
                    technical_score += continuity_factor
                elif prev_persona == "Abbey":
                    emotional_score += continuity_factor
                elif prev_persona == "Alex":
                    creative_score += continuity_factor

        # Decision logic with creative persona consideration
        scores = {
            "Abbey": emotional_score,
            "Aviva": technical_score,
            "Abi": (technical_score + emotional_score + creative_score) / 3,# Balanced score
            "Alex": creative_score
        }

        # Find the highest scoring persona
        max_score = max(scores.values())
        max_persona = max(scores, key=scores.get)

        # If one persona is clearly dominant (2x any other score)
        if all(max_score > 2 * v for k, v in scores.items() if k != max_persona):
            logger.debug(f"Selected single persona {max_persona} with dominant score {max_score}")
            return max_persona

        # Otherwise, calculate a weighted blend
        total = sum(scores.values())

        if total < 0.1:  # Very low scores across the board
            # Default to a balanced blend with Abi having slightly more weight
            logger.debug("No clear pattern detected, using balanced blend with Abi emphasis")
            return {"Abbey": 0.3, "Aviva": 0.3, "Abi": 0.3, "Alex": 0.1}

        # Normalize scores to create weights
        weights = {k: v/total for k, v in scores.items() if v > 0}

        # Ensure we have at least 2 personas for a blend
        if len(weights) < 2:
            # Add Abi as a balancing persona
            weights["Abi"] = 0.2
            # Rescale other weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}

        logger.debug(f"Created persona blend with weights: {weights}")
        return weights

    def generate_response(self, user_input: str, persona_id: Union[str, Dict[str, float]],
                          context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Generates a response using the specified persona or persona blend.

        Args:
            user_input: The user's input text
            persona_id: Either a single persona ID or a dictionary of persona IDs with weights
            context: Optional conversation context

        Returns:
            A tuple containing (response_text, block_id)
        """
        start_time = time.time()

        # Create embedding for user input - in a real implementation this would use an actual embedding model
        user_input_embedding = np.random.randn(self.wdbx.vector_dimension).astype(np.float32)
        norm = np.linalg.norm(user_input_embedding)
        if norm > 0:
            user_input_embedding = user_input_embedding / norm

        # Combine with persona embedding
        if isinstance(persona_id, str):
            combined_embedding = self.persona_token_manager.inject_persona_token(
                persona_id,
                user_input_embedding,
                persona_weight=0.3  # Slightly increased persona influence
            )
            active_persona = persona_id
            persona_desc = self.persona_descriptions.get(persona_id, "balanced and helpful")
        else:
            combined_embedding = self.persona_token_manager.create_blended_embedding(
                persona_id,
                user_input_embedding,
                user_weight=0.75  # Adjusted user weight for blended personas
            )
            # Create a description of the blend
            active_persona = "Blended"
            persona_blend_desc = []
            for p_id, weight in sorted(persona_id.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.1:  # Only include significant contributions
                    persona_blend_desc.append(f"{int(weight*100)}% {p_id}")
            persona_desc = f"blend of {', '.join(persona_blend_desc)}"

        # Create embedding vector with metadata
        embedding_vector = EmbeddingVector(
            vector=combined_embedding,
            metadata={
                "user_input": user_input,
                "active_persona": active_persona,
                "timestamp": time.time(),
                "conversation_id": context.get("conversation_id") if context else None
            }
        )

        # Apply neural backtracking for complex queries
        if len(user_input.split()) > 10 or "?" in user_input:
            backtracking_results = self.backtracker.find_relevant_information(
                query_embedding=combined_embedding,
                top_k=3
            )
            # In a real implementation, these results would inform the response generation
        else:
            backtracking_results = []

        # Generate appropriate response based on active persona
        # In a real implementation, this would use an LLM with the persona as context
        if active_persona == "Abbey":
            response = (f"I understand your question about '{user_input}'. "
                        "I want to help you with this in a supportive way. "
                        "It's important to address both the practical and emotional aspects here...")
        elif active_persona == "Aviva":
            response = (f"Regarding '{user_input}': "
                        "The technical analysis shows that... "
                        "Here is the direct answer based on the most relevant data...")
        elif active_persona == "Abi":
            response = (f"About '{user_input}': "
                        "Let me provide a balanced, methodical analysis. "
                        "There are several factors to consider...")
        elif active_persona == "Alex":
            response = (f"Thinking about '{user_input}' from a creative perspective: "
                        "Let's explore some unconventional approaches... "
                        "Here are some innovative ideas that might help...")
        else:
            # For blended personas, create a more nuanced response
            response = (f"In response to '{user_input}': "
                        f"As a {persona_desc}, I'll address this from multiple angles. "
                        "Considering both the technical details and the human impact...")

        # Apply content filtering and bias detection
        filtered_response, safety_scores, was_filtered = self.content_filter.filter_content(response)
        bias_scores = self.bias_detector.measure_bias(filtered_response)
        bias_mitigated_response = self.bias_detector.mitigate_bias(filtered_response, bias_scores)

        # Create response embedding
        response_embedding = EmbeddingVector(
            vector=np.random.randn(self.wdbx.vector_dimension).astype(np.float32),  # In real implementation, this would be the embedding of the response
            metadata={
                "user_input": user_input,
                "response": bias_mitigated_response,
                "active_persona": active_persona,
                "safety_scores": safety_scores,
                "bias_scores": bias_scores,
                "was_filtered": was_filtered,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time
            }
        )

        # Prepare data for blockchain storage
        data = {
            "user_input": user_input,
            "response": bias_mitigated_response,
            "active_persona": active_persona,
            "persona_description": persona_desc,
            "timestamp": time.time(),
            "safety_metrics": {
                "safety_scores": safety_scores,
                "bias_scores": bias_scores,
                "was_filtered": was_filtered
            },
            "performance_metrics": {
                "processing_time": time.time() - start_time,
                "embedding_dimension": self.wdbx.vector_dimension
            }
        }

        # Get conversation chain information from context
        chain_id = context.get("chain_id") if context else None
        context_references = context.get("block_ids", []) if context else []

        # Add conversation ID for tracking related exchanges
        conversation_id = context.get("conversation_id") if context else f"conv-{int(time.time())}"

        # Store in blockchain
        block_id = self.wdbx.create_conversation_block(
            data=data,
            embeddings=[embedding_vector, response_embedding],
            chain_id=chain_id,
            context_references=context_references,
            conversation_id=conversation_id
        )

        # Cache response for potential future reference
        self.response_cache[user_input] = {
            "response": bias_mitigated_response,
            "block_id": block_id,
            "active_persona": active_persona,
            "timestamp": time.time()
        }

        # Update conversation history
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        self.conversation_history[conversation_id].append({
            "user_input": user_input,
            "response": bias_mitigated_response,
            "active_persona": active_persona,
            "block_id": block_id,
            "timestamp": time.time()
        })

        # Limit history size
        if len(self.conversation_history[conversation_id]) > 20:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-20:]

        return bias_mitigated_response, block_id

    def process_user_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Process user input by determining persona and generating response.

        Args:
            user_input: The user's input text
            context: Optional conversation context

        Returns:
            A tuple containing (response_text, block_id)
        """
        # Initialize context if needed
        if context is None:
            context = {
                "conversation_id": f"conv-{int(time.time())}",
                "block_ids": [],
                "user_preferences": {}
            }

        # Check for explicit persona request in input
        requested_persona = None
        for persona in self.persona_embeddings.keys():
            if f"as {persona}" in user_input.lower() or f"use {persona}" in user_input.lower():
                requested_persona = persona
                # Remove the persona request from the user input
                user_input = user_input.replace(f"as {persona}", "").replace(f"use {persona}", "").strip()
                logger.info(f"User explicitly requested persona: {persona}")
                break

        # Check cache for identical recent queries to improve response time
        cache_key = f"{user_input}_{context.get('conversation_id', '')}"
        if cache_key in self.response_cache and time.time() - self.response_cache[cache_key].get("timestamp", 0) < 300:
            logger.info(f"Using cached response for input: {user_input[:30]}...")
            cached = self.response_cache[cache_key]
            return cached["response"], cached["block_id"]

        # Use requested persona or determine optimal persona based on input
        persona = requested_persona if requested_persona else self.determine_optimal_persona(user_input, context)

        # Generate response using selected persona
        response, block_id = self.generate_response(user_input, persona, context)

        # Update context with the selected persona for future continuity
        if isinstance(persona, str):
            context["previous_persona"] = persona
        else:
            # For blended personas, store the dominant one
            dominant_persona = max(persona.items(), key=lambda x: x[1])[0]
            context["previous_persona"] = dominant_persona

        # Add block ID to context for conversation history
        if "block_ids" not in context:
            context["block_ids"] = []
        context["block_ids"].append(block_id)

        return response, block_id
