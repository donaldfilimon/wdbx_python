# wdbx/attention.py
import numpy as np
from typing import Tuple, Optional, List
from math import sqrt


class MultiHeadAttention:
    """
    Implements the multi-head attention mechanism as described in 
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1) -> None:
        """
        Initialize the multi-head attention layer.

        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            dropout_rate: Probability of dropping attention weights (regularization)
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate

        # Initialize weights with He initialization for better gradient flow
        scale_factor = sqrt(2.0 / d_model)  # He initialization
        self.W_q = np.random.randn(num_heads, d_model, self.d_k) * scale_factor
        self.W_k = np.random.randn(num_heads, d_model, self.d_k) * scale_factor
        self.W_v = np.random.randn(num_heads, d_model, self.d_k) * scale_factor
        self.W_o = np.random.randn(num_heads * self.d_k, d_model) * scale_factor

    def _softmax(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute softmax values for each set of scores in x.
        Applies numerical stability by subtracting the maximum value.

        Args:
            x: Input tensor
            mask: Optional mask to apply before softmax

        Returns:
            Softmax output
        """
        if mask is not None:
            # Apply mask by setting masked positions to large negative value
            x = np.where(mask == 0, -1e9, x)

        # Numerical stability: subtract max value before exponentiating
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-9)  # Add epsilon to prevent division by zero

    def _apply_dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply dropout to input tensor during training.

        Args:
            x: Input tensor
            training: Whether in training mode or not

        Returns:
            Tensor with dropout applied if in training mode
        """
        if not training or self.dropout_rate <= 0:
            return x

        # Create and scale dropout mask
        keep_prob = 1.0 - self.dropout_rate
        mask = (np.random.random(x.shape) < keep_prob) / keep_prob
        return x * mask

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                  mask: Optional[np.ndarray] = None, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query matrix
            K: Key matrix
            V: Value matrix
            mask: Optional mask to prevent attention to certain positions
            training: Whether in training mode (for dropout)

        Returns:
            Attention output and attention weights
        """
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / sqrt(self.d_k)

        # Apply softmax with mask handling integrated
        attention_weights = self._softmax(scores, mask)

        # Apply dropout to attention weights
        attention_weights_dropout = self._apply_dropout(attention_weights, training)

        # Return attention output and original attention weights (for visualization)
        return np.matmul(attention_weights_dropout, V), attention_weights

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                mask: Optional[np.ndarray] = None, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process input through multi-head attention.

        Args:
            Q: Query matrix of shape [batch_size, seq_len, d_model]
            K: Key matrix of shape [batch_size, seq_len, d_model]
            V: Value matrix of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = Q.shape

        # Pre-allocate memory for better performance
        all_head_outputs = []
        attention_weights_list = []

        # Process all attention heads in parallel
        for h in range(self.num_heads):
            # Project queries, keys, and values for this head
            Q_h = np.matmul(Q, self.W_q[h])  # [batch_size, seq_len, d_k]
            K_h = np.matmul(K, self.W_k[h])  # [batch_size, seq_len, d_k]
            V_h = np.matmul(V, self.W_v[h])  # [batch_size, seq_len, d_k]

            # Compute attention
            head_output, attention_weights = self.attention(Q_h, K_h, V_h, mask, training)
            all_head_outputs.append(head_output)
            attention_weights_list.append(attention_weights)

        # Stack and reshape outputs from all heads
        heads = np.stack(all_head_outputs)  # [num_heads, batch_size, seq_len, d_k]
        concat_heads = heads.transpose(1, 2, 0, 3).reshape(batch_size, seq_len, -1)

        # Final linear projection
        output = np.matmul(concat_heads, self.W_o)

        # Average attention weights across heads (for visualization)
        avg_attention_weights = np.mean(np.stack(attention_weights_list), axis=0)

        return output, avg_attention_weights
