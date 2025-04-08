# wdbx/ml/attention.py
"""
Multi-head attention implementation for WDBX.

This module provides a flexible implementation of multi-head attention
that can work with multiple ML backends (PyTorch, JAX, NumPy).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger
from . import JAX_AVAILABLE, TORCH_AVAILABLE, ArrayLike

logger = get_logger("wdbx.ml.attention")

# Create an ML backend instance for optimized operations
ml_backend = get_ml_backend()


class MultiHeadAttention:
    """
    Implements the multi-head attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Automatically uses JAX or PyTorch for acceleration when available,
    with a NumPy fallback implementation.
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

        # Track backend type for optimized operations
        self.backend_type = ml_backend.selected_backend
        logger.info(f"Initializing MultiHeadAttention with {self.backend_type} backend")

        # Initialize weights with He initialization for better gradient flow
        scale_factor = sqrt(2.0 / d_model)  # He initialization

        if self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            # Use PyTorch for parameter initialization
            self.W_q = torch.randn(num_heads, d_model, self.d_k) * scale_factor
            self.W_k = torch.randn(num_heads, d_model, self.d_k) * scale_factor
            self.W_v = torch.randn(num_heads, d_model, self.d_k) * scale_factor
            self.W_o = torch.randn(num_heads * self.d_k, d_model) * scale_factor

            # Keep track of device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.W_q = self.W_q.to(self.device)
            self.W_k = self.W_k.to(self.device)
            self.W_v = self.W_v.to(self.device)
            self.W_o = self.W_o.to(self.device)

        elif self.backend_type == "jax" and JAX_AVAILABLE:
            import jax

            # Use JAX for parameter initialization
            key = jax.random.PRNGKey(0)
            key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

            self.W_q = jax.random.normal(subkey1, (num_heads, d_model, self.d_k)) * scale_factor
            self.W_k = jax.random.normal(subkey2, (num_heads, d_model, self.d_k)) * scale_factor
            self.W_v = jax.random.normal(subkey3, (num_heads, d_model, self.d_k)) * scale_factor
            self.W_o = jax.random.normal(subkey4, (num_heads * self.d_k, d_model)) * scale_factor

            # JIT-compile key functions for performance
            self._jit_attention = jax.jit(self._jax_attention)

        else:
            # Fallback to NumPy
            self.W_q = np.random.randn(num_heads, d_model, self.d_k) * scale_factor
            self.W_k = np.random.randn(num_heads, d_model, self.d_k) * scale_factor
            self.W_v = np.random.randn(num_heads, d_model, self.d_k) * scale_factor
            self.W_o = np.random.randn(num_heads * self.d_k, d_model) * scale_factor

    def _softmax(self, x: ArrayLike, mask: Optional[ArrayLike] = None) -> ArrayLike:
        """
        Compute softmax values for each set of scores in x.
        Applies numerical stability by subtracting the maximum value.

        Args:
            x: Input tensor
            mask: Optional mask to apply before softmax

        Returns:
            Softmax output using the active backend
        """
        if self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            # Convert to PyTorch if needed
            x_torch = x if isinstance(x, torch.Tensor) else torch.tensor(x, device=self.device)
            mask_torch = None
            if mask is not None:
                mask_torch = (
                    mask
                    if isinstance(mask, torch.Tensor)
                    else torch.tensor(mask, device=self.device)
                )
                # Apply mask by setting masked positions to large negative value
                x_torch = torch.where(
                    mask_torch == 0, torch.tensor(-1e9, device=self.device), x_torch
                )

            # Apply softmax using PyTorch for numerical stability
            return torch.nn.functional.softmax(x_torch, dim=-1)

        elif self.backend_type == "jax" and JAX_AVAILABLE:
            import jax.numpy as jnp

            # Convert to JAX if needed
            x_jax = x if isinstance(x, jnp.ndarray) else jnp.array(x)

            if mask is not None:
                mask_jax = mask if isinstance(mask, jnp.ndarray) else jnp.array(mask)
                # Apply mask by setting masked positions to large negative value
                x_jax = jnp.where(mask_jax == 0, -1e9, x_jax)

            # Numerical stability: subtract max value before exponentiating
            max_x = jnp.max(x_jax, axis=-1, keepdims=True)
            exp_x = jnp.exp(x_jax - max_x)
            # Add epsilon to prevent division by zero
            return exp_x / (jnp.sum(exp_x, axis=-1, keepdims=True) + 1e-9)

        else:
            # NumPy fallback
            x_np = x if isinstance(x, np.ndarray) else np.array(x)

            if mask is not None:
                mask_np = mask if isinstance(mask, np.ndarray) else np.array(mask)
                # Apply mask by setting masked positions to large negative value
                x_np = np.where(mask_np == 0, -1e9, x_np)

            # Numerical stability: subtract max value before exponentiating
            exp_x = np.exp(x_np - np.max(x_np, axis=-1, keepdims=True))
            # Add epsilon to prevent division by zero
            return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-9)

    def _apply_dropout(self, x: ArrayLike, training: bool = True) -> ArrayLike:
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

        if self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            # Convert to PyTorch if needed
            x_torch = x if isinstance(x, torch.Tensor) else torch.tensor(x, device=self.device)

            # Use PyTorch's dropout
            return torch.nn.functional.dropout(x_torch, p=self.dropout_rate, training=training)

        elif self.backend_type == "jax" and JAX_AVAILABLE:
            import jax
            import jax.numpy as jnp

            # Convert to JAX if needed
            x_jax = x if isinstance(x, jnp.ndarray) else jnp.array(x)

            # Manual dropout implementation
            if training and self.dropout_rate > 0:
                key = jax.random.PRNGKey(0)  # Should use a random seed in practice
                keep_prob = 1.0 - self.dropout_rate
                mask = jax.random.bernoulli(key, p=keep_prob, shape=x_jax.shape) / keep_prob
                return x_jax * mask
            return x_jax

        else:
            # NumPy fallback
            x_np = x if isinstance(x, np.ndarray) else np.array(x)

            # Create and scale dropout mask
            if training and self.dropout_rate > 0:
                keep_prob = 1.0 - self.dropout_rate
                mask = (np.random.random(x_np.shape) < keep_prob) / keep_prob
                return x_np * mask
            return x_np

    def _jax_attention(self, Q, K, V, mask=None):
        """JAX implementation of scaled dot-product attention for JIT compilation."""
        import jax.numpy as jnp

        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = jnp.matmul(Q, jnp.transpose(K, (0, 2, 1))) / sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Apply softmax
        weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-9)

        # No dropout in JIT version for deterministic compilation
        # Return attention output and weights
        return jnp.matmul(weights, V), weights

    def attention(
        self,
        Q: ArrayLike,
        K: ArrayLike,
        V: ArrayLike,
        mask: Optional[ArrayLike] = None,
        training: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike]:
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
        if self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            # Convert inputs to PyTorch tensors
            Q_torch = Q if isinstance(Q, torch.Tensor) else torch.tensor(Q, device=self.device)
            K_torch = K if isinstance(K, torch.Tensor) else torch.tensor(K, device=self.device)
            V_torch = V if isinstance(V, torch.Tensor) else torch.tensor(V, device=self.device)
            mask_torch = None
            if mask is not None:
                mask_torch = (
                    mask
                    if isinstance(mask, torch.Tensor)
                    else torch.tensor(mask, device=self.device)
                )

            # Compute attention scores: Q * K^T / sqrt(d_k)
            scores = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) / sqrt(self.d_k)

            # Apply softmax with mask
            attention_weights = self._softmax(scores, mask_torch)

            # Apply dropout to attention weights
            if training and self.dropout_rate > 0:
                attention_weights = torch.nn.functional.dropout(
                    attention_weights, p=self.dropout_rate, training=training
                )

            # Compute attention output
            output = torch.matmul(attention_weights, V_torch)

            # Return as PyTorch tensors
            return output, attention_weights

        elif self.backend_type == "jax" and JAX_AVAILABLE:
            import jax.numpy as jnp

            # Convert inputs to JAX arrays
            Q_jax = Q if isinstance(Q, jnp.ndarray) else jnp.array(Q)
            K_jax = K if isinstance(K, jnp.ndarray) else jnp.array(K)
            V_jax = V if isinstance(V, jnp.ndarray) else jnp.array(V)
            mask_jax = None
            if mask is not None:
                mask_jax = mask if isinstance(mask, jnp.ndarray) else jnp.array(mask)

            # Use JIT-compiled version for performance
            if not training:  # JIT-compiled version doesn't support dropout
                return self._jit_attention(Q_jax, K_jax, V_jax, mask_jax)

            # Compute attention scores: Q * K^T / sqrt(d_k)
            scores = jnp.matmul(Q_jax, jnp.transpose(K_jax, (0, 2, 1))) / sqrt(self.d_k)

            # Apply softmax with mask
            attention_weights = self._softmax(scores, mask_jax)

            # Apply dropout to attention weights
            attention_weights_dropout = self._apply_dropout(attention_weights, training)

            # Return attention output and original attention weights (for visualization)
            return jnp.matmul(attention_weights_dropout, V_jax), attention_weights

        else:
            # NumPy fallback implementation
            Q_np = Q if isinstance(Q, np.ndarray) else np.array(Q)
            K_np = K if isinstance(K, np.ndarray) else np.array(K)
            V_np = V if isinstance(V, np.ndarray) else np.array(V)
            mask_np = None
            if mask is not None:
                mask_np = mask if isinstance(mask, np.ndarray) else np.array(mask)

            # Compute attention scores: Q * K^T / sqrt(d_k)
            scores = np.matmul(Q_np, K_np.transpose(0, 2, 1)) / sqrt(self.d_k)

            # Apply softmax with mask handling integrated
            attention_weights = self._softmax(scores, mask_np)

            # Apply dropout to attention weights
            attention_weights_dropout = self._apply_dropout(attention_weights, training)

            # Return attention output and original attention weights (for visualization)
            return np.matmul(attention_weights_dropout, V_np), attention_weights

    def forward(
        self,
        Q: ArrayLike,
        K: ArrayLike,
        V: ArrayLike,
        mask: Optional[ArrayLike] = None,
        training: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike]:
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
        # Convert inputs to the appropriate backend format
        if self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            # Get tensors in PyTorch format
            Q_torch = Q if isinstance(Q, torch.Tensor) else torch.tensor(Q, device=self.device)
            K_torch = K if isinstance(K, torch.Tensor) else torch.tensor(K, device=self.device)
            V_torch = V if isinstance(V, torch.Tensor) else torch.tensor(V, device=self.device)
            W_q = (
                self.W_q
                if isinstance(self.W_q, torch.Tensor)
                else torch.tensor(self.W_q, device=self.device)
            )
            W_k = (
                self.W_k
                if isinstance(self.W_k, torch.Tensor)
                else torch.tensor(self.W_k, device=self.device)
            )
            W_v = (
                self.W_v
                if isinstance(self.W_v, torch.Tensor)
                else torch.tensor(self.W_v, device=self.device)
            )
            W_o = (
                self.W_o
                if isinstance(self.W_o, torch.Tensor)
                else torch.tensor(self.W_o, device=self.device)
            )

            batch_size, seq_len, _ = Q_torch.shape

            # Pre-allocate output tensor
            head_outputs = []
            attention_weights_list = []

            # Process each attention head
            for h in range(self.num_heads):
                # Project inputs for this head
                Q_h = torch.matmul(Q_torch, W_q[h])
                K_h = torch.matmul(K_torch, W_k[h])
                V_h = torch.matmul(V_torch, W_v[h])

                # Compute attention
                head_output, attention_weights = self.attention(Q_h, K_h, V_h, mask, training)
                head_outputs.append(head_output)
                attention_weights_list.append(attention_weights)

            # Stack outputs from all heads
            multi_head = torch.stack(head_outputs)
            concat_heads = multi_head.permute(1, 2, 0, 3).reshape(batch_size, seq_len, -1)

            # Final linear projection
            output = torch.matmul(concat_heads, W_o)

            # Average attention weights for visualization
            avg_attention = torch.mean(torch.stack(attention_weights_list), dim=0)

            return output, avg_attention

        elif self.backend_type == "jax" and JAX_AVAILABLE:
            import jax.numpy as jnp

            # Get arrays in JAX format
            Q_jax = Q if isinstance(Q, jnp.ndarray) else jnp.array(Q)
            K_jax = K if isinstance(K, jnp.ndarray) else jnp.array(K)
            V_jax = V if isinstance(V, jnp.ndarray) else jnp.array(V)
            W_q = self.W_q if isinstance(self.W_q, jnp.ndarray) else jnp.array(self.W_q)
            W_k = self.W_k if isinstance(self.W_k, jnp.ndarray) else jnp.array(self.W_k)
            W_v = self.W_v if isinstance(self.W_v, jnp.ndarray) else jnp.array(self.W_v)
            W_o = self.W_o if isinstance(self.W_o, jnp.ndarray) else jnp.array(self.W_o)

            batch_size, seq_len, _ = Q_jax.shape

            # Process all attention heads
            head_outputs = []
            attention_weights_list = []

            for h in range(self.num_heads):
                # Project inputs for this head
                Q_h = jnp.matmul(Q_jax, W_q[h])
                K_h = jnp.matmul(K_jax, W_k[h])
                V_h = jnp.matmul(V_jax, W_v[h])

                # Compute attention
                head_output, attention_weights = self.attention(Q_h, K_h, V_h, mask, training)
                head_outputs.append(head_output)
                attention_weights_list.append(attention_weights)

            # Stack outputs from all heads
            multi_head = jnp.stack(head_outputs)
            concat_heads = jnp.transpose(multi_head, (1, 2, 0, 3)).reshape(batch_size, seq_len, -1)

            # Final linear projection
            output = jnp.matmul(concat_heads, W_o)

            # Average attention weights for visualization
            avg_attention = jnp.mean(jnp.stack(attention_weights_list), axis=0)

            return output, avg_attention

        else:
            # NumPy fallback implementation
            Q_np = Q if isinstance(Q, np.ndarray) else np.array(Q)
            K_np = K if isinstance(K, np.ndarray) else np.array(K)
            V_np = V if isinstance(V, np.ndarray) else np.array(V)

            batch_size, seq_len, _ = Q_np.shape

            # Pre-allocate memory for better performance
            all_head_outputs = []
            attention_weights_list = []

            # Process all attention heads
            for h in range(self.num_heads):
                # Project queries, keys, and values for this head
                Q_h = np.matmul(Q_np, self.W_q[h])  # [batch_size, seq_len, d_k]
                K_h = np.matmul(K_np, self.W_k[h])  # [batch_size, seq_len, d_k]
                V_h = np.matmul(V_np, self.W_v[h])  # [batch_size, seq_len, d_k]

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

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Convert weights to NumPy arrays for serialization or interoperability.

        Returns:
            Dictionary of weight matrices as NumPy arrays
        """
        if self.backend_type == "pytorch" and TORCH_AVAILABLE:

            return {
                "W_q": self.W_q.detach().cpu().numpy(),
                "W_k": self.W_k.detach().cpu().numpy(),
                "W_v": self.W_v.detach().cpu().numpy(),
                "W_o": self.W_o.detach().cpu().numpy(),
            }

        elif self.backend_type == "jax" and JAX_AVAILABLE:

            return {
                "W_q": np.array(self.W_q),
                "W_k": np.array(self.W_k),
                "W_v": np.array(self.W_v),
                "W_o": np.array(self.W_o),
            }

        else:
            # Already NumPy
            return {
                "W_q": self.W_q,
                "W_k": self.W_k,
                "W_v": self.W_v,
                "W_o": self.W_o,
            }

    def from_numpy(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Load weights from NumPy arrays.

        Args:
            weights: Dictionary of weight matrices as NumPy arrays
        """
        if self.backend_type == "pytorch" and TORCH_AVAILABLE:
            import torch

            self.W_q = torch.tensor(weights["W_q"], device=self.device)
            self.W_k = torch.tensor(weights["W_k"], device=self.device)
            self.W_v = torch.tensor(weights["W_v"], device=self.device)
            self.W_o = torch.tensor(weights["W_o"], device=self.device)

        elif self.backend_type == "jax" and JAX_AVAILABLE:
            import jax.numpy as jnp

            self.W_q = jnp.array(weights["W_q"])
            self.W_k = jnp.array(weights["W_k"])
            self.W_v = jnp.array(weights["W_v"])
            self.W_o = jnp.array(weights["W_o"])

        else:
            # NumPy backend
            self.W_q = weights["W_q"]
            self.W_k = weights["W_k"]
            self.W_v = weights["W_v"]
            self.W_o = weights["W_o"]
