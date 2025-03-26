# wdbx/attention.py
import numpy as np
from typing import Tuple
from math import sqrt

class MultiHeadAttention:
    """
    Implements the multi-head attention mechanism.
    """
    def __init__(self, d_model: int, num_heads: int) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = np.random.randn(num_heads, d_model, self.d_k) / sqrt(d_model)
        self.W_k = np.random.randn(num_heads, d_model, self.d_k) / sqrt(d_model)
        self.W_v = np.random.randn(num_heads, d_model, self.d_k) / sqrt(d_model)
        self.W_o = np.random.randn(num_heads * self.d_k, d_model) / sqrt(num_heads * self.d_k)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / sqrt(self.d_k)
        attention_weights = self._softmax(scores)
        return np.matmul(attention_weights, V)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = Q.shape
        heads = np.zeros((self.num_heads, batch_size, seq_len, self.d_k))
        for h in range(self.num_heads):
            Q_h = np.matmul(Q, self.W_q[h])
            K_h = np.matmul(K, self.W_k[h])
            V_h = np.matmul(V, self.W_v[h])
            heads[h] = self.attention(Q_h, K_h, V_h)
        concat_heads = heads.transpose(1, 2, 0, 3).reshape(batch_size, seq_len, -1)
        output = np.matmul(concat_heads, self.W_o)
        return output
