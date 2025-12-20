"""
Functional operations for tensor manipulations.

This module provides functional-style operations similar to PyTorch's torch.nn.functional.
All operations are backend-agnostic and work with the currently selected backend.
"""

from gpt_rs._native import functional as _functional

__all__ = [
    "softmax_last_dim",
    "gelu",
    "layer_norm",
    "matmul",
    "add_bias",
    "embedding_lookup",
]

# Re-export functional ops
softmax_last_dim = _functional.softmax_last_dim
gelu = _functional.gelu
layer_norm = _functional.layer_norm
matmul = _functional.matmul
add_bias = _functional.add_bias
embedding_lookup = _functional.embedding_lookup
