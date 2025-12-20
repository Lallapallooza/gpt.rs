"""
Neural network layers.

This module provides PyTorch-like layer classes for building transformer models.
"""

from gpt_rs._native import nn as _nn

__all__ = [
    "Embedding",
    "Linear",
    "LayerNorm",
    "FeedForward",
]

# Re-export layer classes
Embedding = _nn.Embedding
Linear = _nn.Linear
LayerNorm = _nn.LayerNorm
FeedForward = _nn.FeedForward
