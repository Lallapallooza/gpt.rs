"""
Unified model namespace.

This module exists to avoid model-type-specific namespaces (e.g. "vision") in user code.
"""

from gpt_rs.gpt import Gpt  # noqa: F401
from gpt_rs.vision import MobileNetV2, ResNet34  # noqa: F401

__all__ = [
    "Gpt",
    "ResNet34",
    "MobileNetV2",
]
