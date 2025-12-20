"""
gpt-rs: Pure Rust GPT-style transformer with Python bindings

A high-performance transformer library with PyTorch-like API.

Quick Start:
    >>> import gpt_rs
    >>> import numpy as np
    >>>
    >>> # Create a tensor from NumPy
    >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    >>> tensor = gpt_rs.Tensor.from_numpy(arr)
"""

from gpt_rs import (
    functional,  # noqa: F401
    gpt,  # noqa: F401
    model,  # noqa: F401
    nn,  # noqa: F401
    vision,  # noqa: F401
)
from gpt_rs._native import (  # noqa: F401
    Tensor,
    Tokenizer,
    clear_dump_dir,
    get_backend,
    list_backends,
    profiling_pop_section,
    profiling_push_section,
    profiling_reset,
    profiling_take_report,
    profiling_take_report_bundle,
    profiling_take_report_json,
    profiling_take_trace_json,
    profiling_trace_disable,
    profiling_trace_enable,
    profiling_trace_reset,
    set_backend,
    set_dump_dir,
)

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "Tokenizer",
    "set_backend",
    "get_backend",
    "list_backends",
    "set_dump_dir",
    "clear_dump_dir",
    "profiling_reset",
    "profiling_take_report",
    "profiling_push_section",
    "profiling_pop_section",
    "profiling_take_report_bundle",
    "profiling_take_report_json",
    "profiling_trace_enable",
    "profiling_trace_disable",
    "profiling_trace_reset",
    "profiling_take_trace_json",
    "functional",
    "gpt",
    "model",
    "nn",
    "vision",
]
