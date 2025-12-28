"""gpt-rs: Python bindings.

This package is intentionally small: it focuses on loading checkpoint-backed models and running
inference (forward / generation), while keeping tensor/layer internals Rust-only.
"""

from gpt_rs._native import (  # noqa: F401
    LoadedModel,
    Tokenizer,
    backend_features,
    clear_dump_dir,
    get_backend,
    list_backends,
    load_model,
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
    supported_backends,
    supported_model_kinds,
    version_info,
)

__version__ = "0.1.0"
__all__ = [
    "LoadedModel",
    "Tokenizer",
    "load_model",
    "set_backend",
    "get_backend",
    "list_backends",
    "supported_backends",
    "supported_model_kinds",
    "backend_features",
    "version_info",
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
]
