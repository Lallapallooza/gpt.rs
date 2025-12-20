"""
GPT models (inference).

This module provides GPT-style transformer inference backed by gpt-rs.
"""

try:
    from gpt_rs._native import gpt as _gpt  # type: ignore
except Exception as _err:  # pragma: no cover
    _gpt = None
    _import_error = _err

__all__ = [
    "Gpt",
]

if _gpt is None:  # pragma: no cover

    def _raise() -> None:
        raise ImportError(
            "gpt_rs._native.gpt is unavailable; rebuild the Python extension"
        ) from _import_error

    def Gpt(*_args, **_kwargs):  # type: ignore[misc]
        _raise()

else:
    Gpt = _gpt.Gpt
