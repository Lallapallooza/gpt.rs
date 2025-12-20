"""
Vision models (inference).

This module provides pretrained-weight compatible ImageNet classifiers implemented in gpt-rs.
"""

try:
    from gpt_rs._native import vision as _vision  # type: ignore
except Exception as _err:  # pragma: no cover
    _vision = None
    _import_error = _err

__all__ = [
    "Conv2d",
    "ResNet34",
    "MobileNetV2",
]

if _vision is None:  # pragma: no cover

    def _raise() -> None:
        raise ImportError(
            "gpt_rs._native.vision is unavailable; rebuild the Python extension"
        ) from _import_error

    def ResNet34(*_args, **_kwargs):  # type: ignore[misc]
        _raise()

    def MobileNetV2(*_args, **_kwargs):  # type: ignore[misc]
        _raise()

    def Conv2d(*_args, **_kwargs):  # type: ignore[misc]
        _raise()
else:
    Conv2d = _vision.Conv2d
    ResNet34 = _vision.ResNet34
    MobileNetV2 = _vision.MobileNetV2
