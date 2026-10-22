from __future__ import annotations

import json
from typing import Any

from ..core import RunConfig
from ..gptrs_py import hf_model_dir
from ._vision_case import VisionCaseBase


class MobileNetV3Case(VisionCaseBase):
    """A timm MobileNetV3 checkpoint, validated against timm: `transformers` has no MobileNetV3."""

    name = "mobilenetv3_large_100"

    def _torch_impl_label(self, cfg: RunConfig) -> str:
        return f"timm:{cfg.params['hf_model']}"

    def _build_torch_model(self, cfg: RunConfig) -> Any:
        import timm

        directory = hf_model_dir(cfg)
        architecture = json.loads((directory / "config.json").read_text())["architecture"]
        model = timm.create_model(
            architecture,
            pretrained=True,
            pretrained_cfg_overlay={"file": str(directory / "model.safetensors")},
        )
        return model.eval()

    def _torch_logits(self, model: Any, pixel_values: Any) -> Any:
        return model(pixel_values)
