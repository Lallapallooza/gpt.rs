from __future__ import annotations

from pathlib import Path
from typing import Any

from ._vision_case import VisionCaseBase


class ResNet34Case(VisionCaseBase):
    name = "resnet34"
    torch_impl_label = "torchvision:resnet34"
    checkpoint_default = Path("checkpoints/resnet34.bin")

    def _build_torch_model(self, tvm: Any) -> Any:
        weights = tvm.ResNet34_Weights.DEFAULT
        return tvm.resnet34(weights=weights)
