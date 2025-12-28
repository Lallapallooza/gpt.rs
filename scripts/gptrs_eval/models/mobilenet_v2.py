from __future__ import annotations

from pathlib import Path
from typing import Any

from ._vision_case import VisionCaseBase


class MobileNetV2Case(VisionCaseBase):
    name = "mobilenet_v2"
    torch_impl_label = "torchvision:mobilenet_v2"
    checkpoint_default = Path("checkpoints/mobilenet_v2.bin")

    def _build_torch_model(self, tvm: Any) -> Any:
        weights = tvm.MobileNet_V2_Weights.DEFAULT
        return tvm.mobilenet_v2(weights=weights)
