from __future__ import annotations

from typing import Any, Dict

from ..weights import build_resnet34_weights
from ._vision_case import VisionCaseBase, torch_to_nhwc


def _trace_resnet34(torch_model: Any, x: Any) -> Dict[str, Any]:
    t: Dict[str, Any] = {}

    t["input.nhwc"] = torch_to_nhwc(x)

    x = torch_model.conv1(x)
    x = torch_model.bn1(x)
    t["stem.conv1"] = torch_to_nhwc(x)
    x = torch_model.relu(x)
    t["stem.relu"] = torch_to_nhwc(x)
    x = torch_model.maxpool(x)
    t["stem.maxpool"] = torch_to_nhwc(x)

    stages = [torch_model.layer1, torch_model.layer2, torch_model.layer3, torch_model.layer4]
    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        for block_idx, block in enumerate(stage):
            prefix = f"layer{stage_num}.{block_idx}"
            identity = x
            if block.downsample is not None:
                identity = block.downsample(x)
                t[f"{prefix}.downsample"] = torch_to_nhwc(identity)

            out = block.conv1(x)
            out = block.bn1(out)
            t[f"{prefix}.conv1"] = torch_to_nhwc(out)
            out = block.relu(out)
            t[f"{prefix}.relu1"] = torch_to_nhwc(out)

            out = block.conv2(out)
            out = block.bn2(out)
            t[f"{prefix}.conv2"] = torch_to_nhwc(out)

            out = out + identity
            t[f"{prefix}.add"] = torch_to_nhwc(out)
            out = block.relu(out)
            t[f"{prefix}.relu2"] = torch_to_nhwc(out)

            x = out

    x = torch_model.avgpool(x)
    t["avgpool"] = torch_to_nhwc(x)
    x = x.flatten(1)
    t["flatten"] = x
    x = torch_model.fc(x)
    t["logits"] = x
    return t


class ResNet34Case(VisionCaseBase):
    name = "resnet34"
    torch_impl_label = "torchvision:resnet34"

    def _build_torch_model(self, tvm: Any) -> Any:
        weights = tvm.ResNet34_Weights.DEFAULT
        return tvm.resnet34(weights=weights)

    def _build_weight_arrays(self, torch_model: Any) -> Dict[str, Any]:
        return build_resnet34_weights(torch_model)

    def _build_gpt_rs_model(self, gpt_rs: Any, weight_tensors: Dict[str, Any]) -> Any:
        return gpt_rs.model.ResNet34(weight_tensors)

    def _trace_torch(self, torch_model: Any, x_torch: Any) -> Dict[str, Any]:
        return _trace_resnet34(torch_model, x_torch)
