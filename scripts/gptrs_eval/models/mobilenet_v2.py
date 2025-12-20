from __future__ import annotations

from typing import Any, Dict

from ..weights import build_mobilenet_v2_weights
from ._vision_case import VisionCaseBase, torch_to_nhwc


def _trace_mobilenet_v2(torch_model: Any, x: Any) -> Dict[str, Any]:
    import torch.nn.functional as F

    t: Dict[str, Any] = {}

    t["input.nhwc"] = torch_to_nhwc(x)

    stem = torch_model.features[0]
    x = stem[0](x)
    x = stem[1](x)
    t["stem.conv"] = torch_to_nhwc(x)
    x = stem[2](x)
    t["stem.relu6"] = torch_to_nhwc(x)

    block_idx = 0
    for features_idx in range(1, len(torch_model.features) - 1):
        block = torch_model.features[features_idx]
        identity = x
        conv_seq = block.conv

        if len(conv_seq) == 4:
            exp = conv_seq[0]
            x = exp[0](x)
            x = exp[1](x)
            t[f"blocks.{block_idx}.expand"] = torch_to_nhwc(x)
            x = exp[2](x)
            t[f"blocks.{block_idx}.expand.relu6"] = torch_to_nhwc(x)
            dw = conv_seq[1]
            proj_conv = conv_seq[2]
            proj_bn = conv_seq[3]
        else:
            dw = conv_seq[0]
            proj_conv = conv_seq[1]
            proj_bn = conv_seq[2]

        x = dw[0](x)
        x = dw[1](x)
        t[f"blocks.{block_idx}.depthwise"] = torch_to_nhwc(x)
        x = dw[2](x)
        t[f"blocks.{block_idx}.depthwise.relu6"] = torch_to_nhwc(x)

        x = proj_conv(x)
        x = proj_bn(x)
        t[f"blocks.{block_idx}.project"] = torch_to_nhwc(x)

        if block.use_res_connect:
            x = x + identity
            t[f"blocks.{block_idx}.add"] = torch_to_nhwc(x)

        block_idx += 1

    head = torch_model.features[-1]
    x = head[0](x)
    x = head[1](x)
    t["head.conv"] = torch_to_nhwc(x)
    x = head[2](x)
    t["head.relu6"] = torch_to_nhwc(x)

    x = F.adaptive_avg_pool2d(x, (1, 1))
    t["avgpool"] = torch_to_nhwc(x)
    x = x.flatten(1)
    t["flatten"] = x
    x = torch_model.classifier(x)
    t["logits"] = x
    return t


class MobileNetV2Case(VisionCaseBase):
    name = "mobilenet_v2"
    torch_impl_label = "torchvision:mobilenet_v2"

    def _build_torch_model(self, tvm: Any) -> Any:
        weights = tvm.MobileNet_V2_Weights.DEFAULT
        return tvm.mobilenet_v2(weights=weights)

    def _build_weight_arrays(self, torch_model: Any) -> Dict[str, Any]:
        return build_mobilenet_v2_weights(torch_model)

    def _build_gpt_rs_model(self, gpt_rs: Any, weight_tensors: Dict[str, Any]) -> Any:
        return gpt_rs.model.MobileNetV2(weight_tensors)

    def _trace_torch(self, torch_model: Any, x_torch: Any) -> Dict[str, Any]:
        return _trace_mobilenet_v2(torch_model, x_torch)
