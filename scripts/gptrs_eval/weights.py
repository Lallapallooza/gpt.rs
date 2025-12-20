from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def fuse_conv_bn(conv: Any, bn: Any) -> Tuple[Any, Any]:
    """Return (weight, bias) for an equivalent fused Conv2d (torch tensors)."""
    w = conv.weight.detach()
    if conv.bias is None:
        b = conv.weight.detach().new_zeros(conv.out_channels)
    else:
        b = conv.bias.detach()

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()

    denom = (var + bn.eps).rsqrt()
    scale = gamma * denom
    w_fused = w * scale.reshape(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta
    return w_fused, b_fused


def conv_weight_to_gptrs(w_fused: Any) -> np.ndarray:
    """torch Conv2d weight [O,I,KH,KW] -> gpt-rs canonical conv weight [O,I,KH,KW]."""
    return w_fused.contiguous().cpu().numpy().astype(np.float32, copy=False)


def depthwise_weight_to_gptrs(w_fused: Any) -> np.ndarray:
    """torch depthwise weight [C,1,KH,KW] -> gpt-rs canonical depthwise weight [C,1,KH,KW]."""
    c, one, kh, kw = w_fused.shape
    if int(one) != 1:
        raise ValueError(f"expected depthwise weight shape [C,1,KH,KW], got {w_fused.shape}")
    _ = (c, kh, kw)
    return w_fused.contiguous().cpu().numpy().astype(np.float32, copy=False)


def linear_weight_to_gptrs(w: Any) -> np.ndarray:
    """torch Linear weight [O,I] -> gpt-rs canonical linear weight [O,I]."""
    return w.detach().cpu().numpy().astype(np.float32, copy=False)


def linear_bias_to_gptrs(b: Any) -> np.ndarray:
    return b.detach().cpu().numpy().astype(np.float32, copy=False)


def build_resnet34_weights(model: Any) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}

    w, b = fuse_conv_bn(model.conv1, model.bn1)
    weights["conv1.weight"] = conv_weight_to_gptrs(w)
    weights["conv1.bias"] = linear_bias_to_gptrs(b)

    for layer_idx, layer in enumerate(
        [model.layer1, model.layer2, model.layer3, model.layer4], start=1
    ):
        for block_idx, block in enumerate(layer):
            w1, b1 = fuse_conv_bn(block.conv1, block.bn1)
            w2, b2 = fuse_conv_bn(block.conv2, block.bn2)
            weights[f"layer{layer_idx}.{block_idx}.conv1.weight"] = conv_weight_to_gptrs(w1)
            weights[f"layer{layer_idx}.{block_idx}.conv1.bias"] = linear_bias_to_gptrs(b1)
            weights[f"layer{layer_idx}.{block_idx}.conv2.weight"] = conv_weight_to_gptrs(w2)
            weights[f"layer{layer_idx}.{block_idx}.conv2.bias"] = linear_bias_to_gptrs(b2)

            if block.downsample is not None:
                ds_conv = block.downsample[0]
                ds_bn = block.downsample[1]
                wds, bds = fuse_conv_bn(ds_conv, ds_bn)
                weights[f"layer{layer_idx}.{block_idx}.downsample.weight"] = conv_weight_to_gptrs(
                    wds
                )
                weights[f"layer{layer_idx}.{block_idx}.downsample.bias"] = linear_bias_to_gptrs(bds)

    weights["fc.weight"] = linear_weight_to_gptrs(model.fc.weight)
    weights["fc.bias"] = linear_bias_to_gptrs(model.fc.bias)
    return weights


def build_mobilenet_v2_weights(model: Any) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}

    stem_conv = model.features[0][0]
    stem_bn = model.features[0][1]
    w, b = fuse_conv_bn(stem_conv, stem_bn)
    weights["stem.weight"] = conv_weight_to_gptrs(w)
    weights["stem.bias"] = linear_bias_to_gptrs(b)

    block_idx = 0
    for features_idx in range(1, len(model.features) - 1):
        block = model.features[features_idx]
        conv_seq = block.conv

        if len(conv_seq) == 4:
            exp = conv_seq[0]
            dw = conv_seq[1]
            proj_conv = conv_seq[2]
            proj_bn = conv_seq[3]

            wexp, bexp = fuse_conv_bn(exp[0], exp[1])
            weights[f"blocks.{block_idx}.expand.weight"] = conv_weight_to_gptrs(wexp)
            weights[f"blocks.{block_idx}.expand.bias"] = linear_bias_to_gptrs(bexp)
        else:
            dw = conv_seq[0]
            proj_conv = conv_seq[1]
            proj_bn = conv_seq[2]

        wdw, bdw = fuse_conv_bn(dw[0], dw[1])
        weights[f"blocks.{block_idx}.depthwise.weight"] = depthwise_weight_to_gptrs(wdw)
        weights[f"blocks.{block_idx}.depthwise.bias"] = linear_bias_to_gptrs(bdw)

        wproj, bproj = fuse_conv_bn(proj_conv, proj_bn)
        weights[f"blocks.{block_idx}.project.weight"] = conv_weight_to_gptrs(wproj)
        weights[f"blocks.{block_idx}.project.bias"] = linear_bias_to_gptrs(bproj)

        block_idx += 1

    head_conv = model.features[-1][0]
    head_bn = model.features[-1][1]
    wh, bh = fuse_conv_bn(head_conv, head_bn)
    weights["head.weight"] = conv_weight_to_gptrs(wh)
    weights["head.bias"] = linear_bias_to_gptrs(bh)

    linear = model.classifier[1]
    weights["classifier.weight"] = linear_weight_to_gptrs(linear.weight)
    weights["classifier.bias"] = linear_bias_to_gptrs(linear.bias)
    return weights
