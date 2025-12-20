#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from gptrs_eval.tensor_archive import save as save_tensor_archive
from gptrs_eval.weights import build_mobilenet_v2_weights, build_resnet34_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Torch model weights for gpt-rs CLI.")
    parser.add_argument("--model", choices=["resnet34", "mobilenet_v2"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output tensor archive path.")
    return parser.parse_args()


def export_resnet34() -> Dict[str, np.ndarray]:
    from torchvision import models as tvm  # type: ignore[import-not-found, import-untyped]

    weights = tvm.ResNet34_Weights.DEFAULT
    model = tvm.resnet34(weights=weights).eval()
    return build_resnet34_weights(model)


def export_mobilenet_v2() -> Dict[str, np.ndarray]:
    from torchvision import models as tvm  # type: ignore[import-not-found, import-untyped]

    weights = tvm.MobileNet_V2_Weights.DEFAULT
    model = tvm.mobilenet_v2(weights=weights).eval()
    return build_mobilenet_v2_weights(model)


def main() -> int:
    args = parse_args()

    if args.model == "resnet34":
        tensors = export_resnet34()
    else:
        tensors = export_mobilenet_v2()

    save_tensor_archive(args.out, tensors)
    print(f"wrote {len(tensors)} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
