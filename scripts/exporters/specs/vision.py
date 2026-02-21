from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from gptrs_eval.checkpoint import save as save_checkpoint
from gptrs_eval.weights import build_mobilenet_v2_weights, build_resnet34_weights

from ..types import (
    ArtifactDefaults,
    EvalCaseRegistration,
    ExporterInfo,
    ExportRequest,
    ExportResult,
)

_TensorMap = Dict[str, np.ndarray]


def _build_torchvision_model(model_name: str) -> Any:
    from torchvision import models as tvm  # type: ignore[import-not-found, import-untyped]

    if model_name == "resnet34":
        return tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT).eval()
    if model_name == "mobilenet_v2":
        return tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT).eval()
    raise ValueError(f"unsupported torchvision model {model_name!r}")


def _build_weights(model_name: str, model: Any) -> _TensorMap:
    builders: Dict[str, Callable[[Any], _TensorMap]] = {
        "resnet34": build_resnet34_weights,
        "mobilenet_v2": build_mobilenet_v2_weights,
    }
    if model_name not in builders:
        raise ValueError(f"unsupported weight builder for {model_name!r}")
    return builders[model_name](model)


class VisionExporter:
    def __init__(self, name: str, description: str) -> None:
        self.info = ExporterInfo(
            name=name,
            kind=name,
            description=description,
            artifacts=ArtifactDefaults(checkpoint=Path(f"checkpoints/{name}.bin")),
            eval_case=EvalCaseRegistration(
                model_name=name,
                module=f"gptrs_eval.models.{name}",
                cls="ResNet34Case" if name == "resnet34" else "MobileNetV2Case",
                default_params={"checkpoint": Path(f"checkpoints/{name}.bin")},
            ),
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _ = parser

    def export(self, request: ExportRequest) -> ExportResult:
        model = _build_torchvision_model(self.info.name)
        tensors = _build_weights(self.info.name, model)
        save_checkpoint(
            request.checkpoint_out,
            kind=self.info.kind,
            config={"num_classes": 1000},
            tensors=tensors,
        )
        return ExportResult(
            exporter=self.info.name,
            kind=self.info.kind,
            checkpoint=request.checkpoint_out,
            tensor_count=len(tensors),
            extras={"num_classes": 1000},
        )

    def validate(self, request: ExportRequest) -> list[str]:
        if request.checkpoint_out.exists():
            return []
        return [f"missing checkpoint: {request.checkpoint_out}"]


def build_vision_exporters() -> List[VisionExporter]:
    return [
        VisionExporter("resnet34", "Export torchvision ResNet-34 weights."),
        VisionExporter("mobilenet_v2", "Export torchvision MobileNetV2 weights."),
    ]
