from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Type

from .core import BenchResult, RunConfig, TraceResult, ValidationResult


class ModelCase(Protocol):
    name: str

    def supported_workloads(self) -> List[str]: ...

    def add_cli_args(self, parser: Any) -> None: ...

    def validate(self, cfg: RunConfig) -> ValidationResult: ...

    def trace(self, cfg: RunConfig) -> TraceResult: ...

    def bench(self, cfg: RunConfig) -> BenchResult: ...

    def run(self, cfg: RunConfig) -> Any: ...


@dataclass(frozen=True)
class CaseSpec:
    module: str
    cls: str


_CASES: Dict[str, CaseSpec] = {
    "conv2d": CaseSpec("gptrs_eval.models.conv2d", "Conv2dCase"),
    "matmul": CaseSpec("gptrs_eval.models.matmul", "MatmulCase"),
    "resnet34": CaseSpec("gptrs_eval.models.resnet34", "ResNet34Case"),
    "mobilenet_v2": CaseSpec("gptrs_eval.models.mobilenet_v2", "MobileNetV2Case"),
    "gpt2": CaseSpec("gptrs_eval.models.gpt2", "Gpt2Case"),
}


def list_models() -> List[str]:
    return sorted(_CASES.keys())


def get_case(name: str) -> ModelCase:
    if name not in _CASES:
        raise KeyError(f"unknown model case: {name!r}")
    spec = _CASES[name]
    mod = importlib.import_module(spec.module)
    case_cls: Type[ModelCase] = getattr(mod, spec.cls)
    return case_cls()
