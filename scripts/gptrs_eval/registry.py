from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .core import BenchResult, RunConfig, ValidationResult


class ModelCase(Protocol):
    name: str

    def supported_workloads(self) -> list[str]: ...

    def add_cli_args(self, parser: Any) -> None: ...

    def validate(self, cfg: RunConfig) -> ValidationResult: ...

    def bench(self, cfg: RunConfig) -> BenchResult: ...

    def run(self, cfg: RunConfig) -> Any: ...


@dataclass(frozen=True)
class CaseSpec:
    module: str
    cls: str
    default_params: Mapping[str, Any] = field(default_factory=dict)


_CASES: dict[str, CaseSpec] = {
    "gpt2": CaseSpec("gptrs_eval.models.gpt2", "Gpt2Case", {"hf_model": "gpt2", "prompt": "Hello"}),
    "ministral_3_3b_instruct_2512": CaseSpec(
        "gptrs_eval.models.ministral",
        "MinistralCase",
        {"hf_model": "mistralai/Ministral-3-3B-Instruct-2512", "prompt": "Hello"},
    ),
    "qwen3_8_27b": CaseSpec(
        "gptrs_eval.models.qwen3_5",
        "Qwen35Case",
        {"hf_model": "Qwen/Qwen3.8-27B", "prompt": "The capital of France is"},
    ),
    "resnet34": CaseSpec(
        "gptrs_eval.models.resnet34", "ResNet34Case", {"hf_model": "microsoft/resnet-34"}
    ),
    "mobilenet_v2": CaseSpec(
        "gptrs_eval.models.mobilenet_v2",
        "MobileNetV2Case",
        {"hf_model": "google/mobilenet_v2_1.0_224"},
    ),
}


def list_models() -> list[str]:
    return sorted(_CASES)


def get_case_default_params(name: str) -> dict[str, Any]:
    spec = _CASES.get(name)
    if spec is None:
        return {}
    return dict(spec.default_params)


def get_case(name: str) -> ModelCase:
    if name not in _CASES:
        known = ", ".join(sorted(_CASES))
        raise KeyError(f"unknown model case: {name!r}. known models: {known}")
    spec = _CASES[name]
    mod = importlib.import_module(spec.module)
    case_cls: type[ModelCase] = getattr(mod, spec.cls)
    return case_cls()
