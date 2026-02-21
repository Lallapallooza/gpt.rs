from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Type

from .core import BenchResult, RunConfig, ValidationResult


class ModelCase(Protocol):
    name: str

    def supported_workloads(self) -> List[str]: ...

    def add_cli_args(self, parser: Any) -> None: ...

    def validate(self, cfg: RunConfig) -> ValidationResult: ...

    def bench(self, cfg: RunConfig) -> BenchResult: ...

    def run(self, cfg: RunConfig) -> Any: ...


@dataclass(frozen=True)
class CaseSpec:
    module: str
    cls: str
    default_params: Mapping[str, Any] = field(default_factory=dict)


_FALLBACK_CASES: Dict[str, CaseSpec] = {
    "resnet34": CaseSpec("gptrs_eval.models.resnet34", "ResNet34Case"),
    "mobilenet_v2": CaseSpec("gptrs_eval.models.mobilenet_v2", "MobileNetV2Case"),
    "gpt2": CaseSpec("gptrs_eval.models.gpt2", "Gpt2Case"),
}
_CASES: Dict[str, CaseSpec] | None = None


def _load_cases() -> Dict[str, CaseSpec]:
    cases = dict(_FALLBACK_CASES)
    try:
        from exporters import iter_exporter_infos

        for info in iter_exporter_infos():
            eval_case = info.eval_case
            if eval_case is None:
                continue
            cases[str(eval_case.model_name)] = CaseSpec(
                module=str(eval_case.module),
                cls=str(eval_case.cls),
                default_params=dict(eval_case.default_params),
            )
    except Exception:
        # Keep static fallback registry when exporter metadata cannot be loaded.
        pass
    return cases


def _case_specs() -> Dict[str, CaseSpec]:
    global _CASES
    if _CASES is None:
        _CASES = _load_cases()
    return _CASES


def list_models() -> List[str]:
    return sorted(_case_specs().keys())


def get_case_default_params(name: str) -> Dict[str, Any]:
    spec = _case_specs().get(name)
    if spec is None:
        return {}
    return dict(spec.default_params)


def get_case(name: str) -> ModelCase:
    cases = _case_specs()
    if name not in cases:
        known = ", ".join(sorted(cases.keys()))
        raise KeyError(f"unknown model case: {name!r}. known models: {known}")
    spec = cases[name]
    mod = importlib.import_module(spec.module)
    case_cls: Type[ModelCase] = getattr(mod, spec.cls)
    return case_cls()
