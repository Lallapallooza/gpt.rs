from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

Workload = Literal["validate", "trace", "bench", "run", "all"]
OutputFormat = Literal["table", "json", "csv"]


@dataclass(frozen=True)
class RunConfig:
    model: str
    backend: str = "faer"
    torch_device: str = "cpu"
    seed: int = 0
    rtol: float = 1e-4
    atol: float = 1e-4
    warmup: int = 1
    iters: int = 3
    threads: int = 1
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    model: str
    ok: bool
    torch_shape: Tuple[int, ...]
    gptrs_shape: Tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceDiff:
    key: str
    shape: Tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    allclose: bool


@dataclass(frozen=True)
class TraceResult:
    model: str
    ok: bool
    diffs: List[TraceDiff]
    missing_in_gptrs: List[str]
    missing_in_torch: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchStats:
    impl: str
    times_s: List[float]
    mean_s: float
    units_per_s: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchResult:
    model: str
    threads: int
    unit_label: str
    units_per_iter: float
    gptrs: BenchStats
    torch: BenchStats
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CliRunResult:
    model: str
    impl: str
    exit_code: int
    wall_s: float
    extra: Dict[str, Any] = field(default_factory=dict)
