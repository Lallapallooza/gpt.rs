from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Workload = Literal["validate", "bench", "run", "all"]
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
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    model: str
    ok: bool
    torch_shape: tuple[int, ...]
    gptrs_shape: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchStats:
    impl: str
    times_s: list[float]
    mean_s: float
    units_per_s: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchResult:
    model: str
    threads: int
    unit_label: str
    units_per_iter: float
    gptrs: BenchStats
    torch: BenchStats
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CliRunResult:
    model: str
    impl: str
    exit_code: int
    wall_s: float
    extra: dict[str, Any] = field(default_factory=dict)


def encode_prompt(tokenizer: Any, text: str, max_prompt_tokens: int) -> list[int]:
    """Encodes `text` without special tokens and keeps the last `max_prompt_tokens` ids.

    0 keeps all of them.
    """
    tokens = [int(tok) for tok in tokenizer.encode(text, add_special_tokens=False)]
    if max_prompt_tokens > 0 and len(tokens) > max_prompt_tokens:
        tokens = tokens[-max_prompt_tokens:]
    if not tokens:
        raise ValueError("prompt produced zero tokens. Provide a non-empty prompt.")
    return tokens


def resolve_torch_dtype(name: str) -> Any:
    """Maps a `--torch-dtype` choice to a torch dtype. `"auto"` passes through unchanged."""
    import torch

    mapping: dict[str, Any] = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        known = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"unsupported torch dtype {name!r}. Expected one of: {known}")
    return mapping[name]
