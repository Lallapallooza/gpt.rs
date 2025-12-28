from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Sequence

from .core import BenchResult, OutputFormat, ValidationResult


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def dumps_json(results: Sequence[Any]) -> str:
    payload = [_to_jsonable(r) for r in results]
    return json.dumps(payload, indent=2, sort_keys=True)


def print_validation(res: ValidationResult) -> None:
    print(f"model={res.model} allclose={res.ok}")
    print(f"max_abs_diff={res.max_abs_diff:.6e} mean_abs_diff={res.mean_abs_diff:.6e}")
    if res.extra:
        print(f"extra={res.extra}")


def print_bench(results: Sequence[BenchResult], fmt: OutputFormat) -> None:
    if fmt == "csv":
        print("model,impl,threads,units_per_iter,unit_label,mean_s,units_per_s")
        for r in results:
            for stats in (r.gptrs, r.torch):
                print(
                    f"{r.model},{stats.impl},{r.threads},{r.units_per_iter},"
                    f"{r.unit_label},{stats.mean_s:.6f},{stats.units_per_s:.2f}"
                )
        return

    if fmt == "json":
        print(dumps_json(results))
        return

    # table-ish
    for r in results:
        print(f"model={r.model} threads={r.threads} units={r.units_per_iter} {r.unit_label}/iter")
        print(f"  gpt-rs: mean_s={r.gptrs.mean_s:.6f} {r.unit_label}/s={r.gptrs.units_per_s:.2f}")
        print(f"  torch:  mean_s={r.torch.mean_s:.6f} {r.unit_label}/s={r.torch.units_per_s:.2f}")
