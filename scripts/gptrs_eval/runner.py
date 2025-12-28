from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .core import BenchStats, ValidationResult


def validate_arrays(
    torch_np: np.ndarray, gptrs_np: np.ndarray, rtol: float, atol: float
) -> Tuple[bool, float, float]:
    if torch_np.shape != gptrs_np.shape:
        return False, float("inf"), float("inf")
    diff = np.abs(torch_np - gptrs_np)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    ok = bool(np.allclose(torch_np, gptrs_np, rtol=rtol, atol=atol))
    return ok, max_abs, mean_abs


def time_many(
    run_once: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    before_warmup: Optional[Callable[[], Any]] = None,
    after_warmup: Optional[Callable[[], Any]] = None,
    before_iters: Optional[Callable[[], Any]] = None,
    after_iters: Optional[Callable[[], Any]] = None,
) -> List[float]:
    if warmup > 0 and before_warmup is not None:
        before_warmup()
    for _ in range(warmup):
        run_once()
    if warmup > 0 and after_warmup is not None:
        after_warmup()

    if before_iters is not None:
        before_iters()
    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - t0)
    if after_iters is not None:
        after_iters()
    return times


def bench_stats(times_s: List[float], *, units_per_iter: float, impl: str) -> BenchStats:
    mean_s = statistics.mean(times_s) if times_s else float("inf")
    units_per_s = (units_per_iter / mean_s) if mean_s > 0 else 0.0
    return BenchStats(impl=impl, times_s=times_s, mean_s=mean_s, units_per_s=units_per_s)


def validation_result(
    *,
    model: str,
    torch_np: np.ndarray,
    gptrs_np: np.ndarray,
    rtol: float,
    atol: float,
    extra: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    ok, max_abs, mean_abs = validate_arrays(torch_np, gptrs_np, rtol=rtol, atol=atol)
    return ValidationResult(
        model=model,
        ok=ok,
        torch_shape=tuple(torch_np.shape),
        gptrs_shape=tuple(gptrs_np.shape),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        extra=extra or {},
    )
