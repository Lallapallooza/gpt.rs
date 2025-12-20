from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .core import BenchStats, TraceDiff, TraceResult, ValidationResult


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


def compare_traces(
    torch_trace: Dict[str, np.ndarray],
    gptrs_trace: Dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    model: str,
    max_lines: int = 200,
    stop_on_first_mismatch: bool = False,
) -> TraceResult:
    ordered_keys = list(torch_trace.keys())
    missing_in_gptrs = [k for k in ordered_keys if k not in gptrs_trace]
    missing_in_torch = [k for k in gptrs_trace.keys() if k not in torch_trace]

    diffs: List[TraceDiff] = []
    ok = True
    printed = 0

    for key in ordered_keys:
        if key not in gptrs_trace:
            continue
        torch_np = torch_trace[key]
        gpt_np = gptrs_trace[key]
        if torch_np.shape != gpt_np.shape:
            ok = False
            diffs.append(
                TraceDiff(
                    key=key,
                    shape=tuple(torch_np.shape),
                    max_abs_diff=float("inf"),
                    mean_abs_diff=float("inf"),
                    allclose=False,
                )
            )
            if stop_on_first_mismatch:
                break
            continue

        close, max_abs, mean_abs = validate_arrays(torch_np, gpt_np, rtol=rtol, atol=atol)
        if not close:
            ok = False
        diffs.append(
            TraceDiff(
                key=key,
                shape=tuple(torch_np.shape),
                max_abs_diff=max_abs,
                mean_abs_diff=mean_abs,
                allclose=close,
            )
        )
        if printed < max_lines and (not close or key == "logits"):
            printed += 1
        if stop_on_first_mismatch and not close:
            break

    if missing_in_gptrs:
        ok = False

    return TraceResult(
        model=model,
        ok=ok,
        diffs=diffs,
        missing_in_gptrs=missing_in_gptrs,
        missing_in_torch=missing_in_torch,
    )


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
