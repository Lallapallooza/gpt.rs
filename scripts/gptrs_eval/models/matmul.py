from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import numpy as np

from ..core import BenchResult, RunConfig
from ..gptrs_py import debug_context
from ..runner import bench_stats, time_many, validation_result


class MatmulCase:
    name = "matmul"
    torch_impl_label = "torch:matmul"

    def supported_workloads(self) -> list[str]:
        return ["validate", "bench", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--m", type=int, default=1024, help="M dimension (default: 1024)")
        parser.add_argument("--n", type=int, default=1024, help="N dimension (default: 1024)")
        parser.add_argument("--k", type=int, default=1024, help="K dimension (default: 1024)")

    def _build_inputs(self, cfg: RunConfig) -> Tuple[np.ndarray, np.ndarray]:
        m = int(cfg.params.get("m", 1024))
        n = int(cfg.params.get("n", 1024))
        k = int(cfg.params.get("k", 1024))

        if m <= 0 or n <= 0 or k <= 0:
            raise SystemExit("m/n/k must be > 0")

        np.random.seed(cfg.seed)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        return a, b

    def validate(self, cfg: RunConfig):
        import torch

        a, b = self._build_inputs(cfg)
        m, k = a.shape
        n = b.shape[1]

        expected = torch.matmul(torch.from_numpy(a), torch.from_numpy(b))
        expected_np = expected.detach().cpu().numpy().astype(np.float32, copy=False)

        import gpt_rs

        gpt_rs.set_backend(cfg.backend)
        a_gpt = gpt_rs.Tensor.from_numpy(a)
        b_gpt = gpt_rs.Tensor.from_numpy(b)
        with debug_context(cfg.params):
            out = gpt_rs.functional.matmul(a_gpt, b_gpt)
            actual_np = out.numpy().astype(np.float32, copy=False)

        extra: Dict[str, Any] = {"m": m, "n": n, "k": k}
        return validation_result(
            model=self.name,
            torch_np=expected_np,
            gptrs_np=actual_np,
            rtol=cfg.rtol,
            atol=cfg.atol,
            extra=extra,
        )

    def bench(self, cfg: RunConfig) -> BenchResult:
        import torch

        try:
            torch.set_num_threads(cfg.threads)
            torch.set_num_interop_threads(cfg.threads)
        except RuntimeError:
            pass

        a, b = self._build_inputs(cfg)
        m, k = a.shape
        n = b.shape[1]

        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)

        import gpt_rs

        gpt_rs.set_backend(cfg.backend)
        a_gpt = gpt_rs.Tensor.from_numpy(a)
        b_gpt = gpt_rs.Tensor.from_numpy(b)

        def run_torch_once() -> None:
            out = torch.matmul(a_t, b_t)
            _ = float(out.reshape(-1)[0].item())

        def run_gptrs_once() -> None:
            out = gpt_rs.functional.matmul(a_gpt, b_gpt)
            _ = out.shape[0]

        units = float(m)
        torch_times = time_many(run_torch_once, warmup=cfg.warmup, iters=cfg.iters)
        with debug_context(cfg.params) as dbg:
            gptrs_times = time_many(
                run_gptrs_once,
                warmup=cfg.warmup,
                iters=cfg.iters,
                before_warmup=lambda: dbg.push_section("warmup"),
                after_warmup=dbg.pop_section,
                before_iters=lambda: dbg.push_section("measure"),
                after_iters=dbg.pop_section,
            )

        return BenchResult(
            model=self.name,
            threads=cfg.threads,
            unit_label="rows",
            units_per_iter=units,
            gptrs=bench_stats(gptrs_times, units_per_iter=units, impl=f"gpt-rs/{cfg.backend}"),
            torch=bench_stats(torch_times, units_per_iter=units, impl=self.torch_impl_label),
            extra={"m": m, "n": n, "k": k},
        )

    def run(self, cfg: RunConfig) -> Dict[str, Any]:
        import gpt_rs

        a, b = self._build_inputs(cfg)
        m, k = a.shape
        n = b.shape[1]

        gpt_rs.set_backend(cfg.backend)
        a_gpt = gpt_rs.Tensor.from_numpy(a)
        b_gpt = gpt_rs.Tensor.from_numpy(b)
        with debug_context(cfg.params):
            out = gpt_rs.functional.matmul(a_gpt, b_gpt)
        return {
            "model": self.name,
            "impl": f"gpt-rs/{cfg.backend}",
            "shape": [m, k, n],
            "output_shape": out.shape,
        }
