from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import numpy as np

from ..core import BenchResult, CliRunResult, RunConfig
from ..gptrs_py import debug_context
from ..runner import bench_stats, compare_traces, time_many, validation_result


def torch_to_nhwc(x: Any) -> Any:
    return x.permute(0, 2, 3, 1).contiguous()


def torch_to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


class VisionCaseBase:
    name: str
    torch_impl_label: str

    def supported_workloads(self) -> list[str]:
        return ["validate", "trace", "bench", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
        parser.add_argument(
            "--image-size",
            type=int,
            default=224,
            help="Input image size H=W (default: 224)",
        )

    def _build_torch_model(self, tvm: Any) -> Any:
        raise NotImplementedError

    def _build_weight_arrays(self, torch_model: Any) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _build_gpt_rs_model(self, gpt_rs: Any, weight_tensors: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def _trace_torch(self, torch_model: Any, x_torch: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_models(self, cfg: RunConfig) -> tuple[Any, Any]:
        try:
            from torchvision import models as tvm  # type: ignore[import-not-found, import-untyped]
        except ImportError as err:
            raise SystemExit("torchvision is required (pip install torchvision).") from err

        try:
            import gpt_rs
        except ImportError as err:
            raise SystemExit(
                "gpt_rs not installed. Install via:\n"
                "  pip install maturin\n"
                "  cd crates/gpt-rs-py && maturin develop --release --features faer --skip-install\n"
            ) from err

        gpt_rs.set_backend(cfg.backend)

        torch_model = self._build_torch_model(tvm).eval()
        weight_arrays = self._build_weight_arrays(torch_model)
        weight_tensors = {k: gpt_rs.Tensor.from_numpy(v) for k, v in weight_arrays.items()}
        model_gpt = self._build_gpt_rs_model(gpt_rs, weight_tensors)
        return torch_model, model_gpt

    def validate(self, cfg: RunConfig):
        import torch

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        torch_model, model_gpt = self._build_models(cfg)

        batch = int(cfg.params.get("batch", 1))
        image_size = int(cfg.params.get("image_size", 224))

        x = np.random.randn(batch, 3, image_size, image_size).astype(np.float32)
        x_torch = torch.from_numpy(x)
        import gpt_rs

        x_gpt = gpt_rs.Tensor.from_numpy(x)

        with torch.no_grad():
            torch_logits = torch_to_numpy(torch_model(x_torch))

        with debug_context(cfg.params):
            gpt_logits = model_gpt.forward(x_gpt).numpy().astype(np.float32, copy=False)

        extra = {
            "batch": batch,
            "image_size": image_size,
            "torch_top1": int(np.argmax(torch_logits[0])),
            "gpt_rs_top1": int(np.argmax(gpt_logits[0])),
        }
        return validation_result(
            model=self.name,
            torch_np=torch_logits,
            gptrs_np=gpt_logits,
            rtol=cfg.rtol,
            atol=cfg.atol,
            extra=extra,
        )

    def trace(self, cfg: RunConfig):
        import torch

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        torch_model, model_gpt = self._build_models(cfg)

        batch = int(cfg.params.get("batch", 1))
        image_size = int(cfg.params.get("image_size", 224))
        max_lines = int(cfg.params.get("max_lines", 200))
        stop_first = bool(cfg.params.get("stop_on_first_mismatch", False))

        x = np.random.randn(batch, 3, image_size, image_size).astype(np.float32)
        x_torch = torch.from_numpy(x)
        import gpt_rs

        x_gpt = gpt_rs.Tensor.from_numpy(x)

        with torch.no_grad():
            torch_trace_raw = self._trace_torch(torch_model, x_torch)
        torch_trace = {k: torch_to_numpy(v) for k, v in torch_trace_raw.items()}

        with debug_context(cfg.params):
            gpt_trace_raw = model_gpt.forward_trace(x_gpt)
            gpt_trace = {
                k: v.numpy().astype(np.float32, copy=False) for k, v in gpt_trace_raw.items()
            }

        return compare_traces(
            torch_trace,
            gpt_trace,
            rtol=cfg.rtol,
            atol=cfg.atol,
            model=self.name,
            max_lines=max_lines,
            stop_on_first_mismatch=stop_first,
        )

    def bench(self, cfg: RunConfig) -> BenchResult:
        import torch

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        try:
            torch.set_num_threads(cfg.threads)
            torch.set_num_interop_threads(cfg.threads)
        except RuntimeError:
            pass

        if cfg.torch_device != "cpu":
            raise SystemExit("vision bench supports torch_device=cpu only (for now)")

        torch_model, model_gpt = self._build_models(cfg)

        batch = int(cfg.params.get("batch", 1))
        image_size = int(cfg.params.get("image_size", 224))

        x = np.random.randn(batch, 3, image_size, image_size).astype(np.float32)
        x_torch = torch.from_numpy(x)
        import gpt_rs

        x_gpt = gpt_rs.Tensor.from_numpy(x)

        def run_torch_once() -> None:
            with torch.no_grad():
                out = torch_model(x_torch)
            _ = float(out.reshape(-1)[0].item())

        def run_gptrs_once() -> None:
            out = model_gpt.forward(x_gpt)
            out_np = out.numpy()
            _ = float(out_np.reshape(-1)[0])

        units = float(batch)
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
            unit_label="images",
            units_per_iter=units,
            gptrs=bench_stats(gptrs_times, units_per_iter=units, impl=f"gpt-rs/{cfg.backend}"),
            torch=bench_stats(torch_times, units_per_iter=units, impl=self.torch_impl_label),
        )

    def run(self, cfg: RunConfig) -> CliRunResult:
        import torch

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        _torch_model, model_gpt = self._build_models(cfg)

        batch = int(cfg.params.get("batch", 1))
        image_size = int(cfg.params.get("image_size", 224))

        x = np.random.randn(batch, 3, image_size, image_size).astype(np.float32)
        import gpt_rs

        x_gpt = gpt_rs.Tensor.from_numpy(x)

        t0 = time.perf_counter()
        with debug_context(cfg.params):
            out = model_gpt.forward(x_gpt)
            out_np = out.numpy()
            _ = float(out_np.reshape(-1)[0])
        wall_s = time.perf_counter() - t0

        return CliRunResult(
            model=self.name,
            impl=f"gpt-rs/{cfg.backend}",
            exit_code=0,
            wall_s=wall_s,
            extra={"batch": batch, "image_size": image_size},
        )
