from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from ..core import BenchResult, CliRunResult, RunConfig
from ..gptrs_py import debug_context
from ..runner import bench_stats, time_many, validation_result


def torch_to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


class VisionCaseBase:
    name: str
    torch_impl_label: str
    checkpoint_default: Any = None

    def supported_workloads(self) -> list[str]:
        return ["validate", "bench", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        from pathlib import Path

        parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
        parser.add_argument(
            "--image-size",
            type=int,
            default=224,
            help="Input image size H=W (default: 224)",
        )
        default_ckpt = (
            self.checkpoint_default
            if self.checkpoint_default is not None
            else Path(f"checkpoints/{self.name}.bin")
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=default_ckpt,
            help="Checkpoint path.",
        )

    def _build_torch_model(self, tvm: Any) -> Any:
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
                "  cd crates/gpt-rs-py && maturin develop --release --features faer\n"
            ) from err

        gpt_rs.set_backend(cfg.backend)

        torch_model = self._build_torch_model(tvm).eval()
        checkpoint = cfg.params.get("checkpoint")
        if checkpoint is None:
            raise SystemExit("missing --checkpoint")
        model_gpt = gpt_rs.load_model(str(checkpoint))
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

        with torch.no_grad():
            torch_logits = torch_to_numpy(torch_model(x_torch))

        with debug_context(cfg.params):
            gpt_logits = model_gpt.forward_vision(x).astype(np.float32, copy=False)

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

        def run_torch_once() -> None:
            with torch.no_grad():
                out = torch_model(x_torch)
            _ = float(out.reshape(-1)[0].item())

        def run_gptrs_once() -> None:
            out = model_gpt.forward_vision(x)
            _ = float(out.reshape(-1)[0])

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

        t0 = time.perf_counter()
        with debug_context(cfg.params):
            out = model_gpt.forward_vision(x)
            _ = float(out.reshape(-1)[0])
        wall_s = time.perf_counter() - t0

        return CliRunResult(
            model=self.name,
            impl=f"gpt-rs/{cfg.backend}",
            exit_code=0,
            wall_s=wall_s,
            extra={"batch": batch, "image_size": image_size},
        )
