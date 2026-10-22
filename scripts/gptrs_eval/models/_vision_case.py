from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from ..core import BenchResult, CliRunResult, RunConfig
from ..gptrs_py import debug_context, hf_model_dir, load_gpt_rs
from ..runner import bench_stats, time_many, validation_result


def torch_to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


class VisionCaseBase:
    name: str

    def supported_workloads(self) -> list[str]:
        return ["validate", "bench", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        from ..registry import get_case_default_params

        defaults = get_case_default_params(self.name)
        parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
        parser.add_argument(
            "--image-size",
            type=int,
            default=224,
            help="Input image size H=W (default: 224)",
        )
        parser.add_argument(
            "--hf-model",
            default=defaults.get("hf_model"),
            help="Hugging Face model id or local directory, loaded by gpt-rs and by Torch",
        )

    def _torch_impl_label(self, cfg: RunConfig) -> str:
        return f"transformers:{cfg.params['hf_model']}"

    def _build_torch_model(self, cfg: RunConfig) -> Any:
        import torch
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(
            str(hf_model_dir(cfg)), dtype=torch.float32
        )
        return model.eval()

    def _torch_logits(self, model: Any, pixel_values: Any) -> Any:
        return model(pixel_values=pixel_values).logits

    def _build_models(self, cfg: RunConfig) -> tuple[Any, Any]:
        return self._build_torch_model(cfg), load_gpt_rs(cfg)

    def validate(self, cfg: RunConfig):
        import torch

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        torch_model, model_gpt = self._build_models(cfg)

        batch = int(cfg.params.get("batch", 1))
        image_size = int(cfg.params.get("image_size", 224))

        # Two inputs: the first call records the forward, the second runs its program again.
        inputs = [
            np.random.randn(batch, 3, image_size, image_size).astype(np.float32) for _ in range(2)
        ]

        with torch.no_grad():
            torch_logits = np.concatenate(
                [
                    torch_to_numpy(self._torch_logits(torch_model, torch.from_numpy(x)))
                    for x in inputs
                ]
            )

        with debug_context(cfg.params):
            gpt_logits = np.concatenate(
                [model_gpt.forward_vision(x).astype(np.float32, copy=False) for x in inputs]
            )

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
                out = self._torch_logits(torch_model, x_torch)
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
            torch=bench_stats(torch_times, units_per_iter=units, impl=self._torch_impl_label(cfg)),
        )

    def run(self, cfg: RunConfig) -> CliRunResult:
        import torch

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        model_gpt = load_gpt_rs(cfg)

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
