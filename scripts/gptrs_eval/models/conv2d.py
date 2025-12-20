from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import numpy as np

from ..core import BenchResult, RunConfig
from ..gptrs_py import debug_context
from ..runner import bench_stats, time_many, validation_result


def _out_hw(h: int, w: int, k: int, s: int, d: int, p: int) -> Tuple[int, int]:
    eff_k = d * (k - 1) + 1
    out_h = (h + 2 * p - eff_k) // s + 1
    out_w = (w + 2 * p - eff_k) // s + 1
    return int(out_h), int(out_w)


class Conv2dCase:
    name = "conv2d"
    torch_impl_label = "torch:conv2d"

    def supported_workloads(self) -> list[str]:
        return ["validate", "bench", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
        parser.add_argument("--height", type=int, default=56, help="Input height (default: 56)")
        parser.add_argument("--width", type=int, default=56, help="Input width (default: 56)")
        parser.add_argument("--in-ch", type=int, default=64, help="Input channels (default: 64)")
        parser.add_argument("--out-ch", type=int, default=64, help="Output channels (default: 64)")
        parser.add_argument("--kernel", type=int, default=3, help="Kernel size (default: 3)")
        parser.add_argument("--stride", type=int, default=1, help="Stride (default: 1)")
        parser.add_argument("--dilation", type=int, default=1, help="Dilation (default: 1)")
        parser.add_argument("--padding", type=int, default=1, help="Symmetric padding (default: 1)")
        parser.add_argument("--groups", type=int, default=1, help="Groups (default: 1)")

    def _build_inputs(
        self, cfg: RunConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = int(cfg.params.get("batch", 1))
        h = int(cfg.params.get("height", 56))
        w = int(cfg.params.get("width", 56))
        c_in = int(cfg.params.get("in_ch", 64))
        c_out = int(cfg.params.get("out_ch", 64))
        k = int(cfg.params.get("kernel", 3))
        s = int(cfg.params.get("stride", 1))
        d = int(cfg.params.get("dilation", 1))
        p = int(cfg.params.get("padding", 1))
        groups = int(cfg.params.get("groups", 1))

        if batch <= 0 or h <= 0 or w <= 0:
            raise SystemExit("batch/h/w must be > 0")
        if c_in <= 0 or c_out <= 0:
            raise SystemExit("in-ch/out-ch must be > 0")
        if k <= 0 or s <= 0 or d <= 0:
            raise SystemExit("kernel/stride/dilation must be > 0")
        if p < 0:
            raise SystemExit("padding must be >= 0")
        if groups <= 0 or (c_in % groups) != 0 or (c_out % groups) != 0:
            raise SystemExit("groups must divide in-ch and out-ch")

        out_h, out_w = _out_hw(h, w, k, s, d, p)
        if out_h <= 0 or out_w <= 0:
            raise SystemExit(
                f"invalid output size: HxW={h}x{w} k={k} s={s} d={d} p={p} -> out={out_h}x{out_w}"
            )

        np.random.seed(cfg.seed)
        x_nchw = np.random.randn(batch, c_in, h, w).astype(np.float32)
        x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).copy()

        weight = np.random.randn(c_out, c_in // groups, k, k).astype(np.float32)
        bias = np.random.randn(c_out).astype(np.float32)
        return x_nchw, x_nhwc, weight, bias

    def validate(self, cfg: RunConfig):
        import torch

        x_nchw, x_nhwc, weight, bias = self._build_inputs(cfg)

        batch = int(cfg.params.get("batch", 1))
        h = int(cfg.params.get("height", 56))
        w = int(cfg.params.get("width", 56))
        c_in = int(cfg.params.get("in_ch", 64))
        c_out = int(cfg.params.get("out_ch", 64))
        k = int(cfg.params.get("kernel", 3))
        s = int(cfg.params.get("stride", 1))
        d = int(cfg.params.get("dilation", 1))
        p = int(cfg.params.get("padding", 1))
        groups = int(cfg.params.get("groups", 1))

        torch.manual_seed(cfg.seed)
        conv = torch.nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(k, k),
            stride=(s, s),
            padding=(p, p),
            dilation=(d, d),
            groups=groups,
            bias=True,
        ).eval()
        assert conv.bias is not None
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(weight))
            conv.bias.copy_(torch.from_numpy(bias))
            y_torch = conv(torch.from_numpy(x_nchw)).permute(0, 2, 3, 1).contiguous()
        y_torch_np = y_torch.detach().cpu().numpy().astype(np.float32, copy=False)

        import gpt_rs

        gpt_rs.set_backend(cfg.backend)
        w_gpt = gpt_rs.Tensor.from_numpy(weight)
        b_gpt = gpt_rs.Tensor.from_numpy(bias)
        x_gpt = gpt_rs.Tensor.from_numpy(x_nhwc)
        conv_gpt = gpt_rs.vision.Conv2d(
            w_gpt,
            b_gpt,
            kernel=(k, k),
            stride=(s, s),
            dilation=(d, d),
            padding=(p, p, p, p),
            groups=groups,
        )
        with debug_context(cfg.params):
            y_gpt_np = conv_gpt.forward(x_gpt).numpy().astype(np.float32, copy=False)

        extra: Dict[str, Any] = {
            "batch": batch,
            "height": h,
            "width": w,
            "in_ch": c_in,
            "out_ch": c_out,
            "kernel": k,
            "stride": s,
            "dilation": d,
            "padding": p,
            "groups": groups,
        }
        return validation_result(
            model=self.name,
            torch_np=y_torch_np,
            gptrs_np=y_gpt_np,
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

        x_nchw, x_nhwc, weight, bias = self._build_inputs(cfg)
        batch = int(cfg.params.get("batch", 1))
        c_in = int(cfg.params.get("in_ch", 64))
        c_out = int(cfg.params.get("out_ch", 64))
        k = int(cfg.params.get("kernel", 3))
        s = int(cfg.params.get("stride", 1))
        d = int(cfg.params.get("dilation", 1))
        p = int(cfg.params.get("padding", 1))
        groups = int(cfg.params.get("groups", 1))

        conv = torch.nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(k, k),
            stride=(s, s),
            padding=(p, p),
            dilation=(d, d),
            groups=groups,
            bias=True,
        ).eval()
        assert conv.bias is not None
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(weight))
            conv.bias.copy_(torch.from_numpy(bias))

        x_torch = torch.from_numpy(x_nchw)

        import gpt_rs

        gpt_rs.set_backend(cfg.backend)
        w_gpt = gpt_rs.Tensor.from_numpy(weight)
        b_gpt = gpt_rs.Tensor.from_numpy(bias)
        x_gpt = gpt_rs.Tensor.from_numpy(x_nhwc)
        conv_gpt = gpt_rs.vision.Conv2d(
            w_gpt,
            b_gpt,
            kernel=(k, k),
            stride=(s, s),
            dilation=(d, d),
            padding=(p, p, p, p),
            groups=groups,
        )

        def run_torch_once() -> None:
            with torch.no_grad():
                out = conv(x_torch)
            _ = float(out.reshape(-1)[0].item())

        def run_gptrs_once() -> None:
            out = conv_gpt.forward(x_gpt)
            _ = out.shape[0]

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
            extra={
                "height": int(cfg.params.get("height", 56)),
                "width": int(cfg.params.get("width", 56)),
                "in_ch": c_in,
                "out_ch": c_out,
                "kernel": k,
                "stride": s,
                "dilation": d,
                "padding": p,
                "groups": groups,
            },
        )

    def run(self, cfg: RunConfig) -> Dict[str, Any]:
        import gpt_rs

        x_nchw, x_nhwc, weight, bias = self._build_inputs(cfg)
        k = int(cfg.params.get("kernel", 3))
        s = int(cfg.params.get("stride", 1))
        d = int(cfg.params.get("dilation", 1))
        p = int(cfg.params.get("padding", 1))
        groups = int(cfg.params.get("groups", 1))

        gpt_rs.set_backend(cfg.backend)
        w_gpt = gpt_rs.Tensor.from_numpy(weight)
        b_gpt = gpt_rs.Tensor.from_numpy(bias)
        x_gpt = gpt_rs.Tensor.from_numpy(x_nhwc)
        conv_gpt = gpt_rs.vision.Conv2d(
            w_gpt,
            b_gpt,
            kernel=(k, k),
            stride=(s, s),
            dilation=(d, d),
            padding=(p, p, p, p),
            groups=groups,
        )
        with debug_context(cfg.params):
            out = conv_gpt.forward(x_gpt)
        return {
            "model": self.name,
            "impl": f"gpt-rs/{cfg.backend}",
            "input_nhwc_shape": list(x_nhwc.shape),
            "output_nhwc_shape": out.shape,
        }
