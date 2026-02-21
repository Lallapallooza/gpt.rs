from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np

from ..core import BenchResult, CliRunResult, RunConfig
from ..gptrs_py import debug_context
from ..runner import bench_stats, time_many, validation_result


def _as_path(value: Any, fallback: Path) -> Path:
    if value is None:
        return fallback
    if isinstance(value, Path):
        return value
    return Path(str(value))


class Gpt2Case:
    name = "gpt2"

    def supported_workloads(self) -> list[str]:
        return ["validate", "bench", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        from ..registry import get_case_default_params

        defaults: Dict[str, Any] = get_case_default_params(self.name)
        prompt_default = str(defaults.get("prompt", "Hello"))
        torch_model_default = str(defaults.get("torch_model", "gpt2"))
        checkpoint_default = _as_path(defaults.get("checkpoint"), Path("checkpoints/gpt2.bin"))
        tokenizer_default = _as_path(defaults.get("tokenizer"), Path("configs/gpt2_tokenizer.json"))

        parser.add_argument("--prompt", default=prompt_default, help="Prompt text.")
        parser.add_argument(
            "--torch-model",
            default=torch_model_default,
            help="HF model id for torch baseline.",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.8,
            help="Sampling temperature for gpt-rs run (default: 0.8).",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Max tokens for gpt-rs run (default: 128).",
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=checkpoint_default,
            help="Checkpoint path.",
        )
        parser.add_argument(
            "--tokenizer",
            type=Path,
            default=tokenizer_default,
            help="Tokenizer JSON.",
        )
        parser.add_argument(
            "--generate-tokens",
            type=int,
            default=0,
            help="Greedy tokens to validate beyond the prompt (default: 0).",
        )
        parser.add_argument(
            "--bench-tokens",
            type=int,
            nargs="+",
            default=[1, 64],
            help="Max tokens to generate for benchmark runs.",
        )
        parser.add_argument(
            "--kv-cache",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable kv-cache for gpt-rs bench (default: true).",
        )

    def _build_gpt_rs(self, cfg: RunConfig) -> Tuple[Any, Any]:
        try:
            import gpt_rs
        except ImportError as err:
            raise SystemExit(
                "gpt_rs not installed. Install via:\n"
                "  pip install maturin\n"
                "  cd crates/gpt-rs-py && maturin develop --release --features faer\n"
            ) from err

        gpt = cast(Any, gpt_rs)
        gpt.set_backend(cfg.backend)

        tokenizer_path = Path(cfg.params["tokenizer"])
        checkpoint = Path(cfg.params["checkpoint"])

        tokenizer = gpt.Tokenizer.from_file(str(tokenizer_path))
        model = gpt.load_model(str(checkpoint))
        return tokenizer, model

    def validate(self, cfg: RunConfig):
        import torch
        from transformers import GPT2LMHeadModel

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        prompt = str(cfg.params.get("prompt", "Hello"))
        torch_model_id = str(cfg.params.get("torch_model", "gpt2"))
        generate_tokens = int(cfg.params.get("generate_tokens", 0))

        rs_tokenizer, rs_model = self._build_gpt_rs(cfg)
        prompt_tokens: List[int] = list(rs_tokenizer.encode(prompt))

        torch_model = GPT2LMHeadModel.from_pretrained(torch_model_id).eval()

        def hf_logits(tokens: List[int]) -> np.ndarray:
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = torch_model(input_ids=input_ids)
            return out.logits[0, -1].detach().cpu().numpy().astype(np.float32, copy=False)

        hf_toks = list(prompt_tokens)
        rs_toks = list(prompt_tokens)

        hf_row = hf_logits(hf_toks)
        with debug_context(cfg.params):
            rs_row = rs_model.logits(rs_toks).astype(np.float32, copy=False)

            ok_steps = 0
            for _step in range(generate_tokens):
                next_hf = int(np.argmax(hf_row))
                next_rs = int(np.argmax(rs_row))
                if next_hf != next_rs:
                    break
                ok_steps += 1
                hf_toks.append(next_hf)
                rs_toks.append(next_rs)
                hf_row = hf_logits(hf_toks)
                rs_row = rs_model.logits(rs_toks).astype(np.float32, copy=False)

        extra = {
            "prompt_len": len(prompt_tokens),
            "validated_generate_tokens": ok_steps,
            "requested_generate_tokens": generate_tokens,
            "hf_top1": int(np.argmax(hf_row)),
            "gpt_rs_top1": int(np.argmax(rs_row)),
        }
        res = validation_result(
            model=self.name,
            torch_np=hf_row,
            gptrs_np=rs_row,
            rtol=cfg.rtol,
            atol=cfg.atol,
            extra=extra,
        )
        ok = bool(ok_steps == generate_tokens) and (
            int(np.argmax(hf_row)) == int(np.argmax(rs_row))
        )
        if ok == res.ok:
            return res
        return res.__class__(
            model=res.model,
            ok=ok,
            torch_shape=res.torch_shape,
            gptrs_shape=res.gptrs_shape,
            max_abs_diff=res.max_abs_diff,
            mean_abs_diff=res.mean_abs_diff,
            extra=res.extra,
        )

    def bench(self, cfg: RunConfig) -> BenchResult:
        import torch
        from transformers import AutoModelForCausalLM

        prompt = str(cfg.params.get("prompt", "Hello"))
        torch_model_id = str(cfg.params.get("torch_model", "gpt2"))
        kv_cache = bool(cfg.params.get("kv_cache", True))
        max_tokens = int(cfg.params.get("bench_tokens", 64))

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Configure torch threads before model load.
        try:
            torch.set_num_threads(cfg.threads)
            torch.set_num_interop_threads(cfg.threads)
        except RuntimeError:
            pass

        rs_tokenizer, rs_model = self._build_gpt_rs(cfg)
        prompt_tokens: List[int] = list(rs_tokenizer.encode(prompt))

        torch_model = AutoModelForCausalLM.from_pretrained(torch_model_id).eval()
        input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = int(getattr(torch_model.config, "eos_token_id", 50256))

        @torch.no_grad()
        def torch_once() -> None:
            _ = torch_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=pad_token_id,
            )

        def gptrs_once() -> None:
            out_tokens = rs_model.generate_tokens(
                prompt_tokens,
                max_tokens,
                temperature=1.0,
                kv_cache=kv_cache,
            )
            _ = int(out_tokens[-1])

        torch_times = time_many(torch_once, warmup=cfg.warmup, iters=cfg.iters)
        with debug_context(cfg.params) as dbg:
            gptrs_times = time_many(
                gptrs_once,
                warmup=cfg.warmup,
                iters=cfg.iters,
                before_warmup=lambda: dbg.push_section("warmup"),
                after_warmup=dbg.pop_section,
                before_iters=lambda: dbg.push_section("measure"),
                after_iters=dbg.pop_section,
            )

        units = float(max_tokens)
        return BenchResult(
            model=self.name,
            threads=cfg.threads,
            unit_label="tokens",
            units_per_iter=units,
            gptrs=bench_stats(gptrs_times, units_per_iter=units, impl=f"gpt-rs/{cfg.backend}"),
            torch=bench_stats(torch_times, units_per_iter=units, impl=f"torch:{torch_model_id}"),
        )

    def run(self, cfg: RunConfig) -> CliRunResult:
        prompt = str(cfg.params.get("prompt", "Hello"))
        temperature = float(cfg.params.get("temperature", 0.8))
        max_tokens = int(cfg.params.get("max_tokens", 128))
        kv_cache = bool(cfg.params.get("kv_cache", True))

        rs_tokenizer, rs_model = self._build_gpt_rs(cfg)
        prompt_tokens: List[int] = list(rs_tokenizer.encode(prompt))

        t0 = time.perf_counter()
        with debug_context(cfg.params):
            out_tokens = rs_model.generate_tokens(
                prompt_tokens,
                max_tokens,
                temperature=temperature,
                kv_cache=kv_cache,
            )
        wall_s = time.perf_counter() - t0

        print(rs_tokenizer.decode(out_tokens))

        return CliRunResult(
            model=self.name,
            impl=f"gpt-rs/{cfg.backend}",
            exit_code=0,
            wall_s=wall_s,
        )
