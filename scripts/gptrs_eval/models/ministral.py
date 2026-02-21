from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np

from ..core import CliRunResult, RunConfig
from ..gptrs_py import debug_context
from ..registry import get_case_default_params
from ..runner import validation_result

_DEFAULT_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"


def _as_path(value: Any, fallback: Path) -> Path:
    if value is None:
        return fallback
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _resolve_torch_dtype(name: str) -> Any:
    import torch

    mapping: Dict[str, Any] = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        known = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"unsupported torch dtype {name!r}; expected one of: {known}")
    return mapping[name]


def _encode_prompt(tokenizer: Any, prompt: str, max_prompt_tokens: int) -> List[int]:
    prompt_tokens = [int(tok) for tok in tokenizer.encode(prompt, add_special_tokens=False)]
    if max_prompt_tokens > 0 and len(prompt_tokens) > max_prompt_tokens:
        prompt_tokens = prompt_tokens[-max_prompt_tokens:]
    if not prompt_tokens:
        raise ValueError("prompt produced zero tokens; provide a non-empty prompt")
    return prompt_tokens


class MinistralCase:
    name = "ministral_3_3b_instruct_2512"

    def supported_workloads(self) -> list[str]:
        return ["validate", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        defaults = get_case_default_params(self.name)
        prompt_default = str(defaults.get("prompt", "Hello"))
        torch_model_default = str(defaults.get("torch_model", _DEFAULT_MODEL_ID))
        checkpoint_default = _as_path(
            defaults.get("checkpoint"),
            Path("checkpoints/ministral_3_3b_instruct_2512.bin"),
        )
        trust_remote_code_default = bool(defaults.get("trust_remote_code", False))
        torch_dtype_default = str(defaults.get("torch_dtype", "auto"))

        parser.add_argument("--prompt", default=prompt_default, help="Prompt text.")
        parser.add_argument(
            "--torch-model",
            default=torch_model_default,
            help="Hugging Face model id for Torch baseline/tokenizer.",
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=checkpoint_default,
            help="Checkpoint path.",
        )
        parser.add_argument(
            "--max-prompt-tokens",
            type=int,
            default=int(defaults.get("max_prompt_tokens", 64)),
            help="Keep only the last N prompt tokens before validation/run (default: 64).",
        )
        parser.add_argument(
            "--generate-tokens",
            type=int,
            default=int(defaults.get("generate_tokens", 0)),
            help="Greedy tokens to validate beyond the prompt (default: 0).",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.8,
            help="Sampling temperature for `run` workload (default: 0.8).",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Max tokens to generate for `run` workload (default: 128).",
        )
        parser.add_argument(
            "--torch-dtype",
            default=torch_dtype_default,
            choices=["auto", "float16", "bfloat16", "float32"],
            help="Torch dtype for baseline model loading (default: auto).",
        )
        parser.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=trust_remote_code_default,
            help="Allow loading Hugging Face model/tokenizer with remote code (default: false).",
        )

    def _build_gpt_rs(self, cfg: RunConfig) -> Any:
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
        checkpoint = Path(cfg.params["checkpoint"])
        return gpt.load_model(str(checkpoint))

    def _build_hf(self, cfg: RunConfig) -> Tuple[Any, Any]:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        torch_model_id = str(cfg.params.get("torch_model", _DEFAULT_MODEL_ID))
        trust_remote_code = bool(cfg.params.get("trust_remote_code", False))
        torch_dtype = _resolve_torch_dtype(str(cfg.params.get("torch_dtype", "auto")))

        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype

        config = AutoConfig.from_pretrained(torch_model_id, trust_remote_code=trust_remote_code)
        model_type = str(getattr(config, "model_type", "")).lower()
        if model_type == "mistral3":
            from transformers import Mistral3ForConditionalGeneration

            torch_model = cast(
                Any,
                Mistral3ForConditionalGeneration.from_pretrained(torch_model_id, **model_kwargs),
            )
        else:
            torch_model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(torch_model_id, **model_kwargs),
            )
        torch_model.eval()
        torch_model.to(cfg.torch_device)

        tokenizer = AutoTokenizer.from_pretrained(
            torch_model_id,
            trust_remote_code=trust_remote_code,
        )
        return tokenizer, torch_model

    def validate(self, cfg: RunConfig):
        import torch

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        prompt = str(cfg.params.get("prompt", "Hello"))
        generate_tokens = int(cfg.params.get("generate_tokens", 0))
        max_prompt_tokens = int(cfg.params.get("max_prompt_tokens", 64))
        rs_model = self._build_gpt_rs(cfg)
        tokenizer, torch_model = self._build_hf(cfg)

        prompt_tokens = _encode_prompt(tokenizer, prompt, max_prompt_tokens)

        def hf_logits(tokens: List[int]) -> np.ndarray:
            input_ids = torch.tensor(tokens, dtype=torch.long, device=cfg.torch_device).unsqueeze(0)
            with torch.no_grad():
                out = torch_model(input_ids=input_ids)
            return out.logits[0, -1].float().detach().cpu().numpy().astype(np.float32, copy=False)

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
            "torch_model": str(cfg.params.get("torch_model", _DEFAULT_MODEL_ID)),
        }
        res = validation_result(
            model=self.name,
            torch_np=hf_row,
            gptrs_np=rs_row,
            rtol=cfg.rtol,
            atol=cfg.atol,
            extra=extra,
        )

        ok = bool(ok_steps == generate_tokens) and (extra["hf_top1"] == extra["gpt_rs_top1"])
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

    def bench(self, cfg: RunConfig):  # pragma: no cover - not part of current milestone
        raise SystemExit(f"bench workload is not implemented for model {self.name!r}")

    def run(self, cfg: RunConfig) -> CliRunResult:
        prompt = str(cfg.params.get("prompt", "Hello"))
        max_prompt_tokens = int(cfg.params.get("max_prompt_tokens", 64))
        temperature = float(cfg.params.get("temperature", 0.8))
        max_tokens = int(cfg.params.get("max_tokens", 128))

        rs_model = self._build_gpt_rs(cfg)
        tokenizer, _torch_model = self._build_hf(cfg)

        prompt_tokens = _encode_prompt(tokenizer, prompt, max_prompt_tokens)

        t0 = time.perf_counter()
        with debug_context(cfg.params):
            out_tokens = rs_model.generate_tokens(
                prompt_tokens,
                max_tokens,
                temperature=temperature,
                kv_cache=True,
            )
        wall_s = time.perf_counter() - t0

        print(tokenizer.decode(out_tokens, skip_special_tokens=True))
        return CliRunResult(
            model=self.name,
            impl=f"gpt-rs/{cfg.backend}",
            exit_code=0,
            wall_s=wall_s,
        )
