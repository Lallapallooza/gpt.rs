from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..checkpoint import hf_text_tensor_names, read_config
from ..core import (
    BenchResult,
    BenchStats,
    CliRunResult,
    RunConfig,
    ValidationResult,
    as_path,
    encode_prompt,
)
from ..gptrs_py import debug_context, gpt_rs_module, load_gpt_rs
from ..registry import get_case_default_params
from ..runner import bench_stats, time_many, validation_result
from ..safetensors_source import SafetensorsSource, resolve_model_dir

_DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"
_DEFAULT_MAX_REL_DIFF = 1e-3


def _hf_text_config(model_dir: Path, num_layers: int) -> Any:
    """HF text config truncated to `num_layers`, tying lm_head as the published checkpoint does."""
    from transformers import Qwen3_5TextConfig

    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = dict(raw.get("text_config", raw))
    cfg = Qwen3_5TextConfig(**text)
    if num_layers > cfg.num_hidden_layers:
        raise ValueError(
            f"checkpoint has {num_layers} layers, but the HF model has {cfg.num_hidden_layers}"
        )
    cfg.num_hidden_layers = num_layers
    cfg.layer_types = list(cfg.layer_types or ())[:num_layers]
    cfg._attn_implementation = "sdpa"
    # Multimodal checkpoints declare weight tying at the top level of config.json.
    cfg.tie_word_embeddings = bool(raw.get("tie_word_embeddings", False))
    return cfg


def hf_reference_logits(model_dir: Path, tokens: list[int], num_layers: int) -> np.ndarray:
    """Logits `[T, vocab]` from Hugging Face's own Qwen3.5 modules, one decoder layer at a time.

    Each `Qwen3_5DecoderLayer` is built on the meta device, filled from the safetensors shards,
    applied and released, so only one layer's float32 weights are in memory at a time.
    `num_layers` truncates the decoder stack to match truncated gpt-rs exports.
    """

    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        Qwen3_5TextRotaryEmbedding,
    )

    dtype = torch.float32
    cfg = _hf_text_config(model_dir, num_layers)
    source = SafetensorsSource(model_dir)
    names = hf_text_tensor_names(source.weight_map)

    with torch.no_grad():
        ids = torch.tensor(tokens, dtype=torch.long)
        embed = source.get(names["model.embed_tokens.weight"])
        hidden = embed.index_select(0, ids).to(dtype).unsqueeze(0)
        del embed
        positions = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)
        rotary = Qwen3_5TextRotaryEmbedding(cfg)
        cos, sin = rotary(hidden, positions)

        for layer_idx in range(num_layers):
            with torch.device("meta"):
                layer = Qwen3_5DecoderLayer(cfg, layer_idx)
            weights = {
                name: source.get(names[f"model.layers.{layer_idx}.{name}"]).to(dtype)
                for name in layer.state_dict()
            }
            layer.load_state_dict(weights, strict=True, assign=True)
            layer.eval()
            hidden = layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=None,
                position_ids=positions,
            )
            del layer, weights

        norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.load_state_dict({"weight": source.get(names["model.norm.weight"]).to(dtype)})
        normed = norm(hidden)[0]

        tied = bool(cfg.tie_word_embeddings)
        lm_head = source.get(names["model.embed_tokens.weight" if tied else "lm_head.weight"])
        rows: list[Any] = []
        block = 16384
        for start in range(0, lm_head.shape[0], block):
            rows.append(normed @ lm_head[start : start + block].to(dtype).T)
        return torch.cat(rows, dim=-1).float().numpy()


def _prompt_text(tokenizer: Any, prompt: str, chat: bool) -> str:
    if not chat:
        return prompt
    text: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text


def _repeat_to_length(tokenizer: Any, text: str, count: int) -> list[int]:
    """The first `count` ids of copies of `text` joined by spaces."""

    def encode(value: str) -> list[int]:
        return [int(t) for t in tokenizer.encode(value, add_special_tokens=False)]

    if not encode(text):
        raise ValueError("prompt produced zero tokens. Provide a non-empty prompt.")
    repeats = 1
    while True:
        ids = encode(" ".join([text] * repeats))
        if len(ids) >= count:
            return ids[:count]
        repeats *= 2


# Strings that exercise the pre-tokenizer split, NFC and multi-byte UTF-8. The chat-templated
# prompt adds special tokens.
_TOKENIZER_SAMPLES = [
    "Hello, world! It's 2026.",
    "  leading,   multiple   spaces\n\n\ttabs and trailing  ",
    "Numbers 1234567 and 3.14159; symbols #@$%^&*()",
    "def f(x):\n    return x ** 2  # comment",
    "日本語のテキスト, émigré café, 🦀 emoji",
]


def _tokenizer_mismatches(model_dir: Path, tokenizer: Any, texts: list[str]) -> list[str]:
    """Texts that the gpt-rs tokenizer, loaded from `tokenizer.json`, encodes unlike HF."""
    rs_tokenizer = gpt_rs_module().Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return [
        text
        for text in texts
        if [int(t) for t in rs_tokenizer.encode(text)]
        != [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    ]


class Qwen35Case:
    name = "qwen3_8_27b"

    def supported_workloads(self) -> list[str]:
        return ["validate", "bench", "run"]

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        defaults = get_case_default_params(self.name)
        parser.add_argument("--prompt", default=str(defaults.get("prompt", "Hello")))
        parser.add_argument(
            "--torch-model",
            default=str(defaults.get("torch_model", _DEFAULT_MODEL_ID)),
            help="Hugging Face model id or local directory (reference weights and tokenizer).",
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=as_path(defaults.get("checkpoint"), Path("checkpoints/qwen3_8_27b.bin")),
        )
        parser.add_argument(
            "--max-prompt-tokens",
            type=int,
            default=int(defaults.get("max_prompt_tokens", 64)),
            help="Keep only the last N prompt tokens. 0 keeps all (default: 64).",
        )
        parser.add_argument(
            "--prompt-tokens",
            type=int,
            default=0,
            help="Build an exact N-token prompt by repeating --prompt. Overrides "
            "--max-prompt-tokens. Cannot be combined with --chat (default: 0 = off).",
        )
        parser.add_argument(
            "--generate-tokens",
            type=int,
            default=int(defaults.get("generate_tokens", 0)),
            help="Greedy continuation tokens to validate against the reference (default: 0).",
        )
        parser.add_argument(
            "--chat",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Wrap the prompt with the model chat template (thinking disabled).",
        )
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=64)
        parser.add_argument(
            "--bench-tokens",
            type=int,
            default=32,
            help="Decode steps timed by the bench workload (default: 32).",
        )
        parser.add_argument(
            "--matmul-input-dtype",
            choices=["f32", "bf16"],
            default=None,
            help="gpt-rs projection matmul input precision (default: the checkpoint runtime "
            "config, else f32).",
        )
        parser.add_argument(
            "--max-rel-diff",
            type=float,
            default=_DEFAULT_MAX_REL_DIFF,
            help="Largest relative logit error that validate accepts (default: "
            f"{_DEFAULT_MAX_REL_DIFF:g}). The error is max|gpt-rs - HF| over all prompt logits, "
            "divided by max|HF logit|. validate also requires the same last-position argmax, the "
            "reference argmax at every --generate-tokens step, and a gpt-rs tokenizer that "
            "matches HF on the prompt and the built-in samples.",
        )
        parser.add_argument(
            "--torch-compile",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Compile the Torch baseline forward with torch.compile (inductor) for bench.",
        )

    def _gpt_rs_model(self, cfg: RunConfig) -> Any:
        return load_gpt_rs(cfg, matmul_input_dtype=cfg.params.get("matmul_input_dtype"))

    def _num_layers(self, cfg: RunConfig) -> int:
        header = read_config(Path(cfg.params["checkpoint"]))
        if header.get("kind") != "qwen3_5":
            raise ValueError(f"checkpoint kind {header.get('kind')!r} is not qwen3_5")
        return len(header["config"]["layer_types"])

    def _prompt(self, cfg: RunConfig, model_dir: Path) -> tuple[Any, str, list[int]]:
        """HF tokenizer, the prompt text, chat-templated when requested, and the prompt ids."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        chat = bool(cfg.params.get("chat"))
        text = _prompt_text(tokenizer, str(cfg.params["prompt"]), chat)
        count = int(cfg.params.get("prompt_tokens", 0))
        if count > 0 and chat:
            raise SystemExit(
                "--prompt-tokens repeats the raw prompt, so it cannot be used with --chat"
            )
        if count > 0:
            tokens = _repeat_to_length(tokenizer, text, count)
        else:
            tokens = encode_prompt(tokenizer, text, int(cfg.params["max_prompt_tokens"]))
        return tokenizer, text, tokens

    def validate(self, cfg: RunConfig) -> ValidationResult:
        """Compares logits, greedy continuation and tokenizer with the streamed float32 HF reference."""
        model_dir = resolve_model_dir(str(cfg.params["torch_model"]))
        num_layers = self._num_layers(cfg)
        tokenizer, text, tokens = self._prompt(cfg, model_dir)
        chat_text = _prompt_text(tokenizer, str(cfg.params["prompt"]), chat=True)
        tokenizer_mismatches = _tokenizer_mismatches(
            model_dir, tokenizer, [text, chat_text, *_TOKENIZER_SAMPLES]
        )

        rs_model = self._gpt_rs_model(cfg)
        with debug_context(cfg.params):
            rs_logits = np.asarray(rs_model.forward_tokens(tokens), dtype=np.float32)
        steps = int(cfg.params.get("generate_tokens", 0))
        continuation: list[int] = []
        if steps > 0:
            generated = rs_model.generate_tokens(tokens, steps, temperature=0.0, kv_cache=True)
            continuation = [int(t) for t in generated[len(tokens) :]]

        # One teacher-forced reference pass over prompt + continuation: causal logits at prompt
        # positions equal the prompt-only pass, and position len(tokens) - 1 + i predicts
        # continuation[i].
        ref_all = hf_reference_logits(model_dir, tokens + continuation[:-1], num_layers)
        ref_logits = ref_all[: len(tokens)]
        ok_steps = 0
        for i, token in enumerate(continuation):
            if int(ref_all[len(tokens) - 1 + i].argmax()) != token:
                break
            ok_steps += 1

        res = validation_result(
            model=self.name,
            torch_np=ref_logits,
            gptrs_np=rs_logits,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )
        ref_absmax = float(np.abs(ref_logits).max())
        rel_diff = res.max_abs_diff / ref_absmax if ref_absmax > 0 else float("inf")
        max_rel_diff = float(cfg.params.get("max_rel_diff", _DEFAULT_MAX_REL_DIFF))
        hf_top1 = int(ref_logits[-1].argmax())
        rs_top1 = int(rs_logits[-1].argmax())
        extra: dict[str, Any] = {
            "prompt_len": len(tokens),
            "num_layers": num_layers,
            "rel_max_abs_diff": rel_diff,
            "max_rel_diff": max_rel_diff,
            "allclose": res.ok,
            "hf_top1": hf_top1,
            "gpt_rs_top1": rs_top1,
            "top1_agreement_all_positions": float(
                np.mean(rs_logits.argmax(-1) == ref_logits.argmax(-1))
            ),
            "tokenizer_ok": not tokenizer_mismatches,
            "tokenizer_mismatches": tokenizer_mismatches,
        }
        if steps > 0:
            extra["validated_generate_tokens"] = ok_steps
            extra["requested_generate_tokens"] = steps
            extra["gpt_rs_continuation"] = tokenizer.decode(continuation)

        ok = (
            rel_diff <= max_rel_diff
            and hf_top1 == rs_top1
            and ok_steps == len(continuation)
            and not tokenizer_mismatches
        )
        return replace(res, ok=ok, extra=extra)

    def bench(self, cfg: RunConfig) -> BenchResult:
        """Per-token decode latency of gpt-rs and of bf16 HF transformers on CPU.

        Prefill times go in `extra`.
        """
        import torch
        from torch._dynamo.utils import counters
        from transformers import Qwen3_5ForCausalLM

        model_dir = resolve_model_dir(str(cfg.params["torch_model"]))
        num_layers = self._num_layers(cfg)
        _, _, tokens = self._prompt(cfg, model_dir)
        steps = int(cfg.params.get("bench_tokens", 32))
        if steps < 1:
            raise SystemExit("--bench-tokens must be at least 1")
        capacity = len(tokens) + steps + 1
        torch.set_num_threads(cfg.threads)

        def measure(generate: Callable[[int], Any]) -> tuple[list[float], list[float], int]:
            """Prefill and full-run times, plus Dynamo graphs compiled during the timed runs."""
            marks: list[int] = []

            def mark() -> None:
                marks.append(int(counters["stats"]["unique_graphs"]))

            hooks = {"before_iters": mark, "after_iters": mark}
            prefill = time_many(lambda: generate(1), warmup=cfg.warmup, iters=cfg.iters, **hooks)
            total = time_many(
                lambda: generate(steps + 1), warmup=cfg.warmup, iters=cfg.iters, **hooks
            )
            return prefill, total, marks[1] - marks[0] + marks[3] - marks[2]

        def measure_gpt_rs() -> tuple[list[float], list[float], int]:
            # Scoped so the gpt-rs model is released before the HF model loads.
            rs_model = self._gpt_rs_model(cfg)
            return measure(
                lambda n: rs_model.generate_tokens(
                    tokens,
                    n,
                    temperature=0.0,
                    kv_cache=True,
                    kv_cache_capacity=capacity,
                    ignore_eos=True,
                )
            )

        rs_prefill, rs_total, _ = measure_gpt_rs()
        gc.collect()

        hf = Qwen3_5ForCausalLM.from_pretrained(
            str(model_dir), config=_hf_text_config(model_dir, num_layers), dtype=torch.bfloat16
        ).eval()
        compiled = bool(cfg.params.get("torch_compile", True))
        if compiled:
            hf.forward = torch.compile(hf.forward, backend="inductor")
        input_ids = torch.tensor([tokens], dtype=torch.long)
        # HF's hybrid Qwen3_5DynamicCache cannot be static: generate ignores cache_implementation.
        with torch.inference_mode():
            hf_prefill, hf_total, graphs_while_timed = measure(
                lambda n: hf.generate(
                    input_ids, max_new_tokens=n, min_new_tokens=n, do_sample=False
                )
            )

        def decode_stats(
            prefill: list[float], total: list[float], impl: str, extra: dict[str, Any]
        ) -> BenchStats:
            prefill_mean = statistics.mean(prefill)
            return bench_stats(
                [(t - prefill_mean) / steps for t in total],
                units_per_iter=1.0,
                impl=impl,
                extra={"prefill_mean_s": prefill_mean, "prefill_times_s": prefill, **extra},
            )

        torch_impl = "transformers/bf16"
        torch_extra: dict[str, Any] = {}
        if compiled:
            torch_extra = {
                "dynamo_graphs_compiled_while_timed": graphs_while_timed,
                "dynamo_graph_breaks": int(sum(counters["graph_break"].values())),
                "dynamo_unsupported": int(sum(counters["unimplemented"].values())),
            }
            torch_impl += "+inductor" if not any(torch_extra.values()) else "+inductor(partial)"
        return BenchResult(
            model=self.name,
            threads=cfg.threads,
            unit_label="decode_tokens",
            units_per_iter=1.0,
            gptrs=decode_stats(rs_prefill, rs_total, f"gpt-rs/{cfg.backend}", {}),
            torch=decode_stats(hf_prefill, hf_total, torch_impl, torch_extra),
            extra={
                "prompt_len": len(tokens),
                "bench_tokens": steps,
                "num_layers": num_layers,
                "matmul_input_dtype": cfg.params.get("matmul_input_dtype") or "checkpoint",
            },
        )

    def run(self, cfg: RunConfig) -> CliRunResult:
        tokenizer, _, tokens = self._prompt(cfg, resolve_model_dir(str(cfg.params["torch_model"])))
        rs_model = self._gpt_rs_model(cfg)
        max_tokens = int(cfg.params.get("max_tokens", 64))
        temperature = float(cfg.params.get("temperature", 0.0))
        t0 = time.perf_counter()
        with debug_context(cfg.params):
            out = rs_model.generate_tokens(
                tokens, max_tokens, temperature=temperature, kv_cache=True
            )
        wall_s = time.perf_counter() - t0
        sys.stdout.write(tokenizer.decode(out[len(tokens) :], skip_special_tokens=True) + "\n")
        return CliRunResult(
            model=self.name, impl=f"gpt-rs/{cfg.backend}", exit_code=0, wall_s=wall_s
        )
