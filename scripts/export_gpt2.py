#!/usr/bin/env python3
"""Export GPT-2 weights into the gpt.rs checkpoint format.

Usage:
    python scripts/export_gpt2.py \
        --model gpt2 \
        --checkpoint-out checkpoints/gpt2.bin \
        --config-out configs/gpt2_model.json \
        --tokenizer-out configs/gpt2_tokenizer.json

Requires 	orch and 	ransformers.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from gptrs_eval.checkpoint import save as save_checkpoint


@dataclass
class ExportTensor:
    name: str
    array: np.ndarray
    requires_grad: bool = False

    def shape(self) -> Iterable[int]:
        return list(self.array.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GPT-2 to gpt.rs checkpoint")
    parser.add_argument("--model", default="gpt2", help="Hugging Face model id (default: gpt2)")
    parser.add_argument(
        "--checkpoint-out",
        required=True,
        type=Path,
        help="Path to write the gpt.rs checkpoint binary",
    )
    parser.add_argument(
        "--config-out",
        required=True,
        type=Path,
        help="Path to write the gpt.rs model config JSON",
    )
    parser.add_argument(
        "--tokenizer-out",
        required=True,
        type=Path,
        help="Path to write the gpt.rs tokenizer JSON",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to load the model on (default: cpu)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_id: str, device: str
) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return model, tokenizer


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def collect_tensors(model: GPT2LMHeadModel) -> Tuple[Dict[str, int], List[ExportTensor]]:
    cfg = model.config
    state: Dict[str, torch.Tensor] = model.state_dict()

    tensors: List[ExportTensor] = []
    add = tensors.append

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        add(ExportTensor(name=name, array=to_numpy(tensor)))

    add_tensor("tok_embeddings.weight", state["transformer.wte.weight"])
    add_tensor("pos_embeddings", state["transformer.wpe.weight"])

    for i in range(cfg.n_layer):
        prefix = f"transformer.h.{i}"
        attn_prefix = f"blocks.{i}.attention"
        mlp_prefix = f"blocks.{i}.feed_forward"
        ln1_prefix = f"blocks.{i}.ln_1"
        ln2_prefix = f"blocks.{i}.ln_2"

        attn_w = state[f"{prefix}.attn.c_attn.weight"]
        attn_b = state[f"{prefix}.attn.c_attn.bias"]
        proj_w = state[f"{prefix}.attn.c_proj.weight"]
        proj_b = state[f"{prefix}.attn.c_proj.bias"]
        add_tensor(f"{attn_prefix}.w_qkv", attn_w)
        add_tensor(f"{attn_prefix}.b_qkv", attn_b)
        add_tensor(f"{attn_prefix}.w_out", proj_w)
        add_tensor(f"{attn_prefix}.b_out", proj_b)

        fc_w = state[f"{prefix}.mlp.c_fc.weight"]
        fc_b = state[f"{prefix}.mlp.c_fc.bias"]
        proj_ff_w = state[f"{prefix}.mlp.c_proj.weight"]
        proj_ff_b = state[f"{prefix}.mlp.c_proj.bias"]
        add_tensor(f"{mlp_prefix}.w_in", fc_w)
        add_tensor(f"{mlp_prefix}.b_in", fc_b)
        add_tensor(f"{mlp_prefix}.w_out", proj_ff_w)
        add_tensor(f"{mlp_prefix}.b_out", proj_ff_b)

        add_tensor(f"{ln1_prefix}.gamma", state[f"{prefix}.ln_1.weight"])
        add_tensor(f"{ln1_prefix}.beta", state[f"{prefix}.ln_1.bias"])
        add_tensor(f"{ln2_prefix}.gamma", state[f"{prefix}.ln_2.weight"])
        add_tensor(f"{ln2_prefix}.beta", state[f"{prefix}.ln_2.bias"])

    add_tensor("final_ln.gamma", state["transformer.ln_f.weight"])
    add_tensor("final_ln.beta", state["transformer.ln_f.bias"])
    add_tensor("lm_head", state["lm_head.weight"].t())

    model_config = {
        "vocab_size": cfg.vocab_size,
        "context_length": cfg.n_positions,
        "embed_dim": cfg.n_embd,
        "num_layers": cfg.n_layer,
        "num_heads": cfg.n_head,
        "mlp_ratio": cfg.n_inner // cfg.n_embd if cfg.n_inner else 4,
        "dropout": float(cfg.resid_pdrop),
    }

    return model_config, tensors


def export_tokenizer(tokenizer: GPT2TokenizerFast, path: Path) -> None:
    vocab = tokenizer.get_vocab()
    model = tokenizer.backend_tokenizer.model
    merges: List[Tuple[str, str]] = []
    if hasattr(model, "get_merges"):
        merges = cast(List[Tuple[str, str]], model.get_merges())  # type: ignore[attr-defined]
    else:
        state: object = getattr(model, "__getstate__", lambda: {})()
        raw_merges: object | None = None
        if isinstance(state, dict):
            raw_merges = state.get("merges")
        elif isinstance(state, (bytes, bytearray)):
            try:
                decoded = state.decode("utf-8")
                parsed_state = json.loads(decoded)
                if isinstance(parsed_state, dict):
                    raw_merges = parsed_state.get("merges")
                else:
                    raw_merges = parsed_state
            except (UnicodeDecodeError, json.JSONDecodeError):
                raw_merges = state
        else:
            raw_merges = state
        if isinstance(raw_merges, (bytes, bytearray)):
            lines = raw_merges.decode("utf-8").splitlines()
            parsed: List[Tuple[str, str]] = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    parsed.append((parts[0], parts[1]))
            merges = parsed
        elif isinstance(raw_merges, list):
            merges = [
                tuple(pair)
                for pair in raw_merges
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ]
        else:
            merges = []
    merges_serialised: List[Tuple[str, str]] = [(str(a), str(b)) for a, b in merges]
    data = {
        "vocab": vocab,
        "merges": merges_serialised,
        "unk_token": tokenizer.unk_token or "<unk>",
    }
    path.write_text(json.dumps(data, indent=2))


def write_checkpoint(path: Path, config: Dict[str, Any], tensors: Iterable[ExportTensor]) -> None:
    tensor_map: Dict[str, np.ndarray] = {t.name: t.array for t in tensors}
    requires_grad = {t.name: bool(t.requires_grad) for t in tensors}
    save_checkpoint(
        path,
        kind="gpt",
        config=config,
        tensors=tensor_map,
        requires_grad=requires_grad,
    )


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    model_config, tensors = collect_tensors(model)

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    args.tokenizer_out.parent.mkdir(parents=True, exist_ok=True)

    args.config_out.write_text(json.dumps(model_config, indent=2))
    export_tokenizer(tokenizer, args.tokenizer_out)
    write_checkpoint(args.checkpoint_out, model_config, tensors)

    print(f"Exported checkpoint to {args.checkpoint_out}")
    print(f"Model config written to {args.config_out}")
    print(f"Tokenizer config written to {args.tokenizer_out}")


if __name__ == "__main__":
    main()
