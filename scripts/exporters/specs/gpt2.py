from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy as np

from gptrs_eval.checkpoint import save as save_checkpoint

from ..types import (
    ArtifactDefaults,
    EvalCaseRegistration,
    ExporterInfo,
    ExportRequest,
    ExportResult,
)


@dataclass
class ExportTensor:
    name: str
    array: np.ndarray
    requires_grad: bool = False

    def shape(self) -> Iterable[int]:
        return list(self.array.shape)


def _to_numpy(tensor: Any) -> np.ndarray:
    import torch

    return tensor.detach().to(torch.float32).cpu().numpy()


def _load_model_and_tokenizer(model_id: str, device: str) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id)
    model: Any = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return model, tokenizer


def _collect_tensors(model: Any) -> Tuple[Dict[str, Any], List[ExportTensor]]:
    cfg = model.config
    state: Dict[str, Any] = model.state_dict()

    tensors: List[ExportTensor] = []
    add = tensors.append

    def add_tensor(name: str, tensor: Any) -> None:
        add(ExportTensor(name=name, array=_to_numpy(tensor)))

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


def _export_tokenizer(tokenizer: Any, path: Path) -> None:
    vocab = tokenizer.get_vocab()
    model = tokenizer.backend_tokenizer.model
    merges: List[Tuple[str, str]] = []
    if hasattr(model, "get_merges"):
        merges = cast(List[Tuple[str, str]], model.get_merges())  # type: ignore[attr-defined]
    else:
        state: Any = getattr(model, "__getstate__", lambda: {})()
        raw_merges: Any = None
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

    merges_serialized: List[Tuple[str, str]] = [(str(a), str(b)) for a, b in merges]
    data = {
        "vocab": vocab,
        "merges": merges_serialized,
        "unk_token": tokenizer.unk_token or "<unk>",
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_checkpoint(path: Path, config: Dict[str, Any], tensors: Iterable[ExportTensor]) -> None:
    tensor_map: Dict[str, np.ndarray] = {t.name: t.array for t in tensors}
    requires_grad = {t.name: bool(t.requires_grad) for t in tensors}
    save_checkpoint(
        path,
        kind="gpt",
        config=config,
        tensors=tensor_map,
        requires_grad=requires_grad,
    )


class GPT2Exporter:
    info = ExporterInfo(
        name="gpt2",
        kind="gpt",
        description="Export Hugging Face GPT-2 checkpoint, model config, and tokenizer.",
        artifacts=ArtifactDefaults(
            checkpoint=Path("checkpoints/gpt2.bin"),
            config=Path("configs/gpt2_model.json"),
            tokenizer=Path("configs/gpt2_tokenizer.json"),
        ),
        eval_case=EvalCaseRegistration(
            model_name="gpt2",
            module="gptrs_eval.models.gpt2",
            cls="Gpt2Case",
            default_params={
                "checkpoint": Path("checkpoints/gpt2.bin"),
                "tokenizer": Path("configs/gpt2_tokenizer.json"),
                "torch_model": "gpt2",
                "prompt": "Hello",
            },
        ),
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-id",
            default="gpt2",
            help="Hugging Face model id (default: gpt2).",
        )
        parser.add_argument(
            "--device",
            default="cpu",
            help="Torch device to load model on (default: cpu).",
        )

    def export(self, request: ExportRequest) -> ExportResult:
        config_out = request.config_out
        tokenizer_out = request.tokenizer_out
        if config_out is None or tokenizer_out is None:
            raise ValueError("gpt2 exporter requires config and tokenizer output paths")

        model_id = str(request.options.get("model_id", "gpt2"))
        device = str(request.options.get("device", "cpu"))

        model, tokenizer = _load_model_and_tokenizer(model_id, device)
        model_config, tensors = _collect_tensors(model)

        config_out.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
        _export_tokenizer(tokenizer, tokenizer_out)
        _write_checkpoint(request.checkpoint_out, model_config, tensors)

        return ExportResult(
            exporter=self.info.name,
            kind=self.info.kind,
            checkpoint=request.checkpoint_out,
            tensor_count=len(tensors),
            config=config_out,
            tokenizer=tokenizer_out,
            extras={"model_id": model_id, "device": device},
        )

    def validate(self, request: ExportRequest) -> list[str]:
        errors: list[str] = []
        if not request.checkpoint_out.exists():
            errors.append(f"missing checkpoint: {request.checkpoint_out}")
        if request.config_out is None:
            errors.append("missing config output path")
        elif not request.config_out.exists():
            errors.append(f"missing config: {request.config_out}")
        if request.tokenizer_out is None:
            errors.append("missing tokenizer output path")
        elif not request.tokenizer_out.exists():
            errors.append(f"missing tokenizer: {request.tokenizer_out}")
        return errors
