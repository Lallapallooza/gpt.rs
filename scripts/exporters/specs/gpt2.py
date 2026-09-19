from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np

from gptrs_eval.checkpoint import hf_eos_token_ids
from gptrs_eval.checkpoint import save as save_checkpoint

from ..pipeline import missing_outputs
from ..types import (
    ArtifactDefaults,
    EvalCaseRegistration,
    ExporterInfo,
    ExportRequest,
    ExportResult,
)


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


def _collect_tensors(model: Any) -> Dict[str, np.ndarray]:
    """GPT-2 weights under the Llama-style decoder names, with projections in `[out, in]` layout.

    GPT-2 projections are `Conv1D` modules that store `[in, out]`, and `c_attn` fuses q, k and v.
    """

    cfg = model.config
    state: Dict[str, Any] = model.state_dict()
    tensors: Dict[str, np.ndarray] = {}

    def add(name: str, tensor: Any) -> None:
        tensors[name] = _to_numpy(tensor)

    def add_affine(dst: str, src: str, conv1d: bool = False) -> None:
        weight = state[f"{src}.weight"]
        add(f"{dst}.weight", weight.t() if conv1d else weight)
        add(f"{dst}.bias", state[f"{src}.bias"])

    add("model.embed_tokens.weight", state["transformer.wte.weight"])
    add("model.embed_positions.weight", state["transformer.wpe.weight"])
    for i in range(cfg.n_layer):
        src = f"transformer.h.{i}"
        dst = f"model.layers.{i}"
        add_affine(f"{dst}.input_layernorm", f"{src}.ln_1")
        qkv_weight = state[f"{src}.attn.c_attn.weight"].t().chunk(3, dim=0)
        qkv_bias = state[f"{src}.attn.c_attn.bias"].chunk(3)
        for proj, weight, bias in zip(("q_proj", "k_proj", "v_proj"), qkv_weight, qkv_bias):
            add(f"{dst}.self_attn.{proj}.weight", weight)
            add(f"{dst}.self_attn.{proj}.bias", bias)
        add_affine(f"{dst}.self_attn.o_proj", f"{src}.attn.c_proj", conv1d=True)
        add_affine(f"{dst}.post_attention_layernorm", f"{src}.ln_2")
        add_affine(f"{dst}.mlp.up_proj", f"{src}.mlp.c_fc", conv1d=True)
        add_affine(f"{dst}.mlp.down_proj", f"{src}.mlp.c_proj", conv1d=True)
    add_affine("model.norm", "transformer.ln_f")
    add("lm_head.weight", state["lm_head.weight"])
    return tensors


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
        tensors = _collect_tensors(model)
        model_config = model.config.to_dict()

        config_out.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
        _export_tokenizer(tokenizer, tokenizer_out)
        eos_token_ids = hf_eos_token_ids(model.generation_config.to_dict(), model_config)
        save_checkpoint(
            request.checkpoint_out,
            kind="gpt",
            config=model_config,
            tensors=tensors,
            eos_token_ids=eos_token_ids,
        )

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
        return missing_outputs(request)
