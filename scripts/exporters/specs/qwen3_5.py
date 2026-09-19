from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gptrs_eval.checkpoint import (
    TensorPlan,
    hf_eos_token_ids,
    hf_text_tensor_names,
    write_streaming,
)
from gptrs_eval.safetensors_source import SafetensorsSource, resolve_model_dir

from ..pipeline import missing_outputs
from ..types import (
    ArtifactDefaults,
    EvalCaseRegistration,
    ExporterInfo,
    ExportRequest,
    ExportResult,
)

_DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"
_DEFAULT_CHECKPOINT = Path("checkpoints/qwen3_8_27b.bin")
_DEFAULT_CONFIG = Path("configs/qwen3_8_27b_model.json")
_DEFAULT_TOKENIZER = Path("configs/qwen3_8_27b_tokenizer.json")

_KIND = "qwen3_5"


def build_model_config(
    raw_config: Mapping[str, Any],
    num_layers_override: Optional[int] = None,
) -> Dict[str, Any]:
    """The Hugging Face text config as `Qwen3_5TextConfig` resolves it.

    Multimodal checkpoints hold it under `text_config`. `num_layers_override` keeps only the first
    N decoder layers, with real weights and the same embedding, final norm and lm_head.
    """
    from transformers import Qwen3_5TextConfig

    cfg: Dict[str, Any] = Qwen3_5TextConfig(**raw_config.get("text_config", raw_config)).to_dict()
    if num_layers_override is not None:
        num_layers = int(cfg["num_hidden_layers"])
        if not 0 < num_layers_override <= num_layers:
            raise ValueError(
                f"--num-layers must be in [1, {num_layers}], got {num_layers_override}"
            )
        cfg["num_hidden_layers"] = num_layers_override
        cfg["layer_types"] = cfg["layer_types"][:num_layers_override]
    return cfg


def _array_of(tensor: Any, dtype: str) -> np.ndarray:
    """Little-endian array of `tensor` in the export dtype, with bf16 as raw 16-bit patterns."""
    import torch

    if dtype == "bf16":
        return tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy()
    if dtype == "f32":
        return tensor.to(torch.float32).contiguous().numpy()
    raise ValueError(f"unsupported export dtype {dtype!r}")


def plan_tensors(
    source: SafetensorsSource,
    model_cfg: Mapping[str, Any],
    weight_dtype: str,
    tie_word_embeddings: bool,
) -> List[tuple[TensorPlan, str]]:
    """The checkpoint tensors, each with the name of the Hugging Face tensor it copies.

    Tensors keep their text-model names. Matrices use `weight_dtype` and every other tensor uses
    f32. The plan skips layers past `model_cfg["layer_types"]`, as in a truncated export.
    """

    names = hf_text_tensor_names(source.weight_map)
    if "lm_head.weight" not in names:
        if not tie_word_embeddings:
            raise KeyError("checkpoint has no lm_head.weight and tie_word_embeddings is false")
        names["lm_head.weight"] = names["model.embed_tokens.weight"]
    num_layers = len(model_cfg["layer_types"])

    def kept(name: str) -> bool:
        parts = name.split(".")
        return parts[:2] != ["model", "layers"] or int(parts[2]) < num_layers

    plans: List[tuple[TensorPlan, str]] = []
    for name, src in names.items():
        if kept(name):
            shape = source.shape(src)
            dtype = weight_dtype if len(shape) == 2 else "f32"
            plans.append((TensorPlan(name=name, shape=shape, dtype=dtype), src))
    return plans


def export_checkpoint(
    model_dir: Path,
    checkpoint_out: Path,
    weight_dtype: str = "bf16",
    num_layers: Optional[int] = None,
) -> tuple[Dict[str, Any], int]:
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    generation_path = model_dir / "generation_config.json"
    generation_config = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.exists()
        else None
    )
    model_cfg = build_model_config(raw_config, num_layers)
    source = SafetensorsSource(model_dir)
    tied = bool(raw_config.get("tie_word_embeddings", False))
    planned = plan_tensors(source, model_cfg, weight_dtype, tied)
    sources = {plan.name: src for plan, src in planned}
    write_streaming(
        checkpoint_out,
        kind=_KIND,
        config=model_cfg,
        plans=[plan for plan, _ in planned],
        produce=lambda plan: _array_of(source.get(sources[plan.name]), plan.dtype),
        eos_token_ids=hf_eos_token_ids(
            generation_config, raw_config, raw_config.get("text_config")
        ),
    )
    return model_cfg, len(planned)


class Qwen35Exporter:
    info = ExporterInfo(
        name="qwen3_8_27b",
        kind=_KIND,
        description="Export Hugging Face Qwen3.8-27B (Qwen3.5 architecture, text model) into gpt.rs format.",
        artifacts=ArtifactDefaults(
            checkpoint=_DEFAULT_CHECKPOINT,
            config=_DEFAULT_CONFIG,
            tokenizer=_DEFAULT_TOKENIZER,
        ),
        eval_case=EvalCaseRegistration(
            model_name="qwen3_8_27b",
            module="gptrs_eval.models.qwen3_5",
            cls="Qwen35Case",
            default_params={
                "checkpoint": _DEFAULT_CHECKPOINT,
                "torch_model": _DEFAULT_MODEL_ID,
                "prompt": "The capital of France is",
                "generate_tokens": 0,
                "max_prompt_tokens": 64,
            },
        ),
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-id",
            default=_DEFAULT_MODEL_ID,
            help=f"Hugging Face model id or local directory (default: {_DEFAULT_MODEL_ID}).",
        )
        parser.add_argument(
            "--weight-dtype",
            default="bf16",
            choices=["bf16", "f32"],
            help="Storage dtype for projection/embedding matrices (default: bf16, the native dtype).",
        )
        parser.add_argument(
            "--num-layers",
            type=int,
            default=None,
            help="Export only the first N decoder layers (for validation/profiling).",
        )

    def export(self, request: ExportRequest) -> ExportResult:
        model_id = str(request.options.get("model_id", _DEFAULT_MODEL_ID))
        weight_dtype = str(request.options.get("weight_dtype", "bf16"))
        num_layers = request.options.get("num_layers")
        model_dir = resolve_model_dir(model_id)
        tokenizer_src = model_dir / "tokenizer.json"
        tokenizer_out: Optional[Path] = request.tokenizer_out
        if tokenizer_out is not None and not tokenizer_src.exists():
            raise FileNotFoundError(f"no tokenizer.json in {model_dir}")

        model_cfg, tensor_count = export_checkpoint(
            model_dir,
            request.checkpoint_out,
            weight_dtype,
            int(num_layers) if num_layers is not None else None,
        )

        config_out: Optional[Path] = request.config_out
        if config_out is not None:
            config_out.write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
        if tokenizer_out is not None:
            shutil.copyfile(tokenizer_src, tokenizer_out)

        return ExportResult(
            exporter=self.info.name,
            kind=self.info.kind,
            checkpoint=request.checkpoint_out,
            tensor_count=tensor_count,
            config=config_out,
            tokenizer=tokenizer_out,
            extras={
                "model_id": model_id,
                "model_dir": str(model_dir),
                "weight_dtype": weight_dtype,
            },
        )

    def validate(self, request: ExportRequest) -> list[str]:
        return missing_outputs(request)
