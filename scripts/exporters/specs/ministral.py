from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

import numpy as np

from gptrs_eval.checkpoint import hf_eos_token_ids, hf_text_tensor_names
from gptrs_eval.checkpoint import save as save_checkpoint
from gptrs_eval.core import resolve_torch_dtype

from ..pipeline import missing_outputs
from ..types import (
    ArtifactDefaults,
    EvalCaseRegistration,
    ExporterInfo,
    ExportRequest,
    ExportResult,
)

_DEFAULT_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
_DEFAULT_CHECKPOINT = Path("checkpoints/ministral_3_3b_instruct_2512.bin")
_DEFAULT_CONFIG = Path("configs/ministral_3_3b_instruct_2512_model.json")
_DEFAULT_TOKENIZER = Path("configs/ministral_3_3b_instruct_2512_tokenizer.json")


def _to_numpy_f32(tensor: Any) -> np.ndarray:
    import torch

    return tensor.detach().to(torch.float32).cpu().numpy()


def _collect_checkpoint(model: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """The Hugging Face text config and the text-decoder tensors as f32, under text-model names."""

    text_cfg = getattr(model.config, "text_config", None) or model.config
    state = model.state_dict()
    names = hf_text_tensor_names(state.keys())
    if "lm_head.weight" not in names:
        # Tied output projection.
        names["lm_head.weight"] = names["model.embed_tokens.weight"]
    tensors = {name: _to_numpy_f32(state[src]) for name, src in names.items()}
    return text_cfg.to_dict(), tensors


def _load_model_and_tokenizer(
    model_id: str,
    device: str,
    torch_dtype_name: str,
    trust_remote_code: bool,
) -> Tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    dtype = resolve_torch_dtype(torch_dtype_name)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model = cast(Any, Mistral3ForConditionalGeneration.from_pretrained(model_id, **kwargs))
    else:
        model = cast(Any, AutoModelForCausalLM.from_pretrained(model_id, **kwargs))
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return model, tokenizer


def _export_hf_tokenizer_json(tokenizer: Any, path: Path) -> None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise ValueError(
            "tokenizer backend does not expose JSON serialization; "
            "this exporter requires a fast tokenizer backend"
        )
    path.write_text(str(backend.to_str()), encoding="utf-8")


def _write_checkpoint(
    path: Path,
    config: Dict[str, Any],
    tensors: Mapping[str, np.ndarray],
    eos_token_ids: List[int],
) -> None:
    save_checkpoint(
        path, kind="ministral", config=config, tensors=tensors, eos_token_ids=eos_token_ids
    )


class MinistralExporter:
    info = ExporterInfo(
        name="ministral_3_3b_instruct_2512",
        kind="ministral",
        description=(
            "Export Hugging Face Ministral-3-3B-Instruct-2512 checkpoint/config into gpt.rs format."
        ),
        artifacts=ArtifactDefaults(
            checkpoint=_DEFAULT_CHECKPOINT,
            config=_DEFAULT_CONFIG,
            tokenizer=_DEFAULT_TOKENIZER,
        ),
        eval_case=EvalCaseRegistration(
            model_name="ministral_3_3b_instruct_2512",
            module="gptrs_eval.models.ministral",
            cls="MinistralCase",
            default_params={
                "checkpoint": _DEFAULT_CHECKPOINT,
                "torch_model": _DEFAULT_MODEL_ID,
                "prompt": "Hello",
                "generate_tokens": 0,
                "max_prompt_tokens": 64,
                "trust_remote_code": False,
                "torch_dtype": "float32",
            },
        ),
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-id",
            default=_DEFAULT_MODEL_ID,
            help=f"Hugging Face model id (default: {_DEFAULT_MODEL_ID}).",
        )
        parser.add_argument(
            "--device",
            default="cpu",
            help="Torch device to load model on (default: cpu).",
        )
        parser.add_argument(
            "--torch-dtype",
            default="auto",
            choices=["auto", "float16", "bfloat16", "float32"],
            help="Torch dtype for loading model weights (default: auto).",
        )
        parser.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Allow loading Hugging Face model/tokenizer with remote code (default: false).",
        )

    def export(self, request: ExportRequest) -> ExportResult:
        config_out = request.config_out
        tokenizer_out = request.tokenizer_out
        if config_out is None:
            raise ValueError("ministral exporter requires config output path")
        if tokenizer_out is None:
            raise ValueError("ministral exporter requires tokenizer output path")

        model_id = str(request.options.get("model_id", _DEFAULT_MODEL_ID))
        device = str(request.options.get("device", "cpu"))
        torch_dtype_name = str(request.options.get("torch_dtype", "auto"))
        trust_remote_code = bool(request.options.get("trust_remote_code", False))

        model, tokenizer = _load_model_and_tokenizer(
            model_id=model_id,
            device=device,
            torch_dtype_name=torch_dtype_name,
            trust_remote_code=trust_remote_code,
        )
        model_config, tensors = _collect_checkpoint(model)

        config_out.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
        _export_hf_tokenizer_json(tokenizer, tokenizer_out)
        hf_config = model.config.to_dict()
        eos_token_ids = hf_eos_token_ids(
            model.generation_config.to_dict(), hf_config, hf_config.get("text_config")
        )
        _write_checkpoint(request.checkpoint_out, model_config, tensors, eos_token_ids)

        return ExportResult(
            exporter=self.info.name,
            kind=self.info.kind,
            checkpoint=request.checkpoint_out,
            tensor_count=len(tensors),
            config=config_out,
            tokenizer=tokenizer_out,
            extras={
                "model_id": model_id,
                "device": device,
                "torch_dtype": torch_dtype_name,
                "trust_remote_code": trust_remote_code,
            },
        )

    def validate(self, request: ExportRequest) -> list[str]:
        return missing_outputs(request)
