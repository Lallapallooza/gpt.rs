from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, cast

import numpy as np

from gptrs_eval.checkpoint import save as save_checkpoint

from ..types import ArtifactDefaults, ExporterInfo, ExportRequest, ExportResult

_DEFAULT_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
_DEFAULT_CHECKPOINT = Path("checkpoints/ministral_3_3b_instruct_2512.bin")
_DEFAULT_CONFIG = Path("configs/ministral_3_3b_instruct_2512_model.json")
_DEFAULT_TOKENIZER = Path("configs/ministral_3_3b_instruct_2512_tokenizer.json")


def _to_numpy_f32(tensor: Any) -> np.ndarray:
    import torch

    return tensor.detach().to(torch.float32).cpu().numpy()


def _state_tensor(state: Mapping[str, Any], key: str) -> Any:
    if key not in state:
        raise KeyError(f"missing tensor in state dict: {key}")
    return state[key]


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
        raise ValueError(f"unsupported --torch-dtype {name!r}; expected one of: {known}")
    return mapping[name]


def _rope_scaling_from_hf(cfg: Any) -> Dict[str, Any]:
    raw = getattr(cfg, "rope_scaling", None)
    if raw is None:
        return {"kind": "none"}

    rope_scaling: Mapping[str, Any]
    if isinstance(raw, dict):
        rope_scaling = raw
    elif hasattr(raw, "to_dict"):
        rope_scaling = raw.to_dict()
    else:
        raise ValueError(f"unsupported rope_scaling payload type: {type(raw)!r}")

    rope_type = str(rope_scaling.get("rope_type", rope_scaling.get("type", "none"))).lower()
    if rope_type in {"none", ""}:
        return {"kind": "none"}

    factor_value = rope_scaling.get("factor")
    if factor_value is None:
        factor_value = 1.0
    factor = float(factor_value)
    if rope_type == "linear":
        return {"kind": "linear", "factor": factor}
    if rope_type == "yarn":
        mscale_value = rope_scaling.get("mscale")
        if mscale_value is None:
            mscale_value = rope_scaling.get("attention_factor")
        if mscale_value is None:
            mscale_value = 1.0
        return {"kind": "yarn", "factor": factor, "mscale": float(mscale_value)}
    raise ValueError(f"unsupported rope scaling type for Ministral exporter: {rope_type!r}")


def _rotary_dim_from_hf(cfg: Any) -> int:
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    if hidden_size % num_heads != 0:
        raise ValueError(
            "invalid attention dimensions: "
            f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}"
        )

    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    partial_rotary = float(getattr(cfg, "partial_rotary_factor", 1.0))

    rotary_dim = int(round(head_dim * partial_rotary))
    rotary_dim = min(head_dim, max(2, rotary_dim))
    if rotary_dim % 2 != 0:
        rotary_dim -= 1
    if rotary_dim <= 0:
        raise ValueError(f"derived rotary_dim is invalid: {rotary_dim}")
    return rotary_dim


def _context_length_from_hf(cfg: Any) -> int:
    if hasattr(cfg, "max_position_embeddings"):
        return int(cfg.max_position_embeddings)
    if hasattr(cfg, "max_seq_len"):
        return int(cfg.max_seq_len)
    return 2048


def _collect_checkpoint(model: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    import torch

    cfg = model.config
    state = model.state_dict()

    embed_dim = int(cfg.hidden_size)
    num_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    mlp_hidden_dim = int(cfg.intermediate_size)
    vocab_size = int(cfg.vocab_size)
    context_length = _context_length_from_hf(cfg)

    if embed_dim % num_heads != 0:
        raise ValueError(
            "invalid attention dimensions: "
            f"hidden_size={embed_dim} is not divisible by num_heads={num_heads}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "invalid grouped-query attention dimensions: "
            f"num_heads={num_heads} is not divisible by num_kv_heads={num_kv_heads}"
        )

    head_dim = embed_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    qkv_dim = embed_dim + kv_dim + kv_dim

    tensors: Dict[str, np.ndarray] = {}
    add_tensor = tensors.__setitem__

    add_tensor(
        "tok_embeddings.weight",
        _to_numpy_f32(_state_tensor(state, "model.embed_tokens.weight")),
    )

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        q_proj = _state_tensor(state, f"{prefix}.self_attn.q_proj.weight").t()
        k_proj = _state_tensor(state, f"{prefix}.self_attn.k_proj.weight").t()
        v_proj = _state_tensor(state, f"{prefix}.self_attn.v_proj.weight").t()
        packed_qkv = torch.cat((q_proj, k_proj, v_proj), dim=1)
        if tuple(packed_qkv.shape) != (embed_dim, qkv_dim):
            raise ValueError(
                f"layer {i} packed qkv shape mismatch: expected {(embed_dim, qkv_dim)}, "
                f"got {tuple(packed_qkv.shape)}"
            )
        add_tensor(f"blocks.{i}.attention.w_qkv", _to_numpy_f32(packed_qkv))

        add_tensor(
            f"blocks.{i}.attention.w_out",
            _to_numpy_f32(_state_tensor(state, f"{prefix}.self_attn.o_proj.weight").t()),
        )
        add_tensor(
            f"blocks.{i}.feed_forward.w_gate",
            _to_numpy_f32(_state_tensor(state, f"{prefix}.mlp.gate_proj.weight").t()),
        )
        add_tensor(
            f"blocks.{i}.feed_forward.w_up",
            _to_numpy_f32(_state_tensor(state, f"{prefix}.mlp.up_proj.weight").t()),
        )
        add_tensor(
            f"blocks.{i}.feed_forward.w_down",
            _to_numpy_f32(_state_tensor(state, f"{prefix}.mlp.down_proj.weight").t()),
        )
        add_tensor(
            f"blocks.{i}.norm_1.gamma",
            _to_numpy_f32(_state_tensor(state, f"{prefix}.input_layernorm.weight")),
        )
        add_tensor(
            f"blocks.{i}.norm_2.gamma",
            _to_numpy_f32(_state_tensor(state, f"{prefix}.post_attention_layernorm.weight")),
        )

    add_tensor("final_norm.gamma", _to_numpy_f32(_state_tensor(state, "model.norm.weight")))
    lm_head_weight = state.get("lm_head.weight")
    if lm_head_weight is None:
        lm_head_weight = _state_tensor(state, "model.embed_tokens.weight")
    add_tensor("lm_head", _to_numpy_f32(lm_head_weight.t()))

    model_config = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_hidden_dim": mlp_hidden_dim,
        "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-5)),
        "rope_theta": float(getattr(cfg, "rope_theta", 10_000.0)),
        "rotary_dim": _rotary_dim_from_hf(cfg),
        "rope_scaling": _rope_scaling_from_hf(cfg),
    }
    return model_config, tensors


def _load_model_and_tokenizer(
    model_id: str,
    device: str,
    torch_dtype_name: str,
    trust_remote_code: bool,
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    dtype = _resolve_torch_dtype(torch_dtype_name)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype

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
) -> None:
    save_checkpoint(path, kind="ministral", config=config, tensors=tensors)


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
        _write_checkpoint(request.checkpoint_out, model_config, tensors)

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
