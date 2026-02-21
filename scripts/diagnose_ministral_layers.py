from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy as np
import torch


def _resolve_torch_dtype(name: str) -> Any:
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


def _encode_prompt(tokenizer: Any, prompt: str, max_prompt_tokens: int) -> List[int]:
    tokens = [int(tok) for tok in tokenizer.encode(prompt, add_special_tokens=False)]
    if max_prompt_tokens > 0 and len(tokens) > max_prompt_tokens:
        tokens = tokens[-max_prompt_tokens:]
    if not tokens:
        raise ValueError("prompt produced zero tokens")
    return tokens


def _find_text_model(torch_model: Any) -> Any:
    top_model = getattr(torch_model, "model", None)
    if top_model is not None:
        language_model = getattr(top_model, "language_model", None)
        if language_model is not None:
            return language_model
        return top_model
    language_model = getattr(torch_model, "language_model", None)
    if language_model is not None:
        return language_model
    return torch_model


def _find_final_norm_module(torch_model: Any) -> Any:
    text_model = _find_text_model(torch_model)
    for attr in ("norm", "final_layernorm"):
        mod = getattr(text_model, attr, None)
        if mod is not None:
            return mod
    raise RuntimeError(
        "could not locate final normalization module on Torch text model "
        f"(type={type(text_model)!r})"
    )


def _build_hf(args: argparse.Namespace) -> Tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_model_id = str(args.torch_model)
    trust_remote_code = bool(args.trust_remote_code)
    torch_dtype = _resolve_torch_dtype(str(args.torch_dtype))

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
            Any, AutoModelForCausalLM.from_pretrained(torch_model_id, **model_kwargs)
        )

    torch_model.eval()
    torch_model.to(args.torch_device)
    if bool(args.force_float32):
        torch_model = torch_model.to(dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(torch_model_id, trust_remote_code=trust_remote_code)
    return tokenizer, torch_model


def _build_gpt_rs(args: argparse.Namespace) -> Any:
    import gpt_rs

    gpt_rs.set_backend(str(args.backend))
    return gpt_rs.load_model(str(args.checkpoint))


def _to_np_f32(tensor: Any) -> np.ndarray:
    if torch.is_tensor(tensor):
        return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(tensor, dtype=np.float32)


def _collect_hf_activations(
    torch_model: Any, tokens: List[int], device: str
) -> Dict[str, np.ndarray]:
    text_model = _find_text_model(torch_model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise RuntimeError(f"text model {type(text_model)!r} does not expose .layers")
    num_layers = int(len(layers))
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    embeddings: Dict[str, np.ndarray] = {}
    norm1_outputs: Dict[int, np.ndarray] = {}
    q_outputs: Dict[int, np.ndarray] = {}
    k_outputs: Dict[int, np.ndarray] = {}
    v_outputs: Dict[int, np.ndarray] = {}
    attn_outputs: Dict[int, np.ndarray] = {}
    norm2_outputs: Dict[int, np.ndarray] = {}
    ff_outputs: Dict[int, np.ndarray] = {}
    layer_outputs: Dict[int, np.ndarray] = {}
    norm_outputs: Dict[str, np.ndarray] = {}
    handles: List[Any] = []

    embed_tokens = getattr(text_model, "embed_tokens", None)
    if embed_tokens is None:
        raise RuntimeError(f"text model {type(text_model)!r} does not expose .embed_tokens")
    norm_module = _find_final_norm_module(torch_model)

    def _embed_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
        embeddings["tok_embeddings"] = _to_np_f32(output[0])

    def _layer_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            layer_outputs[layer_idx] = _to_np_f32(value[0])

        return hook

    def _norm1_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            norm1_outputs[layer_idx] = _to_np_f32(output[0])

        return hook

    def _linear_hook(target: Dict[int, np.ndarray], layer_idx: int):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            target[layer_idx] = _to_np_f32(output[0])

        return hook

    def _attn_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            attn_outputs[layer_idx] = _to_np_f32(value[0])

        return hook

    def _norm2_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            norm2_outputs[layer_idx] = _to_np_f32(output[0])

        return hook

    def _ff_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            ff_outputs[layer_idx] = _to_np_f32(output[0])

        return hook

    def _norm_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
        norm_outputs["final_norm"] = _to_np_f32(output[0])

    handles.append(embed_tokens.register_forward_hook(_embed_hook))
    for layer_idx, layer in enumerate(layers):
        handles.append(layer.input_layernorm.register_forward_hook(_norm1_hook(layer_idx)))
        handles.append(
            layer.self_attn.q_proj.register_forward_hook(_linear_hook(q_outputs, layer_idx))
        )
        handles.append(
            layer.self_attn.k_proj.register_forward_hook(_linear_hook(k_outputs, layer_idx))
        )
        handles.append(
            layer.self_attn.v_proj.register_forward_hook(_linear_hook(v_outputs, layer_idx))
        )
        handles.append(layer.self_attn.register_forward_hook(_attn_hook(layer_idx)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(_norm2_hook(layer_idx)))
        handles.append(layer.mlp.register_forward_hook(_ff_hook(layer_idx)))
        handles.append(layer.register_forward_hook(_layer_hook(layer_idx)))
    handles.append(norm_module.register_forward_hook(_norm_hook))

    with torch.no_grad():
        out = torch_model(input_ids=input_ids, return_dict=True)
    for handle in handles:
        handle.remove()

    activations: Dict[str, np.ndarray] = {}
    if "tok_embeddings" not in embeddings:
        raise RuntimeError("failed to capture tok_embeddings from Torch hooks")
    activations["tok_embeddings"] = embeddings["tok_embeddings"]
    for layer_idx in range(num_layers):
        if layer_idx in norm1_outputs:
            activations[f"blocks.{layer_idx}.norm_1"] = norm1_outputs[layer_idx]
        if layer_idx in q_outputs and layer_idx in k_outputs and layer_idx in v_outputs:
            activations[f"blocks.{layer_idx}.attention.qkv"] = np.concatenate(
                [q_outputs[layer_idx], k_outputs[layer_idx], v_outputs[layer_idx]], axis=-1
            )
        if layer_idx in attn_outputs:
            activations[f"blocks.{layer_idx}.attention.output"] = attn_outputs[layer_idx]
        if layer_idx in norm2_outputs:
            activations[f"blocks.{layer_idx}.norm_2"] = norm2_outputs[layer_idx]
        if layer_idx in ff_outputs:
            activations[f"blocks.{layer_idx}.feed_forward.output"] = ff_outputs[layer_idx]
        if layer_idx not in layer_outputs:
            raise RuntimeError(f"failed to capture blocks.{layer_idx}.output from Torch hooks")
        activations[f"blocks.{layer_idx}.output"] = layer_outputs[layer_idx]
    if "final_norm" not in norm_outputs:
        raise RuntimeError("failed to capture final_norm from Torch hooks")
    activations["final_norm"] = norm_outputs["final_norm"]
    activations["logits"] = _to_np_f32(out.logits[0])
    return activations


def _collect_gpt_rs_activations(rs_model: Any, tokens: List[int]) -> Dict[str, np.ndarray]:
    raw = rs_model.debug_token_activations(tokens)
    activations: Dict[str, np.ndarray] = {}
    for item in raw:
        name = str(item["name"])
        tensor = np.asarray(item["tensor"], dtype=np.float32)
        activations[name] = tensor
    return activations


def _ordered_keys(keys: Iterable[str]) -> List[str]:
    def _key_rank(name: str) -> Tuple[int, int]:
        if name == "tok_embeddings":
            return (0, 0)
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                layer_idx = int(parts[1])
                suffix = ".".join(parts[2:])
                stage_rank = {
                    "norm_1": 0,
                    "attention.qkv": 1,
                    "attention.qkv_rope": 2,
                    "attention.context": 3,
                    "attention.output": 4,
                    "residual": 5,
                    "norm_2": 6,
                    "feed_forward.output": 7,
                    "output": 8,
                }.get(suffix, 9)
                return (1, layer_idx * 16 + stage_rank)
        if name == "final_norm":
            return (2, 1)
        if name == "logits":
            return (3, 1)
        return (4, 1)

    return sorted(keys, key=_key_rank)


def _print_report(
    hf_acts: Dict[str, np.ndarray],
    rs_acts: Dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> int:
    shared = [k for k in _ordered_keys(hf_acts.keys()) if k in rs_acts]
    missing_hf = sorted(set(rs_acts.keys()) - set(hf_acts.keys()))
    missing_rs = sorted(set(hf_acts.keys()) - set(rs_acts.keys()))

    if missing_hf:
        print(f"warning: names only in gpt-rs: {missing_hf}")
    if missing_rs:
        print(f"warning: names only in torch: {missing_rs}")

    first_fail = -1
    print(f"{'activation':34} {'shape':18} {'max_abs':>12} {'mean_abs':>12} {'allclose':>10}")
    print("-" * 92)
    for idx, name in enumerate(shared):
        hf = hf_acts[name]
        rs = rs_acts[name]
        if hf.shape != rs.shape:
            print(f"{name:34} shape_mismatch torch={hf.shape} gpt-rs={rs.shape}")
            if first_fail < 0:
                first_fail = idx
            continue
        diff = np.abs(hf - rs)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        ok = bool(np.allclose(hf, rs, rtol=rtol, atol=atol))
        print(f"{name:34} {str(hf.shape):18} {max_abs:12.6e} {mean_abs:12.6e} {str(ok):>10}")
        if not ok and first_fail < 0:
            first_fail = idx

    if first_fail < 0:
        print("first_fail=none")
        return 0

    first_name = shared[first_fail]
    print(f"first_fail_index={first_fail} first_fail_name={first_name}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Layer-by-layer Torch vs gpt-rs parity for Ministral."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ministral_3_3b_instruct_2512.bin"),
        help="gpt-rs checkpoint path",
    )
    parser.add_argument(
        "--torch-model",
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="Torch/HF model id",
    )
    parser.add_argument("--backend", default="faer", help="gpt-rs backend")
    parser.add_argument("--torch-device", default="cpu", help="Torch device")
    parser.add_argument(
        "--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--force-float32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cast Torch model to float32 after load.",
    )
    parser.add_argument("--prompt", default="Hello world", help="Prompt text to compare")
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow loading model/tokenizer with remote code",
    )
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    tokenizer, torch_model = _build_hf(args)
    rs_model = _build_gpt_rs(args)

    prompt_tokens = _encode_prompt(tokenizer, str(args.prompt), int(args.max_prompt_tokens))
    print(f"prompt_tokens={prompt_tokens}")

    hf_acts = _collect_hf_activations(torch_model, prompt_tokens, str(args.torch_device))
    rs_acts = _collect_gpt_rs_activations(rs_model, prompt_tokens)
    return _print_report(hf_acts, rs_acts, rtol=float(args.rtol), atol=float(args.atol))


if __name__ == "__main__":
    raise SystemExit(main())
