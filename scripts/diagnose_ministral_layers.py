from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy as np
import torch

from gptrs_eval.core import encode_prompt, resolve_torch_dtype


def _build_hf(args: argparse.Namespace) -> Tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_model_id = str(args.torch_model)
    trust_remote_code = bool(args.trust_remote_code)
    torch_dtype = resolve_torch_dtype(str(args.torch_dtype))

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
    torch_model: Any, tokens: List[int], device: str, names: Iterable[str]
) -> Dict[str, np.ndarray]:
    """Outputs of the Hugging Face modules whose paths are the gpt-rs activation names."""

    wanted = set(names)
    activations: Dict[str, np.ndarray] = {}
    handles: List[Any] = []
    for module_name, module in torch_model.named_modules():
        # Multimodal checkpoints nest the text model under `model.language_model.`.
        name = module_name.replace("model.language_model.", "model.", 1)
        if name not in wanted:
            continue

        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any, name: str = name) -> None:
            value = output[0] if isinstance(output, tuple) else output
            activations[name] = _to_np_f32(value[0])

        handles.append(module.register_forward_hook(hook))

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        torch_model(input_ids=input_ids)
    for handle in handles:
        handle.remove()
    return activations


def _collect_gpt_rs_activations(rs_model: Any, tokens: List[int]) -> Dict[str, np.ndarray]:
    return {
        str(item["name"]): np.asarray(item["tensor"], dtype=np.float32)
        for item in rs_model.debug_token_activations(tokens)
    }


def _print_report(
    hf_acts: Dict[str, np.ndarray],
    rs_acts: Dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> int:
    shared = [k for k in rs_acts if k in hf_acts]
    missing_hf = [k for k in rs_acts if k not in hf_acts]
    if missing_hf:
        print(f"warning: names only in gpt-rs: {missing_hf}")

    first_fail = -1
    print(f"{'activation':44} {'shape':18} {'max_abs':>12} {'mean_abs':>12} {'allclose':>10}")
    print("-" * 102)
    for idx, name in enumerate(shared):
        hf = hf_acts[name]
        rs = rs_acts[name]
        if hf.shape != rs.shape:
            print(f"{name:44} shape_mismatch torch={hf.shape} gpt-rs={rs.shape}")
            if first_fail < 0:
                first_fail = idx
            continue
        diff = np.abs(hf - rs)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        ok = bool(np.allclose(hf, rs, rtol=rtol, atol=atol))
        print(f"{name:44} {str(hf.shape):18} {max_abs:12.6e} {mean_abs:12.6e} {str(ok):>10}")
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

    prompt_tokens = encode_prompt(tokenizer, str(args.prompt), int(args.max_prompt_tokens))
    print(f"prompt_tokens={prompt_tokens}")

    rs_acts = _collect_gpt_rs_activations(rs_model, prompt_tokens)
    hf_acts = _collect_hf_activations(
        torch_model, prompt_tokens, str(args.torch_device), rs_acts.keys()
    )
    return _print_report(hf_acts, rs_acts, rtol=float(args.rtol), atol=float(args.atol))


if __name__ == "__main__":
    raise SystemExit(main())
