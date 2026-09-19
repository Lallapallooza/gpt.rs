from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

_MAGIC = b"GPTRSCHK"
_VERSION = 2

# The writer pads tensor payload offsets to this boundary, so readers can hand the memory-mapped
# bytes straight to SIMD kernels.
_DATA_ALIGNMENT = 64

_DTYPE_TAGS: Dict[str, int] = {"f32": 0, "f16": 1, "bf16": 2, "i32": 3}
_DTYPE_SIZES: Dict[str, int] = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4}


@dataclass(frozen=True)
class TensorPlan:
    """Name, shape and dtype of a checkpoint tensor whose bytes are produced later."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def byte_len(self) -> int:
        return math.prod(int(dim) for dim in self.shape) * _DTYPE_SIZES[self.dtype]


def _write_u32(f: BinaryIO, value: int) -> None:
    f.write(struct.pack("<I", int(value)))


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _build_index_bytes(
    plans: Sequence[TensorPlan],
    offsets: Sequence[int],
    requires_grad: Mapping[str, bool],
) -> bytes:
    out = bytearray()
    out += struct.pack("<I", int(len(plans)))
    for plan, offset in zip(plans, offsets):
        name_b = plan.name.encode("utf-8")
        out += struct.pack("<I", int(len(name_b)))
        out += name_b
        # base_id 0: the Rust loader derives it from the name.
        out += (0).to_bytes(16, byteorder="little", signed=False)
        out += struct.pack("<I", int(len(plan.shape)))
        for dim in plan.shape:
            out += struct.pack("<Q", int(dim))
        out += struct.pack("<I", _DTYPE_TAGS[plan.dtype])
        out += b"\x01" if requires_grad.get(plan.name, False) else b"\x00"
        out += struct.pack("<Q", int(offset))
        out += struct.pack("<Q", int(plan.byte_len))
    return bytes(out)


def write_streaming(
    path: Path,
    *,
    kind: str,
    config: Mapping[str, Any],
    plans: Iterable[TensorPlan],
    produce: Callable[[TensorPlan], bytes | memoryview | np.ndarray],
    requires_grad: Optional[Mapping[str, bool]] = None,
    eos_token_ids: Sequence[int] = (),
) -> None:
    """Write a gpt-rs checkpoint one tensor at a time.

    `produce(plan)` runs once per tensor, in name order, and must return exactly `plan.byte_len`
    little-endian bytes, so peak memory stays at one tensor. The file is written next to `path` and
    renamed into place, because readers memory-map checkpoints and an in-place rewrite would
    corrupt their view of it.
    """

    req = requires_grad or {}
    ordered = sorted(plans, key=lambda plan: plan.name)
    names = [plan.name for plan in ordered]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor names in checkpoint plan")
    for plan in ordered:
        if plan.dtype not in _DTYPE_TAGS:
            raise ValueError(f"unsupported dtype {plan.dtype!r} for {plan.name}")

    config_payload: Dict[str, Any] = {"kind": str(kind), "config": dict(config)}
    if eos_token_ids:
        config_payload["eos_token_ids"] = [int(token) for token in eos_token_ids]
    config_bytes = json.dumps(config_payload, separators=(",", ":")).encode("utf-8")

    placeholder = [0] * len(ordered)
    index_len = len(_build_index_bytes(ordered, placeholder, req))
    header_len = len(_MAGIC) + 4 + 4 + len(config_bytes) + 4 + index_len

    offsets: list[int] = []
    cursor = header_len
    for plan in ordered:
        cursor = _align_up(cursor, _DATA_ALIGNMENT)
        offsets.append(cursor)
        cursor += plan.byte_len
    index_bytes = _build_index_bytes(ordered, offsets, req)

    tmp_path = Path(path).with_name(f"{Path(path).name}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(_MAGIC)
            _write_u32(f, _VERSION)
            _write_u32(f, len(config_bytes))
            f.write(config_bytes)
            _write_u32(f, index_len)
            f.write(index_bytes)

            for plan, offset in zip(ordered, offsets):
                f.write(b"\x00" * (offset - f.tell()))
                payload = produce(plan)
                if isinstance(payload, np.ndarray):
                    payload = np.ascontiguousarray(payload)
                view = memoryview(payload).cast("B")
                if view.nbytes != plan.byte_len:
                    raise ValueError(
                        f"tensor {plan.name}: produced {view.nbytes} bytes, "
                        f"expected {plan.byte_len}"
                    )
                f.write(view)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _numpy_dtype_name(arr: np.ndarray, name: str) -> str:
    if arr.dtype == np.float32:
        return "f32"
    if arr.dtype == np.int32:
        return "i32"
    raise ValueError(f"unsupported dtype for {name}: {arr.dtype}")


def save(
    path: Path,
    *,
    kind: str,
    config: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
    requires_grad: Optional[Mapping[str, bool]] = None,
    eos_token_ids: Sequence[int] = (),
) -> None:
    """Save a gpt-rs checkpoint from in-memory f32 or i32 numpy arrays.

    Other dtypes go through `write_streaming`.
    """

    arrays = {name: np.ascontiguousarray(arr) for name, arr in tensors.items()}
    plans = [
        TensorPlan(
            name=name,
            shape=tuple(int(x) for x in arr.shape),
            dtype=_numpy_dtype_name(arr, name),
        )
        for name, arr in arrays.items()
    ]
    write_streaming(
        path,
        kind=kind,
        config=config,
        plans=plans,
        produce=lambda plan: arrays[plan.name],
        requires_grad=requires_grad,
        eos_token_ids=eos_token_ids,
    )


def hf_eos_token_ids(*configs: Optional[Mapping[str, Any]]) -> list[int]:
    """The deduplicated `eos_token_id` values of Hugging Face configs, in order.

    Pass `generation_config` first: its ids are what Hugging Face `generate` stops on.
    """

    ids: list[int] = []
    for cfg in configs:
        value = (cfg or {}).get("eos_token_id")
        for token in value if isinstance(value, list) else [value]:
            if token is not None and int(token) not in ids:
                ids.append(int(token))
    return ids


_HF_TEXT_PREFIXES = ("model.language_model.", "model.")


def hf_text_tensor_names(names: Iterable[str]) -> Dict[str, str]:
    """gpt-rs name -> Hugging Face name for the text decoder of a Hugging Face checkpoint.

    The decoder prefix becomes `model.`, so the names equal those of the text model. Of the other
    modules, only `lm_head.weight` is kept.
    """

    names = list(names)
    prefix = next(
        (p for p in _HF_TEXT_PREFIXES if f"{p}embed_tokens.weight" in names),
        None,
    )
    if prefix is None:
        raise KeyError("cannot find the text model (embed_tokens.weight) in the checkpoint")
    mapped = {"model." + n[len(prefix) :]: n for n in names if n.startswith(prefix)}
    if "lm_head.weight" in names:
        mapped["lm_head.weight"] = "lm_head.weight"
    return mapped


def read_config(path: Path) -> Dict[str, Any]:
    """Read only the `{kind, config}` header of a checkpoint."""

    with open(path, "rb") as f:
        if f.read(8) != _MAGIC:
            raise ValueError("invalid checkpoint magic header")
        (version,) = struct.unpack("<I", f.read(4))
        if int(version) != _VERSION:
            raise ValueError(f"unsupported checkpoint version {version}")
        (config_len,) = struct.unpack("<I", f.read(4))
        payload: Dict[str, Any] = json.loads(f.read(int(config_len)).decode("utf-8"))
        return payload
