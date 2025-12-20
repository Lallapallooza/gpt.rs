from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

_MAGIC = b"GPTRSTEN"
_VERSION = 1


def _write_u32(f, value: int) -> None:
    f.write(struct.pack("<I", int(value)))


def _write_u64(f, value: int) -> None:
    f.write(struct.pack("<Q", int(value)))


def save(
    path: Path,
    tensors: Mapping[str, np.ndarray],
    *,
    requires_grad: Optional[Mapping[str, bool]] = None,
) -> None:
    """Save named tensors to a gpt-rs tensor archive.

    The format is intentionally minimal and only supports:
      - f32 (dtype tag 0)
      - i32 (dtype tag 3)
    """

    req = requires_grad or {}
    ordered = sorted(tensors.items(), key=lambda kv: kv[0])

    with open(path, "wb") as f:
        f.write(_MAGIC)
        _write_u32(f, _VERSION)
        _write_u32(f, len(ordered))

        for name, arr in ordered:
            name_b = name.encode("utf-8")
            _write_u32(f, len(name_b))
            f.write(name_b)

            if arr.dtype == np.float32:
                dtype_tag = 0
            elif arr.dtype == np.int32:
                dtype_tag = 3
            else:
                raise ValueError(f"unsupported dtype for {name}: {arr.dtype}")

            arr_c = np.ascontiguousarray(arr)
            shape = tuple(int(x) for x in arr_c.shape)

            _write_u32(f, len(shape))
            for dim in shape:
                _write_u64(f, dim)

            _write_u32(f, dtype_tag)
            f.write(b"\x01" if bool(req.get(name, False)) else b"\x00")

            data = arr_c.tobytes(order="C")
            _write_u64(f, len(data))
            f.write(data)


def load(path: Path) -> Dict[str, np.ndarray]:
    """Load a tensor archive written by `save` (debug/helper)."""

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError("invalid tensor archive magic header")
        (version,) = struct.unpack("<I", f.read(4))
        if int(version) != _VERSION:
            raise ValueError(f"unsupported tensor archive version {version}")
        (count,) = struct.unpack("<I", f.read(4))

        out: Dict[str, np.ndarray] = {}
        for _ in range(int(count)):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(int(name_len)).decode("utf-8")

            (rank,) = struct.unpack("<I", f.read(4))
            shape = []
            for _ in range(int(rank)):
                (dim,) = struct.unpack("<Q", f.read(8))
                shape.append(int(dim))

            (dtype_tag,) = struct.unpack("<I", f.read(4))
            _requires_grad = f.read(1)  # ignored in Python

            (byte_len,) = struct.unpack("<Q", f.read(8))
            raw = f.read(int(byte_len))

            if int(dtype_tag) == 0:
                out[name] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif int(dtype_tag) == 3:
                out[name] = np.frombuffer(raw, dtype=np.int32).reshape(shape)
            else:
                raise ValueError(f"unsupported dtype tag {dtype_tag} for {name}")

    return out
