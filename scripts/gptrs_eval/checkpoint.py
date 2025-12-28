from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, TypedDict

import numpy as np

_MAGIC = b"GPTRSCHK"
_VERSION = 2


class _SaveEntry(TypedDict):
    name: str
    base_id: int
    shape: tuple[int, ...]
    dtype_tag: int
    requires_grad: bool
    offset_rel: int
    byte_len: int
    data: bytes


class _IndexEntry(TypedDict):
    name: str
    base_id: int
    shape: list[int]
    dtype_tag: int
    requires_grad: bool
    offset: int
    byte_len: int


def _write_u32(f, value: int) -> None:
    f.write(struct.pack("<I", int(value)))


def _write_u64(f, value: int) -> None:
    f.write(struct.pack("<Q", int(value)))


def _write_u128(f, value: int) -> None:
    f.write(int(value).to_bytes(16, byteorder="little", signed=False))


def _build_index_bytes(entries: Sequence[_SaveEntry], *, data_start: int) -> bytes:
    out = bytearray()
    out += struct.pack("<I", int(len(entries)))
    for e in entries:
        name_b = e["name"].encode("utf-8")
        out += struct.pack("<I", int(len(name_b)))
        out += name_b

        out += int(e["base_id"]).to_bytes(16, byteorder="little", signed=False)

        shape = e["shape"]
        out += struct.pack("<I", int(len(shape)))
        for dim in shape:
            out += struct.pack("<Q", int(dim))

        out += struct.pack("<I", int(e["dtype_tag"]))
        out += b"\x01" if e["requires_grad"] else b"\x00"

        offset_abs = int(data_start) + int(e["offset_rel"])
        out += struct.pack("<Q", int(offset_abs))
        out += struct.pack("<Q", int(e["byte_len"]))

    return bytes(out)


def save(
    path: Path,
    *,
    kind: str,
    config: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
    requires_grad: Optional[Mapping[str, bool]] = None,
) -> None:
    """Save a self-describing gpt-rs checkpoint (v2 indexed).

    Notes:
      - `base_id` is written as 0 for every tensor (Rust loader computes it from the name).
      - Only supports f32 (dtype tag 0) and i32 (dtype tag 3).
    """

    req = requires_grad or {}
    ordered = sorted(tensors.items(), key=lambda kv: kv[0])

    config_payload = {"kind": str(kind), "config": dict(config)}
    config_bytes = json.dumps(config_payload, separators=(",", ":")).encode("utf-8")

    entries: list[_SaveEntry] = []
    running_offset = 0
    for name, arr in ordered:
        if arr.dtype == np.float32:
            dtype_tag = 0
        elif arr.dtype == np.int32:
            dtype_tag = 3
        else:
            raise ValueError(f"unsupported dtype for {name}: {arr.dtype}")

        arr_c = np.ascontiguousarray(arr)
        shape = tuple(int(x) for x in arr_c.shape)
        data = arr_c.tobytes(order="C")

        entries.append(
            {
                "name": name,
                "base_id": 0,
                "shape": shape,
                "dtype_tag": dtype_tag,
                "requires_grad": bool(req.get(name, False)),
                "offset_rel": running_offset,
                "byte_len": len(data),
                "data": data,
            }
        )
        running_offset += len(data)

    index_bytes_rel = _build_index_bytes(entries, data_start=0)
    index_len = len(index_bytes_rel)
    data_start = len(_MAGIC) + 4 + 4 + len(config_bytes) + 4 + index_len
    index_bytes_abs = _build_index_bytes(entries, data_start=data_start)
    if len(index_bytes_abs) != len(index_bytes_rel):
        raise ValueError("checkpoint index length mismatch after offset fixup")

    with open(path, "wb") as f:
        f.write(_MAGIC)
        _write_u32(f, _VERSION)

        _write_u32(f, len(config_bytes))
        f.write(config_bytes)

        _write_u32(f, index_len)
        f.write(index_bytes_abs)

        for e in entries:
            f.write(e["data"])


def load(path: Path) -> Dict[str, Any]:
    """Load a checkpoint written by `save` (debug/helper)."""

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError("invalid checkpoint magic header")
        (version,) = struct.unpack("<I", f.read(4))
        if int(version) != _VERSION:
            raise ValueError(f"unsupported checkpoint version {version}")

        (config_len,) = struct.unpack("<I", f.read(4))
        config_bytes = f.read(int(config_len))
        config = json.loads(config_bytes.decode("utf-8"))

        (index_len,) = struct.unpack("<I", f.read(4))
        index_bytes = f.read(int(index_len))

        pos = 0
        (count,) = struct.unpack_from("<I", index_bytes, pos)
        pos += 4

        entries: list[_IndexEntry] = []
        for _ in range(int(count)):
            (name_len,) = struct.unpack_from("<I", index_bytes, pos)
            pos += 4
            name = index_bytes[pos : pos + int(name_len)].decode("utf-8")
            pos += int(name_len)

            base_id = int.from_bytes(index_bytes[pos : pos + 16], byteorder="little", signed=False)
            pos += 16

            (rank,) = struct.unpack_from("<I", index_bytes, pos)
            pos += 4
            shape: list[int] = []
            for _ in range(int(rank)):
                (dim,) = struct.unpack_from("<Q", index_bytes, pos)
                pos += 8
                shape.append(int(dim))

            (dtype_tag,) = struct.unpack_from("<I", index_bytes, pos)
            pos += 4
            requires_grad = bool(index_bytes[pos])
            pos += 1

            (offset,) = struct.unpack_from("<Q", index_bytes, pos)
            pos += 8
            (byte_len,) = struct.unpack_from("<Q", index_bytes, pos)
            pos += 8

            entries.append(
                {
                    "name": name,
                    "base_id": base_id,
                    "shape": shape,
                    "dtype_tag": int(dtype_tag),
                    "requires_grad": requires_grad,
                    "offset": int(offset),
                    "byte_len": int(byte_len),
                }
            )

        tensors: Dict[str, np.ndarray] = {}
        for e in entries:
            f.seek(int(e["offset"]))
            raw = f.read(int(e["byte_len"]))
            if e["dtype_tag"] == 0:
                tensors[e["name"]] = np.frombuffer(raw, dtype=np.float32).reshape(e["shape"])
            elif e["dtype_tag"] == 3:
                tensors[e["name"]] = np.frombuffer(raw, dtype=np.int32).reshape(e["shape"])
            else:
                raise ValueError(f"unsupported dtype tag {e['dtype_tag']} for {e['name']}")

        return {"config": config, "entries": entries, "tensors": tensors}
