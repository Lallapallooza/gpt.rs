#!/usr/bin/env python3
"""Compile Triton kernel sources emitted by Rust to PTX using Triton Python API.

The kernel source and metadata live in Rust-owned `.triton` assets under
`crates/gpt-rs-backend-triton/src/kernels/prepacked/`. This script only drives
compilation and metadata emission.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, cast

import triton  # type: ignore[import-untyped]
from triton.backends.compiler import GPUTarget  # type: ignore[import-untyped]
from triton.compiler import ASTSource  # type: ignore[import-untyped]

SCHEMA_VERSION = 1
TOOL_NAME = "tritoncc-python"


class CompileError(RuntimeError):
    """Raised when compile inputs are invalid or unsupported."""


@dataclass(frozen=True)
class KernelSpec:
    kernel_name: str
    symbol: str
    signature: dict[str, str]
    param_abi: list[str]
    constexprs: dict[str, Any]
    num_warps: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="CUDA arch (sm_80 or 80)")
    parser.add_argument("--in", dest="input", required=True, help="Input .triton source file")
    parser.add_argument("--out", dest="output", required=True, help="Output PTX path")
    parser.add_argument("--meta", dest="meta", required=True, help="Output metadata JSON path")
    parser.add_argument(
        "--block-size",
        dest="block_size",
        type=int,
        default=None,
        help="Override BLOCK_SIZE constexpr in metadata (optional)",
    )
    return parser.parse_args(argv)


def parse_cuda_arch(value: str) -> int:
    text = value.strip().lower()
    for prefix in ("sm_", "compute_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if not text.isdigit():
        raise CompileError(f"invalid CUDA arch '{value}', expected sm_XX or numeric XX")
    arch = int(text)
    if arch < 50:
        raise CompileError(f"unsupported CUDA arch {arch}; expected >= 50")
    return arch


def parse_metadata(source: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    prefixes = ("# gpt_rs.", "// gpt_rs.")
    for raw_line in source.splitlines():
        line = raw_line.strip()
        payload = None
        for prefix in prefixes:
            if line.startswith(prefix):
                payload = line[len(prefix) :].strip()
                break
        if payload is None or ":" not in payload:
            continue
        key, value = payload.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def parse_kv_map(payload: str, context: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in payload.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise CompileError(f"invalid {context} item '{token}', expected key=value")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise CompileError(f"invalid {context} item '{token}', empty key/value")
        result[key] = value
    if not result:
        raise CompileError(f"{context} is empty")
    return result


def parse_literal(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value, 0)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def parse_constexprs(payload: str) -> dict[str, Any]:
    raw_map = parse_kv_map(payload, "constexpr")
    return {key: parse_literal(value) for key, value in raw_map.items()}


def parse_kernel_spec(metadata: dict[str, str]) -> KernelSpec:
    kernel_name = metadata.get("kernel", "unknown")
    symbol = metadata.get("symbol")
    if not symbol:
        raise CompileError("missing required metadata key 'gpt_rs.symbol'")

    signature_payload = metadata.get("signature")
    if not signature_payload:
        raise CompileError("missing required metadata key 'gpt_rs.signature'")
    signature = parse_kv_map(signature_payload, "signature")

    param_abi_payload = metadata.get("param_abi")
    if not param_abi_payload:
        raise CompileError("missing required metadata key 'gpt_rs.param_abi'")
    param_abi = [item.strip() for item in param_abi_payload.split(",") if item.strip()]
    if not param_abi:
        raise CompileError("param_abi is empty")

    constexprs: dict[str, Any] = {}
    constexpr_payload = metadata.get("constexpr")
    if constexpr_payload:
        constexprs = parse_constexprs(constexpr_payload)

    num_warps = 8
    if "num_warps" in metadata:
        try:
            num_warps = int(metadata["num_warps"])
        except ValueError as err:
            raise CompileError("num_warps must be integer") from err
        if num_warps <= 0:
            raise CompileError("num_warps must be > 0")

    return KernelSpec(
        kernel_name=kernel_name,
        symbol=symbol,
        signature=signature,
        param_abi=param_abi,
        constexprs=constexprs,
        num_warps=num_warps,
    )


def ptx_version_for_arch(arch: int) -> int:
    if arch >= 90:
        return 80
    if arch >= 80:
        return 76
    return 72


def load_kernel_function(source: str, input_path: pathlib.Path, symbol: str) -> Any:
    scope: dict[str, Any] = {}
    code = compile(source, str(input_path), "exec")
    exec(code, scope, scope)
    fn = scope.get(symbol)
    if fn is None or not callable(fn):
        raise CompileError(f"source does not define callable kernel symbol '{symbol}'")
    return fn


def compile_kernel(
    source: str,
    input_path: pathlib.Path,
    spec: KernelSpec,
    arch: int,
    block_size_override: int | None,
) -> tuple[str, str, dict[str, Any]]:
    constexprs = dict(spec.constexprs)
    if block_size_override is not None:
        constexprs["BLOCK_SIZE"] = block_size_override

    fn = load_kernel_function(source, input_path, spec.symbol)
    source_obj = ASTSource(fn, signature=spec.signature, constexprs=constexprs)
    target = GPUTarget("cuda", arch, 32)
    ptx_version = ptx_version_for_arch(arch)
    compiled = triton.compile(
        source_obj,
        target=target,
        options={"num_warps": spec.num_warps, "ptx_version": ptx_version},
    )

    ptx_blob = compiled.asm.get("ptx")
    if ptx_blob is None:
        raise CompileError("triton compile did not produce PTX output")
    if isinstance(ptx_blob, bytes):
        ptx = ptx_blob.decode("utf-8")
    else:
        ptx = cast(str, ptx_blob)

    return ptx, compiled.name, {"num_warps": spec.num_warps, "ptx_version": ptx_version}


def write_outputs(
    arch_flag: str,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    meta_path: pathlib.Path,
    ptx: str,
    kernel_symbol: str,
    options: dict[str, Any],
    spec: KernelSpec,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ptx, encoding="utf-8")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "tool_version": triton.__version__,
        "arch": arch_flag,
        "input": str(input_path),
        "output": str(output_path),
        "kernel_symbol": kernel_symbol,
        "param_abi": spec.param_abi,
        "shared_mem_bytes": 0,
        "num_warps": options["num_warps"],
        "ptx_version": options["ptx_version"],
        "note": (f"compiled via Python Triton source backend ({spec.kernel_name}:{spec.symbol})"),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    meta_path = pathlib.Path(args.meta)

    source = input_path.read_text(encoding="utf-8")
    metadata = parse_metadata(source)
    spec = parse_kernel_spec(metadata)
    arch = parse_cuda_arch(args.arch)

    ptx, kernel_symbol, options = compile_kernel(
        source,
        input_path,
        spec,
        arch,
        args.block_size,
    )
    write_outputs(
        args.arch,
        input_path,
        output_path,
        meta_path,
        ptx,
        kernel_symbol,
        options,
        spec,
    )

    print(f"compiled {input_path} -> {output_path}")
    print(f"metadata {meta_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CompileError as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1) from err
