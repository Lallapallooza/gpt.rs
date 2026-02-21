#!/usr/bin/env python3
"""Compile known kernel markers to PTX using Triton Python API.

This tool is intentionally marker-driven to keep the contract stable while the
native MLIR-based compiler is being built.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from triton.backends.compiler import GPUTarget  # type: ignore[import-untyped]
from triton.compiler import ASTSource  # type: ignore[import-untyped]

SCHEMA_VERSION = 1
TOOL_NAME = "tritoncc-python"


class CompileError(RuntimeError):
    """Raised when compile inputs are invalid or unsupported."""


@dataclass(frozen=True)
class KernelSpec:
    marker: str
    symbol: str
    signature: dict[str, str]
    param_abi: list[str]
    num_warps: int = 8


KERNEL_SPECS: dict[str, KernelSpec] = {
    "elementwise_binary_f32": KernelSpec(
        marker="// gpt_rs.kernel: elementwise_binary_f32",
        symbol="gpt_rs_triton_ewise_binary_f32",
        signature={
            "lhs_ptr": "*fp32",
            "rhs_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n": "i32",
            "op": "i32",
        },
        param_abi=["*fp32", "*fp32", "*fp32", "u32", "u32", "*opaque"],
    ),
    "elementwise_unary_f32": KernelSpec(
        marker="// gpt_rs.kernel: elementwise_unary_f32",
        symbol="gpt_rs_triton_ewise_unary_f32",
        signature={
            "in_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n": "i32",
            "op": "i32",
        },
        param_abi=["*fp32", "*fp32", "u32", "u32", "*opaque"],
    ),
}


@triton.jit
def gpt_rs_triton_ewise_binary_f32(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    n,
    op,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    lhs = tl.load(lhs_ptr + offs, mask=mask, other=0.0)
    rhs = tl.load(rhs_ptr + offs, mask=mask, other=0.0)

    add = lhs + rhs
    sub = lhs - rhs
    mul = lhs * rhs
    div = lhs / rhs
    mx = tl.maximum(lhs, rhs)
    mn = tl.minimum(lhs, rhs)

    out = tl.where(
        op == 0,
        add,
        tl.where(
            op == 1,
            sub,
            tl.where(op == 2, mul, tl.where(op == 3, div, tl.where(op == 4, mx, mn))),
        ),
    )
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def gpt_rs_triton_ewise_unary_f32(in_ptr, out_ptr, n, op, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    neg = -x
    absv = tl.abs(x)
    out = tl.where(op == 0, neg, absv)
    tl.store(out_ptr + offs, out, mask=mask)


KERNEL_FNS = {
    "elementwise_binary_f32": gpt_rs_triton_ewise_binary_f32,
    "elementwise_unary_f32": gpt_rs_triton_ewise_unary_f32,
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="CUDA arch (sm_80 or 80)")
    parser.add_argument("--in", dest="input", required=True, help="Input .triton marker file")
    parser.add_argument("--out", dest="output", required=True, help="Output PTX path")
    parser.add_argument("--meta", dest="meta", required=True, help="Output metadata JSON path")
    parser.add_argument(
        "--block-size",
        dest="block_size",
        type=int,
        default=256,
        help="constexpr BLOCK_SIZE used for marker kernels",
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


def detect_kernel_kind(source: str) -> str:
    for kind, spec in KERNEL_SPECS.items():
        if spec.marker in source:
            return kind
    markers = ", ".join(spec.marker for spec in KERNEL_SPECS.values())
    raise CompileError("unsupported kernel marker in input source; expected one of: " + markers)


def compile_kernel(
    kind: str, arch: int, block_size: int
) -> tuple[str, str, dict[str, object], KernelSpec]:
    spec = KERNEL_SPECS[kind]
    fn = KERNEL_FNS[kind]
    source = ASTSource(fn, signature=spec.signature, constexprs={"BLOCK_SIZE": block_size})
    target = GPUTarget("cuda", arch, 32)
    compiled = triton.compile(source, target=target, options={"num_warps": spec.num_warps})

    ptx_blob = compiled.asm.get("ptx")
    if ptx_blob is None:
        raise CompileError("triton compile did not produce PTX output")
    if isinstance(ptx_blob, bytes):
        ptx = ptx_blob.decode("utf-8")
    else:
        ptx = ptx_blob

    kernel_symbol = compiled.name
    return ptx, kernel_symbol, {"num_warps": spec.num_warps, "block_size": block_size}, spec


def write_outputs(
    arch_flag: str,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    meta_path: pathlib.Path,
    ptx: str,
    kernel_symbol: str,
    options: dict[str, object],
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
        "note": f"compiled via Python Triton marker backend ({spec.symbol})",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    meta_path = pathlib.Path(args.meta)

    source = input_path.read_text(encoding="utf-8")
    kind = detect_kernel_kind(source)
    arch = parse_cuda_arch(args.arch)
    ptx, kernel_symbol, options, spec = compile_kernel(kind, arch, args.block_size)
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
