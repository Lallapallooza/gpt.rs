#!/usr/bin/env python3
"""Rebuild the `gpt_rs` Python bindings into the repo's uv environment.

This is a small convenience wrapper around the documented flow:
  uv sync
  uv pip install maturin
  cd crates/gpt-rs-py && uv run maturin develop --release --features faer && cd ../..

Examples:
  uv run python scripts/rebuild_py.py --sync
  uv run python scripts/rebuild_py.py --features faer,profiler
  uv run python scripts/rebuild_py.py --no-release --no-test
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_crate_dir(root: Path) -> Path:
    return root / "crates" / "gpt-rs-py"


def _which(prog: str) -> Optional[str]:
    return shutil.which(prog)


def _run(cmd: Sequence[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def _split_features(raw: str) -> List[str]:
    parts = [p for p in raw.replace(",", " ").split() if p]
    out: List[str] = []
    seen = set()
    for item in parts:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild gpt_rs Python bindings via uv+maturin.")
    parser.add_argument(
        "--sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run `uv sync` before building (default: false).",
    )
    parser.add_argument(
        "--install-maturin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run `uv pip install maturin` before building (default: false).",
    )
    parser.add_argument(
        "--release",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build in release mode (default: true).",
    )
    parser.add_argument(
        "--features",
        default="faer",
        help='Comma/space-separated Cargo features for gpt-rs-py (default: "faer").',
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `crates/gpt-rs-py/test_install.py` after building (default: true).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    root = _repo_root()
    py_crate = _python_crate_dir(root)
    if not py_crate.is_dir():
        raise SystemExit(f"missing python crate directory: {py_crate}")

    if _which("uv") is None:
        raise SystemExit(
            "uv not found in PATH. Install uv, or run manually:\n"
            "  uv sync\n"
            "  uv pip install maturin\n"
            "  cd crates/gpt-rs-py && uv run maturin develop --release --features faer\n"
        )

    if args.sync:
        _run(["uv", "sync"], cwd=root)

    if args.install_maturin:
        _run(["uv", "pip", "install", "maturin"], cwd=root)

    cmd = ["uv", "run", "maturin", "develop"]
    if bool(args.release):
        cmd.append("--release")

    features = _split_features(str(args.features))
    if features:
        cmd.extend(["--features", ",".join(features)])

    _run(cmd, cwd=py_crate)

    if bool(args.test):
        _run(["uv", "run", "python", "crates/gpt-rs-py/test_install.py"], cwd=root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
