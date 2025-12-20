from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def default_gptrs_bin() -> List[str]:
    exe = "gpt-rs-cli.exe" if os.name == "nt" else "gpt-rs-cli"
    return [str(Path("target") / "release" / exe)]


def run_cmd_capture(cmd: Sequence[str], env: Optional[Dict[str, str]] = None) -> Tuple[str, float]:
    t0 = time.perf_counter()
    completed = subprocess.run(
        list(cmd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)
    return completed.stdout, dt


def parse_gptrs_generate(
    stdout: str, fallback_time_s: float, max_tokens: int
) -> Tuple[float, float]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("Generated") and "tokens/sec" in line:
            try:
                parts = line.replace("(", "").replace(")", "").split()
                tokens = float(parts[1])
                time_field = parts[4]
                if "ms" in time_field:
                    time_s = float(time_field.replace("ms", "")) / 1000.0
                else:
                    time_s = float(time_field.replace("s", ""))
                tps = float(parts[5])
                if tokens > 0 and time_s > 0:
                    return time_s, tps
            except Exception:
                continue
    tps = (max_tokens / fallback_time_s) if fallback_time_s > 0 else 0.0
    return fallback_time_s, tps


def gptrs_logits(
    *,
    cmd_prefix: Optional[Sequence[str]],
    prompt: str,
    config: Path,
    tokenizer: Path,
    checkpoint: Path,
    tokens: List[int],
    backend: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    dump_dir: Optional[Path] = None,
    dump_mode: str = "all",
) -> np.ndarray:
    prefix = list(cmd_prefix) if cmd_prefix else default_gptrs_bin()
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        json.dump(tokens, tmp)
        tmp_path = Path(tmp.name)
    try:
        cmd: List[str] = prefix + [
            "generate",
            "--prompt",
            prompt,
            "--max-tokens",
            "0",
            "--config",
            str(config),
            "--tokenizer",
            str(tokenizer),
            "--checkpoint",
            str(checkpoint),
            "--logits-only",
            "--tokens",
            str(tmp_path),
        ]
        if backend:
            cmd += ["--backend", backend]
        if dump_dir is not None:
            cmd += ["--dump-dir", str(dump_dir), "--dump-mode", str(dump_mode)]
        out = subprocess.check_output(cmd, text=True, env=env)
    finally:
        os.unlink(tmp_path)

    data: Any = json.loads(out)
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        row: Iterable[float] = data
    elif isinstance(data, list) and data:
        row = data[-1]
    elif isinstance(data, dict):
        logits = data.get("logits")
        if logits is None:
            raise RuntimeError("gpt-rs-cli response missing logits")
        if isinstance(logits, list) and logits and isinstance(logits[0], (int, float)):
            row = logits
        elif isinstance(logits, list) and logits:
            row = logits[-1]
        else:
            raise RuntimeError("unexpected logits structure from gpt-rs-cli")
    else:
        raise RuntimeError("unexpected payload from gpt-rs-cli")

    return np.array(list(row), dtype=np.float32)
