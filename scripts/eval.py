#!/usr/bin/env python3
"""Unified validation/bench runner for gpt.rs models.

Models are provided via small "case" adapters under `scripts/gptrs_eval/models/`.
This script:
  - runs gpt-rs (default backend: faer) and Torch baselines
  - can validate logits and benchmark performance
  - can run a suite (JSON/YAML) to define multiple runs/sweeps
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _set_thread_env(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["RAYON_NUM_THREADS"] = str(threads)


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    return obj


def _load_suite(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found, import-untyped]
        except ImportError as err:
            raise SystemExit("YAML suite requires PyYAML (pip install pyyaml).") from err
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("suite root must be an object/mapping")
    return data


def _bench_child(cfg_dict: Dict[str, Any]) -> int:
    from gptrs_eval.core import RunConfig
    from gptrs_eval.registry import get_case
    from gptrs_eval.report import dumps_json

    cfg = RunConfig(
        model=str(cfg_dict["model"]),
        backend=str(cfg_dict.get("backend", "faer")),
        torch_device=str(cfg_dict.get("torch_device", "cpu")),
        seed=int(cfg_dict.get("seed", 0)),
        rtol=float(cfg_dict.get("rtol", 1e-4)),
        atol=float(cfg_dict.get("atol", 1e-4)),
        warmup=int(cfg_dict.get("warmup", 1)),
        iters=int(cfg_dict.get("iters", 3)),
        threads=int(cfg_dict.get("threads", 1)),
        params=dict(cfg_dict.get("params", {})),
    )

    _set_thread_env(cfg.threads)
    case = get_case(cfg.model)
    result = case.bench(cfg)
    print(dumps_json([result]))
    return 0


def _spawn_bench(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_bench-child",
        "--child-config",
        json.dumps(_json_sanitize(cfg_dict)),
        "--format",
        "json",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise SystemExit("bench child returned unexpected JSON")
    return payload[0]


def _bench_result_from_dict(d: Dict[str, Any]) -> Any:
    from gptrs_eval.core import BenchResult, BenchStats

    return BenchResult(
        model=str(d["model"]),
        threads=int(d["threads"]),
        unit_label=str(d["unit_label"]),
        units_per_iter=float(d["units_per_iter"]),
        gptrs=BenchStats(
            impl=str(d["gptrs"]["impl"]),
            times_s=[float(x) for x in d["gptrs"]["times_s"]],
            mean_s=float(d["gptrs"]["mean_s"]),
            units_per_s=float(d["gptrs"]["units_per_s"]),
            extra=dict(d["gptrs"].get("extra", {})),
        ),
        torch=BenchStats(
            impl=str(d["torch"]["impl"]),
            times_s=[float(x) for x in d["torch"]["times_s"]],
            mean_s=float(d["torch"]["mean_s"]),
            units_per_s=float(d["torch"]["units_per_s"]),
            extra=dict(d["torch"].get("extra", {})),
        ),
        extra=dict(d.get("extra", {})),
    )


def _add_base_args(parser: argparse.ArgumentParser, *, include_hidden: bool) -> None:
    parser.add_argument("--model", help="Model case to run")
    parser.add_argument(
        "--workload",
        choices=["validate", "bench", "run", "all"],
        default="all",
        help="What to run (default: all supported)",
    )
    parser.add_argument("--backend", default="faer", help="gpt-rs backend (default: faer)")
    parser.add_argument("--torch-device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--rtol", type=float, default=1e-4, help="np.allclose rtol")
    parser.add_argument("--atol", type=float, default=1e-4, help="np.allclose atol")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations for bench")
    parser.add_argument("--iters", type=int, default=3, help="Measured iterations for bench")
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[1], help="Thread counts to bench"
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="Dump PTIR programs and metadata to this directory (gpt-rs side).",
    )
    parser.add_argument(
        "--dump-mode",
        choices=["all", "compile"],
        default="all",
        help="Controls which PTIR executions are dumped (default: all).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print gpt-rs profiler tables when available (requires profiler build).",
    )
    parser.add_argument(
        "--profile-json",
        action="store_true",
        help="Also emit profiler report as JSON (writes to dump_dir when set).",
    )
    parser.add_argument(
        "--profile-trace",
        action="store_true",
        help="Capture a Chrome trace of profiler scopes (writes to dump_dir when set).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format",
    )
    parser.add_argument("--json-out", type=Path, help="Write results JSON to this path")

    # Suite controls (pre-parse only)
    parser.add_argument("--suite", type=Path, help="JSON/YAML suite file")
    parser.add_argument("--list-models", action="store_true", help="List available model cases")

    if include_hidden:
        parser.add_argument("--_bench-child", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--child-config", help=argparse.SUPPRESS)


def _build_full_parser(model: str) -> argparse.ArgumentParser:
    from gptrs_eval.registry import get_case

    parser = argparse.ArgumentParser(description="gpt.rs eval runner")
    _add_base_args(parser, include_hidden=False)
    case = get_case(model)
    case.add_cli_args(parser)
    return parser


def _namespace_to_cfg_dict(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[int]]:
    raw = vars(args).copy()
    model = raw.pop("model")
    if not isinstance(model, str) or not model:
        raise SystemExit("--model is required (or use --suite)")

    threads_list = raw.pop("threads")
    if not isinstance(threads_list, list) or not threads_list:
        threads_list = [1]
    threads_list = [int(x) for x in threads_list]

    cfg: Dict[str, Any] = {
        "model": model,
        "backend": raw.pop("backend"),
        "torch_device": raw.pop("torch_device"),
        "seed": raw.pop("seed"),
        "rtol": raw.pop("rtol"),
        "atol": raw.pop("atol"),
        "warmup": raw.pop("warmup"),
        "iters": raw.pop("iters"),
        "params": {},
    }

    # Drop non-config keys.
    raw.pop("suite", None)
    raw.pop("list_models", None)
    raw.pop("workload", None)
    raw.pop("format", None)
    raw.pop("json_out", None)

    # Remaining keys are model params.
    cfg["params"].update(raw)
    return cfg, threads_list


def _run_case(cfg_dict: Dict[str, Any], threads_list: List[int], workload: str) -> List[Any]:
    from gptrs_eval.core import RunConfig
    from gptrs_eval.registry import get_case

    case = get_case(str(cfg_dict["model"]))
    if workload == "all":
        workloads = case.supported_workloads()
    else:
        workloads = [workload]

    results: List[Any] = []
    for w in workloads:
        if w == "bench":
            # Optional token sweep for GPT-style models.
            bench_tokens = cfg_dict.get("params", {}).get("bench_tokens")
            if isinstance(bench_tokens, list):
                tokens_list: List[Optional[int]] = [int(x) for x in bench_tokens]
            elif bench_tokens is None:
                tokens_list = [None]
            else:
                tokens_list = [int(bench_tokens)]

            for threads in threads_list:
                for tok in tokens_list:
                    child_cfg = dict(cfg_dict)
                    child_cfg["threads"] = int(threads)
                    child_cfg["params"] = dict(cfg_dict.get("params", {}))
                    if tok is not None:
                        child_cfg["params"]["bench_tokens"] = int(tok)
                    res = _bench_result_from_dict(_spawn_bench(child_cfg))
                    results.append(res)
            continue

        cfg = RunConfig(
            model=str(cfg_dict["model"]),
            backend=str(cfg_dict.get("backend", "faer")),
            torch_device=str(cfg_dict.get("torch_device", "cpu")),
            seed=int(cfg_dict.get("seed", 0)),
            rtol=float(cfg_dict.get("rtol", 1e-4)),
            atol=float(cfg_dict.get("atol", 1e-4)),
            warmup=int(cfg_dict.get("warmup", 1)),
            iters=int(cfg_dict.get("iters", 3)),
            threads=int(threads_list[0]) if threads_list else 1,
            params=dict(cfg_dict.get("params", {})),
        )

        if w == "validate":
            results.append(case.validate(cfg))
        elif w == "run":
            results.append(case.run(cfg))
        else:
            raise SystemExit(f"unknown workload: {w}")

    return results


def _print_results(results: List[Any], *, fmt: str) -> None:
    from gptrs_eval.core import BenchResult, CliRunResult, ValidationResult
    from gptrs_eval.report import print_bench, print_validation

    bench: List[BenchResult] = []
    for r in results:
        if isinstance(r, ValidationResult):
            print_validation(r)
        elif isinstance(r, BenchResult):
            bench.append(r)
        elif isinstance(r, CliRunResult):
            print(f"model={r.model} impl={r.impl} exit_code={r.exit_code} wall_s={r.wall_s:.3f}")
        else:
            print(r)

    if bench:
        print_bench(bench, fmt=fmt)  # type: ignore[arg-type]


def _all_ok(results: List[Any]) -> bool:
    from gptrs_eval.core import CliRunResult, ValidationResult

    ok = True
    for r in results:
        if isinstance(r, ValidationResult) and not r.ok:
            ok = False
        if isinstance(r, CliRunResult) and r.exit_code != 0:
            ok = False
    return ok


def _run_suite(path: Path, *, fmt: str, json_out: Optional[Path]) -> int:
    from gptrs_eval.report import dumps_json

    suite = _load_suite(path)
    defaults = suite.get("defaults", {})
    runs = suite.get("runs")
    if not isinstance(runs, list):
        raise SystemExit("suite.runs must be a list")

    results: List[Any] = []
    for run in runs:
        if not isinstance(run, dict):
            raise SystemExit("each suite run must be a mapping")
        model = run.get("model")
        if not isinstance(model, str):
            raise SystemExit("suite run missing string 'model'")

        workload = run.get("workload", "all")
        workloads = workload if isinstance(workload, list) else [workload]

        cfg: Dict[str, Any] = dict(defaults)
        cfg.update({k: v for k, v in run.items() if k not in {"params", "workload"}})
        cfg["model"] = model

        cfg.setdefault("backend", "faer")
        cfg.setdefault("torch_device", "cpu")
        cfg.setdefault("seed", 0)
        cfg.setdefault("rtol", 1e-4)
        cfg.setdefault("atol", 1e-4)
        cfg.setdefault("warmup", 1)
        cfg.setdefault("iters", 3)
        cfg.setdefault("threads", [1])

        cfg["params"] = dict(defaults.get("params", {}))
        params_override = run.get("params", {})
        if isinstance(params_override, dict):
            cfg["params"].update(params_override)

        threads = cfg["threads"]
        if isinstance(threads, int):
            threads_list = [threads]
        elif isinstance(threads, list):
            threads_list = [int(x) for x in threads]
        else:
            raise SystemExit("suite threads must be int or list[int]")

        for w in workloads:
            results.extend(_run_case(cfg, threads_list, str(w)))

    if fmt == "json":
        out = dumps_json(results)
        print(out)
        if json_out:
            json_out.write_text(out, encoding="utf-8")
        return 0 if _all_ok(results) else 1

    _print_results(results, fmt=fmt)
    if json_out:
        from gptrs_eval.report import dumps_json

        json_out.write_text(dumps_json(results), encoding="utf-8")
    return 0 if _all_ok(results) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    _add_base_args(pre, include_hidden=True)
    args0, _unknown = pre.parse_known_args(list(argv) if argv is not None else None)

    if getattr(args0, "_bench_child", False):
        if not args0.child_config:
            raise SystemExit("--child-config is required with --_bench-child")
        cfg = json.loads(args0.child_config)
        if not isinstance(cfg, dict):
            raise SystemExit("child-config must be a JSON object")
        return _bench_child(cfg)

    if args0.list_models:
        from gptrs_eval.registry import list_models

        for name in list_models():
            print(name)
        return 0

    if args0.suite:
        return _run_suite(args0.suite, fmt=str(args0.format), json_out=args0.json_out)

    if not args0.model:
        raise SystemExit("--model is required (or use --suite)")

    parser = _build_full_parser(args0.model)
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_dict, threads_list = _namespace_to_cfg_dict(args)
    results = _run_case(cfg_dict, threads_list, str(args.workload))

    from gptrs_eval.report import dumps_json

    if str(args.format) == "json":
        out = dumps_json(results)
        print(out)
        if args.json_out:
            args.json_out.write_text(out, encoding="utf-8")
        return 0 if _all_ok(results) else 1

    _print_results(results, fmt=str(args.format))
    if args.json_out:
        args.json_out.write_text(dumps_json(results), encoding="utf-8")
    return 0 if _all_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
