#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from exporters import build_request, get_exporter, iter_exporter_infos, list_exporter_names
from exporters.pipeline import run_export, run_validation
from exporters.types import ExporterInfo


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _info_to_dict(info: ExporterInfo) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": info.name,
        "kind": info.kind,
        "description": info.description,
        "artifacts": {
            "checkpoint": info.artifacts.checkpoint,
            "config": info.artifacts.config,
            "tokenizer": info.artifacts.tokenizer,
        },
    }
    if info.eval_case is not None:
        payload["eval_case"] = {
            "model_name": info.eval_case.model_name,
            "module": info.eval_case.module,
            "cls": info.eval_case.cls,
            "default_params": dict(info.eval_case.default_params),
        }
    return _json_sanitize(payload)


def _build_base_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified exporter CLI for gpt.rs", add_help=add_help
    )
    parser.add_argument("command", choices=["list", "inspect", "export", "validate"])
    parser.add_argument("--exporter", help="Exporter name")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--checkpoint-out", type=Path, help="Override checkpoint output path")
    parser.add_argument("--config-out", type=Path, help="Override config output path")
    parser.add_argument("--tokenizer-out", type=Path, help="Override tokenizer output path")
    return parser


def _build_export_parser(command: str, exporter_name: str) -> argparse.ArgumentParser:
    exporter = get_exporter(exporter_name)
    parser = argparse.ArgumentParser(description=f"{command} artifacts for {exporter_name}")
    parser.add_argument("command", choices=[command])
    parser.add_argument("--exporter", required=True, choices=list_exporter_names())
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--checkpoint-out", type=Path, help="Override checkpoint output path")
    parser.add_argument("--config-out", type=Path, help="Override config output path")
    parser.add_argument("--tokenizer-out", type=Path, help="Override tokenizer output path")
    exporter.add_arguments(parser)
    return parser


def parse_args(argv: List[str]) -> argparse.Namespace:
    if not argv or (len(argv) == 1 and argv[0] in {"-h", "--help"}):
        return _build_base_parser().parse_args(argv)

    base = _build_base_parser(add_help=False)
    known, _remaining = base.parse_known_args(argv)

    if known.command in {"export", "validate"}:
        if not known.exporter:
            if any(arg in {"-h", "--help"} for arg in argv):
                return _build_base_parser().parse_args(["--help"])
            base.error("--exporter is required for export/validate commands")
        parser = _build_export_parser(known.command, known.exporter)
        return parser.parse_args(argv)

    if known.command == "inspect":
        parser = argparse.ArgumentParser(description="Inspect exporter metadata")
        parser.add_argument("command", choices=["inspect"])
        parser.add_argument("--exporter", choices=list_exporter_names())
        parser.add_argument("--format", choices=["table", "json"], default="table")
        return parser.parse_args(argv)

    parser = argparse.ArgumentParser(description="List available exporters")
    parser.add_argument("command", choices=["list"])
    parser.add_argument("--format", choices=["table", "json"], default="table")
    return parser.parse_args(argv)


def _render_infos(infos: List[ExporterInfo], fmt: str) -> None:
    payload = [_info_to_dict(info) for info in infos]
    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return

    for info in infos:
        print(f"{info.name:<16} kind={info.kind:<12} {info.description}")


def _render_export_result(result: Dict[str, Any], fmt: str) -> None:
    payload = _json_sanitize(result)
    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return
    print(f"exporter={payload['exporter']} kind={payload['kind']}")
    print(f"checkpoint={payload['checkpoint']}")
    print(f"tensors={payload['tensor_count']}")
    config = payload.get("config")
    tokenizer = payload.get("tokenizer")
    if config:
        print(f"config={config}")
    if tokenizer:
        print(f"tokenizer={tokenizer}")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    command = str(args.command)

    if command == "list":
        infos = list(iter_exporter_infos())
        _render_infos(infos, str(args.format))
        return 0

    if command == "inspect":
        if args.exporter:
            info = get_exporter(str(args.exporter)).info
            _render_infos([info], str(args.format))
            return 0
        infos = list(iter_exporter_infos())
        _render_infos(infos, str(args.format))
        return 0

    exporter = get_exporter(str(args.exporter))
    request = build_request(exporter, args)

    if command == "validate":
        errors = run_validation(exporter, request)
        if errors:
            for err in errors:
                print(f"validation error: {err}", file=sys.stderr)
            return 1
        print(f"validation ok for exporter {exporter.info.name}")
        return 0

    result = run_export(exporter, request)
    _render_export_result(
        {
            "exporter": result.exporter,
            "kind": result.kind,
            "checkpoint": result.checkpoint,
            "tensor_count": result.tensor_count,
            "config": result.config,
            "tokenizer": result.tokenizer,
            "extras": dict(result.extras),
        },
        str(args.format),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
