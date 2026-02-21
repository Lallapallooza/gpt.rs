from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from .types import ExporterSpec, ExportRequest, ExportResult

_COMMON_ARG_KEYS = {
    "command",
    "exporter",
    "format",
    "checkpoint_out",
    "config_out",
    "tokenizer_out",
}


def _select_path(explicit: Optional[Path], fallback: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit
    return fallback


def build_request(exporter: ExporterSpec, args: argparse.Namespace) -> ExportRequest:
    checkpoint = _select_path(
        getattr(args, "checkpoint_out", None),
        exporter.info.artifacts.checkpoint,
    )
    if checkpoint is None:
        raise ValueError(f"exporter {exporter.info.name!r} has no checkpoint output path")
    config_out = _select_path(getattr(args, "config_out", None), exporter.info.artifacts.config)
    tokenizer_out = _select_path(
        getattr(args, "tokenizer_out", None), exporter.info.artifacts.tokenizer
    )

    options: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in _COMMON_ARG_KEYS or value is None:
            continue
        options[key] = value

    return ExportRequest(
        checkpoint_out=checkpoint,
        config_out=config_out,
        tokenizer_out=tokenizer_out,
        options=options,
    )


def _ensure_output_dirs(request: ExportRequest) -> None:
    request.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    if request.config_out is not None:
        request.config_out.parent.mkdir(parents=True, exist_ok=True)
    if request.tokenizer_out is not None:
        request.tokenizer_out.parent.mkdir(parents=True, exist_ok=True)


def run_export(exporter: ExporterSpec, request: ExportRequest) -> ExportResult:
    _ensure_output_dirs(request)
    return exporter.export(request)


def run_validation(exporter: ExporterSpec, request: ExportRequest) -> list[str]:
    return exporter.validate(request)
