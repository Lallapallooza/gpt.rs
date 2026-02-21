from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class ArtifactDefaults:
    checkpoint: Path
    config: Optional[Path] = None
    tokenizer: Optional[Path] = None


@dataclass(frozen=True)
class EvalCaseRegistration:
    model_name: str
    module: str
    cls: str
    default_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExporterInfo:
    name: str
    kind: str
    description: str
    artifacts: ArtifactDefaults
    eval_case: Optional[EvalCaseRegistration] = None


@dataclass(frozen=True)
class ExportRequest:
    checkpoint_out: Path
    config_out: Optional[Path]
    tokenizer_out: Optional[Path]
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportResult:
    exporter: str
    kind: str
    checkpoint: Path
    tensor_count: int
    config: Optional[Path] = None
    tokenizer: Optional[Path] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


class ExporterSpec(Protocol):
    info: ExporterInfo

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register exporter-specific CLI arguments."""

    def export(self, request: ExportRequest) -> ExportResult:
        """Export model artifacts."""

    def validate(self, request: ExportRequest) -> list[str]:
        """Return validation errors. Empty list means valid."""
