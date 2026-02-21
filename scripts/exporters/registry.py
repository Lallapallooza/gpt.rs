from __future__ import annotations

from typing import Dict, Iterable, List

from .types import ExporterInfo, ExporterSpec

_EXPORTERS: Dict[str, ExporterSpec] = {}
_LOADED = False


def register_exporter(exporter: ExporterSpec) -> None:
    name = exporter.info.name
    existing = _EXPORTERS.get(name)
    if existing is not None:
        raise ValueError(f"exporter {name!r} already registered")
    _EXPORTERS[name] = exporter


def _load_default_exporters() -> None:
    global _LOADED
    if _LOADED:
        return
    from .specs.gpt2 import GPT2Exporter
    from .specs.ministral import MinistralExporter
    from .specs.vision import build_vision_exporters

    register_exporter(GPT2Exporter())
    register_exporter(MinistralExporter())
    for exporter in build_vision_exporters():
        register_exporter(exporter)
    _LOADED = True


def list_exporter_names() -> List[str]:
    _load_default_exporters()
    return sorted(_EXPORTERS.keys())


def get_exporter(name: str) -> ExporterSpec:
    _load_default_exporters()
    if name not in _EXPORTERS:
        known = ", ".join(sorted(_EXPORTERS.keys()))
        raise KeyError(f"unknown exporter {name!r}. known exporters: {known}")
    return _EXPORTERS[name]


def iter_exporters() -> Iterable[ExporterSpec]:
    _load_default_exporters()
    for name in sorted(_EXPORTERS.keys()):
        yield _EXPORTERS[name]


def iter_exporter_infos() -> Iterable[ExporterInfo]:
    for exporter in iter_exporters():
        yield exporter.info
