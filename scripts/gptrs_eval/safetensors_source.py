from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_model_dir(model_id: str) -> Path:
    """Returns a local directory holding the Hugging Face config, tokenizer and safetensors."""

    local = Path(model_id)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "model*.safetensors", "*.jinja", "*.txt"],
        )
    )


class SafetensorsSource:
    """Lazy tensor access over a safetensors checkpoint, sharded or not.

    A weight that a compressed-tensors checkpoint stores packed (`weight_packed` with its scale, its
    shape, and an optional zero point) reads under its module's `weight` name, decompressed by
    compressed-tensors.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            self.weight_map: dict[str, str] = {str(k): str(v) for k, v in weight_map.items()}
        else:
            single = model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors checkpoint found in {model_dir}")
            from safetensors import safe_open

            with safe_open(str(single), framework="pt") as handle:
                self.weight_map = {str(k): single.name for k in handle.keys()}
        self._handles: dict[str, Any] = {}

    def _handle(self, filename: str) -> Any:
        handle = self._handles.get(filename)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(self.model_dir / filename), framework="pt")
            self._handles[filename] = handle
        return handle

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map or self._packed(name) is not None

    def shape(self, name: str) -> tuple[int, ...]:
        if self._packed(name) is not None:
            return tuple(int(d) for d in self.get(name).shape)
        if name not in self.weight_map:
            raise KeyError(f"missing tensor in checkpoint: {name}")
        sl = self._handle(self.weight_map[name]).get_slice(name)
        return tuple(int(d) for d in sl.get_shape())

    def get(self, name: str) -> Any:
        module = self._packed(name)
        if module is not None:
            return self._decompressed(module)
        if name not in self.weight_map:
            raise KeyError(f"missing tensor in checkpoint: {name}")
        return self._handle(self.weight_map[name]).get_tensor(name)

    def _packed(self, name: str) -> str | None:
        """The module of weight `name`, when the checkpoint stores that weight packed, or None."""
        if name in self.weight_map or not name.endswith(".weight"):
            return None
        module = name.removesuffix(".weight")
        return module if f"{module}.weight_packed" in self.weight_map else None

    def _decompressed(self, module: str) -> Any:
        """The weight of `module`, decompressed by compressed-tensors in the dtype of its scale."""
        from compressed_tensors.compressors import PackedQuantizationCompressor
        from compressed_tensors.quantization import QuantizationScheme

        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        groups = config["quantization_config"]["config_groups"]
        if len(groups) != 1:
            raise ValueError(f"expected one quantization group, found {sorted(groups)}")
        scheme = QuantizationScheme(**next(iter(groups.values())))
        state = {}
        for leaf in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
            key = f"{module}.{leaf}"
            if key in self.weight_map:
                state[leaf] = self._handle(self.weight_map[key]).get_tensor(key)
        return PackedQuantizationCompressor.decompress(state, scheme)["weight"]
