from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def resolve_model_dir(model_id: str) -> Path:
    """Returns a local directory holding the Hugging Face config, tokenizer and safetensors."""

    local = Path(model_id)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
        )
    )


class SafetensorsSource:
    """Lazy tensor access over a safetensors checkpoint, sharded or not."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            self.weight_map: Dict[str, str] = {str(k): str(v) for k, v in weight_map.items()}
        else:
            single = model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors checkpoint found in {model_dir}")
            from safetensors import safe_open

            with safe_open(str(single), framework="pt") as handle:
                self.weight_map = {str(k): single.name for k in handle.keys()}
        self._handles: Dict[str, Any] = {}

    def _handle(self, filename: str) -> Any:
        handle = self._handles.get(filename)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(self.model_dir / filename), framework="pt")
            self._handles[filename] = handle
        return handle

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def shape(self, name: str) -> tuple[int, ...]:
        if name not in self.weight_map:
            raise KeyError(f"missing tensor in checkpoint: {name}")
        sl = self._handle(self.weight_map[name]).get_slice(name)
        return tuple(int(d) for d in sl.get_shape())

    def get(self, name: str) -> Any:
        if name not in self.weight_map:
            raise KeyError(f"missing tensor in checkpoint: {name}")
        return self._handle(self.weight_map[name]).get_tensor(name)
