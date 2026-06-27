"""Configuration loader — reads YAML config and provides typed access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)
