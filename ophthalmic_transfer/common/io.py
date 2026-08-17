from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
