"""Generic utilities for Modern BioJazz."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json_snapshot(data: Dict[str, Any], path: str | Path) -> None:
    """Save a snapshot dictionary to JSON for offline/reproducible use."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
