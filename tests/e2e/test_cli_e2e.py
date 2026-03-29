from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_e2e(fixtures_dir: Path):
    seed = fixtures_dir / "seed_network.json"
    grounding = fixtures_dir / "grounding_payload.json"

    cmd = [
        sys.executable,
        "-m",
        "modern_biojazz.cli",
        "--seed",
        str(seed),
        "--grounding",
        str(grounding),
        "--generations",
        "2",
        "--population",
        "6",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)

    assert "best_score" in payload
    assert "best_network" in payload
    assert payload["grounding"] is not None
