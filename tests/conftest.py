from __future__ import annotations

import json
from pathlib import Path

import pytest

from modern_biojazz.site_graph import ReactionNetwork


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def seed_network(fixtures_dir: Path) -> ReactionNetwork:
    with open(fixtures_dir / "seed_network.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ReactionNetwork.from_dict(payload)


@pytest.fixture
def grounding_payload(fixtures_dir: Path) -> dict:
    with open(fixtures_dir / "grounding_payload.json", "r", encoding="utf-8") as f:
        return json.load(f)
