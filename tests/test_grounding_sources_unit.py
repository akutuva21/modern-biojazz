import json
from pathlib import Path

import pytest

from modern_biojazz.grounding_sources import load_grounding_snapshot


def test_load_grounding_snapshot_success(tmp_path: Path):
    """Test that a valid JSON file is loaded correctly."""
    snapshot_data = {
        "abstract_types": {"Kinase": "Kinase", "Phosphatase": "Phosphatase"},
        "real_nodes": [{"name": "ERK", "type": "Kinase"}],
        "real_interactions": [["RAF", "MEK", "phosphorylation"]],
        "confidence_by_pair": {"Kinase->ERK": 0.95},
    }
    file_path = tmp_path / "snapshot.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f)

    loaded_data = load_grounding_snapshot(file_path)
    assert loaded_data == snapshot_data


def test_load_grounding_snapshot_file_not_found():
    """Test that FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_grounding_snapshot("non_existent_file.json")


def test_load_grounding_snapshot_invalid_json(tmp_path: Path):
    """Test that JSONDecodeError is raised when file contains invalid JSON."""
    file_path = tmp_path / "invalid.json"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("{invalid_json:")

    with pytest.raises(json.JSONDecodeError):
        load_grounding_snapshot(file_path)
