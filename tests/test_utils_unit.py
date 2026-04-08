from __future__ import annotations
import json
from pathlib import Path
import pytest
from modern_biojazz.utils import save_json_snapshot

def test_save_json_snapshot_happy_path(tmp_path):
    """Test that save_json_snapshot correctly saves a dictionary to a JSON file."""
    data = {"key": "value", "number": 42}
    file_path = tmp_path / "snapshot.json"
    save_json_snapshot(data, str(file_path))
    assert file_path.exists()
    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == data

def test_save_json_snapshot_with_path_object(tmp_path):
    """Test that save_json_snapshot accepts a Path object."""
    data = {"list": [1, 2, 3]}
    file_path = tmp_path / "subdir" / "snapshot.json"
    file_path.parent.mkdir()
    save_json_snapshot(data, file_path)
    assert file_path.exists()
    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == data

def test_save_json_snapshot_invalid_path():
    """Test that save_json_snapshot raises FileNotFoundError when the directory doesn't exist."""
    data = {"key": "value"}
    invalid_path = "/nonexistent_directory_12345/snapshot.json"
    with pytest.raises(FileNotFoundError):
        save_json_snapshot(data, invalid_path)
