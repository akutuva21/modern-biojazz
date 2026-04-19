import json
from unittest.mock import patch, MagicMock, mock_open
import pytest
from modern_biojazz.indra_assembly import (
    AssemblyResult,
    INDRAGraphProposer,
    load_assembly_snapshot,
    save_assembly_snapshot,
)

@patch("urllib.request.urlopen")
def test_indra_graph_proposer_success(mock_urlopen):
    # Mock INDRA REST API Response
    mock_response = MagicMock()
    # E.g., querying for STAT3 returns a phosphorylation statement
    mock_response.read.return_value = b"""
    {
        "statements": [
            {
                "type": "Phosphorylation",
                "enz": {"name": "JAK2"},
                "sub": {"name": "STAT3"}
            }
        ]
    }
    """
    mock_urlopen.return_value.__enter__.return_value = mock_response

    proposer = INDRAGraphProposer()

    # To reliably hit Phosphorylation, we mock the random generator
    import random
    rng = random.Random(42)
    rng.choice = lambda x: x[0] if isinstance(x, list) else list(x)[0]
    proposer = INDRAGraphProposer(rng=rng)

    # We pass a model code with STAT3 as the only protein.
    actions = proposer.propose("proteins=['STAT3']", ["add_phosphorylation", "add_binding"], budget=2)

    # Since it found a 'Phosphorylation' statement, and 'add_phosphorylation' is allowed:
    assert actions[0] == "add_phosphorylation"
    # Second action is random because budget is 2
    assert len(actions) == 2

@patch("urllib.request.urlopen")
def test_indra_graph_proposer_fallback_on_error(mock_urlopen):
    # Simulate an API error (e.g. timeout)
    mock_urlopen.side_effect = Exception("API Timeout")

    proposer = INDRAGraphProposer()
    actions = proposer.propose("proteins=['STAT3']", ["add_site"], budget=1)

    # Should fallback gracefully to the random action space provided
    assert actions == ["add_site"]

def test_indra_graph_proposer_fallback_no_proteins():
    proposer = INDRAGraphProposer()
    # Missing proteins=[] in model_code
    actions = proposer.propose("empty string", ["add_site"], budget=1)

    # Should fallback gracefully
    assert actions == ["add_site"]

def test_save_assembly_snapshot_success(tmp_path):
    result = AssemblyResult(
        species=["STAT3"],
        statements=[{"type": "Phosphorylation", "enz": "JAK2", "sub": "STAT3"}],
        bngl_text="begin model\nend model",
        source="indra_live"
    )

    file_path = tmp_path / "snapshot.json"
    save_assembly_snapshot(result, file_path)

    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["species"] == ["STAT3"]
    assert data["statements"][0]["type"] == "Phosphorylation"
    assert data["bngl_text"] == "begin model\nend model"
    assert data["source"] == "indra_live"

def test_load_assembly_snapshot_success():
    """Test loading a fully populated assembly snapshot."""
    mock_data = {
        "species": ["STAT3"],
        "statements": [{"type": "Phosphorylation", "enz": {"name": "JAK2"}, "sub": {"name": "STAT3"}}],
        "bngl_text": "begin model\\nend model",
        "source": "indra_live"
    }
    mock_json = json.dumps(mock_data)

    with patch("builtins.open", mock_open(read_data=mock_json)):
        result = load_assembly_snapshot("dummy_path.json")

    assert result.species == ["STAT3"]
    assert len(result.statements) == 1
    assert result.statements[0]["type"] == "Phosphorylation"
    assert result.bngl_text == "begin model\\nend model"
    assert result.source == "indra_live"

def test_load_assembly_snapshot_missing_optional_fields():
    """Test loading an assembly snapshot where optional fields like statements and source are missing."""
    mock_data = {
        "species": ["STAT3"],
        "bngl_text": "begin model\\nend model"
    }
    mock_json = json.dumps(mock_data)

    with patch("builtins.open", mock_open(read_data=mock_json)):
        result = load_assembly_snapshot("dummy_path.json")

    assert result.species == ["STAT3"]
    assert result.statements == [] # default value
    assert result.bngl_text == "begin model\\nend model"
    assert result.source == "file" # default value

def test_load_assembly_snapshot_invalid_json():
    """Test loading an assembly snapshot with invalid JSON content."""
    invalid_json = "{ invalid json data"

    with patch("builtins.open", mock_open(read_data=invalid_json)):
        with pytest.raises(json.JSONDecodeError):
            load_assembly_snapshot("dummy_path.json")

def test_load_assembly_snapshot_file_not_found():
    """Test loading an assembly snapshot where the file does not exist."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_assembly_snapshot("non_existent_path.json")
