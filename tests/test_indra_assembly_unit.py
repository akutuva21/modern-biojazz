import pytest
from unittest.mock import patch, MagicMock
from modern_biojazz.indra_assembly import INDRAGraphProposer, load_bngl_file

def test_load_bngl_file_success(fixtures_dir):
    path = fixtures_dir / "sample_indra.bngl"
    result = load_bngl_file(path)

    assert result.source == "file"
    assert result.species == []
    assert result.statements == []
    assert "begin model" in result.bngl_text

def test_load_bngl_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_bngl_file("non_existent_file.bngl")

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
