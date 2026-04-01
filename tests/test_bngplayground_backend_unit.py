import pytest
from unittest.mock import MagicMock, patch
from modern_biojazz.bngplayground_backend import BNGPlaygroundBackend, BNGLParsingError
from modern_biojazz.site_graph import ReactionNetwork, Protein, Site, Rule

def test_parse_bngl_success():
    """Test parse_bngl by mocking the MCP tool caller on success."""
    backend = BNGPlaygroundBackend(bngplayground_path="/dummy/path")
    expected_response = {"parsed_model": "some_data", "success": True}

    with patch.object(backend, '_call_mcp_tool', return_value=expected_response) as mock_call_mcp:
        sample_bngl = "begin model\nend model"
        result = backend.parse_bngl(sample_bngl)
        assert result == expected_response
        mock_call_mcp.assert_called_once_with("parse_bngl", {"bngl": sample_bngl})

def test_parse_bngl_error():
    """Test parse_bngl by mocking the MCP tool caller on error."""
    backend = BNGPlaygroundBackend(bngplayground_path="/dummy/path")
    error_message = "Syntax error at line 1"
    error_response = {"error": error_message}

    with patch.object(backend, '_call_mcp_tool', return_value=error_response) as mock_call_mcp:
        sample_bngl = "invalid bngl"
        with pytest.raises(BNGLParsingError) as exc_info:
            backend.parse_bngl(sample_bngl)
        assert str(exc_info.value) == error_message
        mock_call_mcp.assert_called_once_with("parse_bngl", {"bngl": sample_bngl})

def test_network_to_bngl():
    """Test the generation of BNGL text from a ReactionNetwork."""
    proteins = {
        "A": Protein(name="A", sites=[Site(name="s1", site_type="binding")]),
        "B": Protein(name="B", sites=[Site(name="s1", site_type="binding")])
    }
    rules = [
        Rule(name="r1", rule_type="binding", reactants=["A", "B"], products=["A_B"], rate=0.1)
    ]

    network = ReactionNetwork(proteins=proteins, rules=rules)
    backend = BNGPlaygroundBackend(bngplayground_path="/dummy/path")

    bngl = backend._network_to_bngl(network, t_end=10.0, dt=1.0)

    assert "begin model" in bngl
    assert "r1_rate 0.1" in bngl
    assert "A(s1)" in bngl
    assert "B(s1)" in bngl
    # Note: PR 33 updated rule formatting to "r1: A() + B() -> A_B() r1_rate"
    assert "r1: A() + B() -> A_B() r1_rate" in bngl
