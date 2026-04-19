import json
from pathlib import Path

import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from modern_biojazz.grounding_sources import INDRAClient, OmniPathClient, load_grounding_snapshot


@patch("urllib.request.urlopen")
def test_indra_client_fetch_statements_success(mock_urlopen):
    """Test successful fetching of statements from INDRA."""
    mock_response = MagicMock()
    # Read first returns valid data, second (read(1)) returns empty (to pass the size check)
    mock_response.read.side_effect = [
        json.dumps({"statements": [{"id": "stmt1"}, {"id": "stmt2"}]}).encode("utf-8"),
        b"",
    ]
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    client = INDRAClient()
    statements = client.fetch_statements(["ERK", "MEK"], "Phosphorylation")

    assert len(statements) == 2
    assert statements[0]["id"] == "stmt1"

    # Verify request payload and headers
    req = mock_urlopen.call_args[0][0]
    assert req.method == "POST"
    assert req.headers["Content-type"] == "application/json"

    payload = json.loads(req.data.decode("utf-8"))
    assert payload["subject"] == ["ERK", "MEK"]
    assert payload["object"] == ["ERK", "MEK"]
    assert payload["type"] == "Phosphorylation"
    assert payload["format"] == "json"


@patch("urllib.request.urlopen")
def test_indra_client_fetch_statements_exceeds_size(mock_urlopen):
    """Test that a ValueError is raised if the payload exceeds 10MB."""
    mock_response = MagicMock()
    # read(1) returns some data, meaning there is more data beyond 10MB
    mock_response.read.side_effect = [b"initial_data", b"1"]
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    client = INDRAClient()
    with pytest.raises(ValueError, match="INDRA response payload exceeded 10MB limit"):
        client.fetch_statements(["ERK"])


@patch("urllib.request.urlopen")
def test_indra_client_fetch_statements_not_dict(mock_urlopen):
    """Test that a TypeError is raised if the response is not a dict."""
    mock_response = MagicMock()
    # Return a list instead of a dict
    mock_response.read.side_effect = [
        json.dumps([{"id": "stmt1"}]).encode("utf-8"),
        b"",
    ]
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    client = INDRAClient()
    with pytest.raises(TypeError, match="Expected INDRA payload to be a dict, got list"):
        client.fetch_statements(["ERK"])


@patch("urllib.request.urlopen")
def test_indra_client_fetch_statements_no_statements_key(mock_urlopen):
    """Test fallback when 'statements' key is missing from response."""
    mock_response = MagicMock()
    # Return a dict with no 'statements' key
    mock_response.read.side_effect = [
        json.dumps({"other_key": "value"}).encode("utf-8"),
        b"",
    ]
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    client = INDRAClient()
    statements = client.fetch_statements(["ERK"])

    assert statements == []


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


@patch("urllib.request.urlopen")
def test_omnipath_client_fetch_success(mock_urlopen):
    """Test OmniPathClient fetches and parses correctly."""
    mock_response = MagicMock()
    mock_response.read.side_effect = [
        json.dumps([{"source": "BRAF", "target": "MEK1", "type": "phosphorylation"}]).encode("utf-8"),
        b"",
    ]
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = OmniPathClient()
    result = client.fetch_interactions(["BRAF", "MEK1"])

    assert len(result) == 1
    assert result[0]["source"] == "BRAF"
    assert result[0]["target"] == "MEK1"

    # Assert correct URL was constructed
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert "https://omnipathdb.org/interactions/" in req.full_url
    assert "genesymbols=1" in req.full_url
    assert "sources=BRAF%2CMEK1" in req.full_url
    assert "targets=BRAF%2CMEK1" in req.full_url


@patch("urllib.request.urlopen")
def test_omnipath_client_fetch_payload_too_large(mock_urlopen):
    """Test OmniPathClient raises error when payload exceeds limit."""
    mock_response = MagicMock()
    # First read returns data, second read returns 1 byte (indicating more data beyond limit)
    mock_response.read.side_effect = [b"data" * (1024 * 1024), b"1"]
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = OmniPathClient()
    with pytest.raises(ValueError, match="OmniPath response payload exceeded 10MB limit"):
        client.fetch_interactions(["ERK"])


@patch("urllib.request.urlopen")
def test_omnipath_client_fetch_invalid_payload_type(mock_urlopen):
    """Test OmniPathClient raises TypeError when response is not a list."""
    mock_response = MagicMock()
    mock_response.read.side_effect = [
        json.dumps({"error": "not a list"}).encode("utf-8"),
        b"",
    ]
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = OmniPathClient()
    with pytest.raises(TypeError, match="Expected OmniPath payload to be a list"):
        client.fetch_interactions(["ERK"])
