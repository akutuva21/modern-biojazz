import pytest
from unittest.mock import MagicMock, patch
from modern_biojazz.bngplayground_backend import BNGPlaygroundBackend, BNGLParsingError

def test_parse_bngl_success():
    """Test parse_bngl by mocking the MCP tool caller on success."""

    # Initialize backend without needing a real BNGPLAYGROUND_PATH
    backend = BNGPlaygroundBackend(bngplayground_path="/dummy/path")

    # Create a mock response
    expected_response = {"parsed_model": "some_data", "success": True}

    # Mock the internal _call_mcp_tool method
    with patch.object(backend, '_call_mcp_tool', return_value=expected_response) as mock_call_mcp:
        sample_bngl = "begin model\nend model"

        # Call the method under test
        result = backend.parse_bngl(sample_bngl)

        # Assert the result matches the mocked response
        assert result == expected_response

        # Assert the internal method was called with expected arguments
        mock_call_mcp.assert_called_once_with("parse_bngl", {"bngl": sample_bngl})

def test_parse_bngl_error():
    """Test parse_bngl by mocking the MCP tool caller on error."""

    backend = BNGPlaygroundBackend(bngplayground_path="/dummy/path")

    # Create an error response
    error_message = "Syntax error at line 1"
    error_response = {"error": error_message}

    # Mock the internal _call_mcp_tool method
    with patch.object(backend, '_call_mcp_tool', return_value=error_response) as mock_call_mcp:
        sample_bngl = "invalid bngl"

        # Call the method and assert BNGLParsingError is raised
        with pytest.raises(BNGLParsingError) as exc_info:
            backend.parse_bngl(sample_bngl)

        assert str(exc_info.value) == error_message
        mock_call_mcp.assert_called_once_with("parse_bngl", {"bngl": sample_bngl})
