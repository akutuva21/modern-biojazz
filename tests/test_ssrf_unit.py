import pytest
from modern_biojazz.simulation import CatalystHTTPClient

def test_catalyst_client_ssrf_validation():
    # Insecure scheme
    client = CatalystHTTPClient("http://example.com")
    with pytest.raises(ValueError, match="Insecure scheme"):
        client._validate_url(client.base_url)

    # Localhost / loopback
    client = CatalystHTTPClient("https://127.0.0.1")
    with pytest.raises(ValueError, match="internal/reserved IP address"):
        client._validate_url(client.base_url)

    client = CatalystHTTPClient("https://localhost")
    with pytest.raises(ValueError, match="internal/reserved IP address"):
        client._validate_url(client.base_url)

    # Private IP
    client = CatalystHTTPClient("https://10.0.0.1")
    with pytest.raises(ValueError, match="internal/reserved IP address"):
        client._validate_url(client.base_url)
