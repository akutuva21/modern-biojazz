from __future__ import annotations

import builtins
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from modern_biojazz.simulation import (
    CatalystHTTPClient,
    FitnessEvaluator,
    LocalCatalystEngine,
    UltrasensitiveFitnessEvaluator,
    SimulationOptions,
    DoseResponseConfig,
    CatalystHTTPClient,
)


def test_fitness_evaluator_accepts_backend_network(seed_network):
    engine = LocalCatalystEngine()
    evaluator = FitnessEvaluator(target_output=1.0)
    score = evaluator.score(backend=engine, network=seed_network, t_end=5.0, dt=1.0)
    assert score >= 0.0


def test_ultrasensitive_evaluator_matches_unified_interface(seed_network):
    engine = LocalCatalystEngine()
    config = DoseResponseConfig(input_species="STAT3", output_species="SOCS3")
    evaluator = UltrasensitiveFitnessEvaluator(config=config)
    score = evaluator.score(backend=engine, network=seed_network, t_end=5.0, dt=1.0)
    assert score >= 0.0


def test_local_engine_euler_fallback_when_scipy_unavailable(seed_network, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("scipy"):
            raise ImportError("forced missing scipy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    engine = LocalCatalystEngine()
    result = engine.simulate(seed_network, SimulationOptions(t_end=3.0, dt=1.0, solver="Rodas5P"))

    assert result["solver"] == "EulerFallback"
    assert len(result["trajectory"]) == 4


def test_local_engine_handles_simulation_error_euler(seed_network, monkeypatch):
    # Force Euler path by making scipy unavailable
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("scipy"):
            raise ImportError("forced missing scipy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Cause an error by making rule rates invalid for the solver loop
    # but not for the setup. Actually, let's mock the 'rhs' function inside simulate.
    # But rhs is defined inside simulate.
    # Let's mock 'zip' to raise an error instead of max, it's safer.
    real_zip = builtins.zip

    def fake_zip(*args):
        # The Euler loop uses zip(current, deriv)
        if len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
            raise RuntimeError("Euler loop failure")
        return real_zip(*args)

    monkeypatch.setattr(builtins, "zip", fake_zip)

    engine = LocalCatalystEngine()
    result = engine.simulate(seed_network, SimulationOptions(t_end=1.0, dt=1.0))

    assert result["trajectory"] == []
    assert "error" in result["stats"]
    assert "Euler loop failure" in result["stats"]["error"]
    assert result["solver"] == "EulerFallback"


def test_local_engine_handles_simulation_error_scipy(seed_network, monkeypatch):
    # Mock solve_ivp to exist but fail
    mock_solve_ivp = MagicMock(side_effect=RuntimeError("SciPy solver failure"))

    # We need to make sure 'from scipy.integrate import solve_ivp' returns our mock
    # A cleaner way to mock the import:
    import sys

    mock_scipy = MagicMock()
    mock_scipy.integrate.solve_ivp = mock_solve_ivp
    monkeypatch.setitem(sys.modules, "scipy", mock_scipy)
    monkeypatch.setitem(sys.modules, "scipy.integrate", mock_scipy.integrate)

    engine = LocalCatalystEngine()
    result = engine.simulate(seed_network, SimulationOptions(t_end=1.0, dt=1.0))

    assert result["trajectory"] == []
    assert "error" in result["stats"]
    assert "SciPy solver failure" in result["stats"]["error"]
    assert result["solver"] == "BDF"


def test_catalyst_http_client_simulate_success(seed_network):
    client = CatalystHTTPClient(base_url="https://example.com", _allow_insecure_for_testing=True)
    options = SimulationOptions(t_end=5.0, dt=1.0, solver="Rodas5P")

    expected_response = {"solver": "Rodas5P", "trajectory": []}

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(expected_response).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        # urlopen is used as a context manager in the code
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result = client.simulate(seed_network, options)

        assert result == expected_response
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://example.com/simulate"
        assert req.method == "POST"

        # Verify payload contains expected keys
        payload = json.loads(req.data.decode("utf-8"))
        assert "network" in payload
        assert "t_end" in payload
        assert "dt" in payload
        assert "solver" in payload
        assert "initial_conditions" in payload


def test_catalyst_http_client_simulate_http_error(seed_network):
    client = CatalystHTTPClient(base_url="https://example.com", retry_count=1, _allow_insecure_for_testing=True)
    options = SimulationOptions(t_end=5.0)

    mock_response = MagicMock()
    mock_response.status = 500

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen, patch("time.sleep"):
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with pytest.raises(RuntimeError, match="Catalyst service returned HTTP 500"):
            client.simulate(seed_network, options)

        # Should be called retry_count + 1 times
        assert mock_urlopen.call_count == 2


def test_catalyst_http_client_simulate_retries_and_fails(seed_network):
    client = CatalystHTTPClient(base_url="https://example.com", retry_count=2, _allow_insecure_for_testing=True)
    options = SimulationOptions(t_end=5.0)

    with (
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")) as mock_urlopen,
        patch("time.sleep") as mock_sleep,
    ):

        with pytest.raises(RuntimeError, match="Failed to simulate network via Catalyst service"):
            client.simulate(seed_network, options)

        # Initial attempt + 2 retries = 3 calls
        assert mock_urlopen.call_count == 3
        # Should sleep 2 times between the 3 attempts
        assert mock_sleep.call_count == 2


def test_catalyst_http_client_validate_url_insecure():
    client = CatalystHTTPClient(base_url="http://example.com")
    with pytest.raises(ValueError, match="Insecure scheme: http"):
        client._validate_url(client.base_url)


def test_catalyst_http_client_validate_url_missing_hostname():
    client = CatalystHTTPClient(base_url="https://")
    with pytest.raises(ValueError, match="Invalid URL: missing hostname"):
        client._validate_url(client.base_url)


def test_catalyst_http_client_validate_url_resolve_error():
    client = CatalystHTTPClient(base_url="https://nonexistent.domain.test")
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
        with pytest.raises(ValueError, match="Could not resolve hostname"):
            client._validate_url(client.base_url)


def test_catalyst_http_client_validate_url_private_ip():
    client = CatalystHTTPClient(base_url="https://internal-service.local")

    # Mock socket.getaddrinfo to return a private IP address (127.0.0.1)
    # The format is a list of tuples: (family, type, proto, canonname, sockaddr)
    # sockaddr for IPv4 is (address, port)
    mock_addr_info = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    with patch("socket.getaddrinfo", return_value=mock_addr_info):
        with pytest.raises(ValueError, match="URL resolves to internal/reserved IP address: 127.0.0.1"):
            client._validate_url(client.base_url)

def test_catalyst_client_simulate_validates_url(seed_network):
    # Testing that simulate calls url validation when _allow_insecure_for_testing is False
    client = CatalystHTTPClient("http://example.com", _allow_insecure_for_testing=False)
    with pytest.raises(ValueError, match="Insecure scheme"):
        client.simulate(seed_network, SimulationOptions(t_end=1.0, dt=1.0))
