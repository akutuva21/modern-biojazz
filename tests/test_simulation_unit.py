from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import pytest

from modern_biojazz.simulation import (
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


def test_catalyst_client_rejects_insecure_urls():
    client = CatalystHTTPClient(base_url="http://example.com")
    with pytest.raises(ValueError, match="Insecure scheme: http"):
        client._validate_url("http://example.com")


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
