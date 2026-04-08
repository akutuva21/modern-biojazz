from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from modern_biojazz.benchmarking import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_backend,
    compare_backends,
)

def test_benchmark_backend_default_config(seed_network):
    mock_backend = MagicMock()
    mock_backend.simulate.return_value = {"trajectory": [{"output": 1.0}]}

    mock_evaluator = MagicMock()
    mock_evaluator.score.return_value = 0.8

    result = benchmark_backend(
        backend=mock_backend,
        backend_name="test_backend",
        network=seed_network,
        evaluator=mock_evaluator,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.backend_name == "test_backend"
    assert result.runs == 5
    assert result.mean_score == 0.8
    assert result.mean_seconds >= 0.0

    assert mock_backend.simulate.call_count == 5
    assert mock_evaluator.score.call_count == 5

def test_benchmark_backend_custom_config(seed_network):
    mock_backend = MagicMock()
    mock_backend.simulate.return_value = {"trajectory": [{"output": 1.0}]}

    mock_evaluator = MagicMock()
    mock_evaluator.score.return_value = 0.9

    config = BenchmarkConfig(runs=2, t_end=10.0, dt=0.5, solver="Euler")

    result = benchmark_backend(
        backend=mock_backend,
        backend_name="custom_backend",
        network=seed_network,
        evaluator=mock_evaluator,
        config=config,
    )

    assert result.backend_name == "custom_backend"
    assert result.runs == 2
    assert result.mean_score == 0.9

    assert mock_backend.simulate.call_count == 2
    assert mock_evaluator.score.call_count == 2

    call_args = mock_backend.simulate.call_args[0]
    passed_options = call_args[1]
    assert passed_options.t_end == 10.0
    assert passed_options.dt == 0.5
    assert passed_options.solver == "Euler"

def test_compare_backends_speedup(seed_network, monkeypatch):
    mock_benchmark_backend = MagicMock()

    c_res = BenchmarkResult(backend_name="candidate", runs=1, mean_seconds=1.0, mean_score=0.9)
    b_res = BenchmarkResult(backend_name="baseline", runs=1, mean_seconds=2.0, mean_score=0.8)

    mock_benchmark_backend.side_effect = [c_res, b_res]

    monkeypatch.setattr("modern_biojazz.benchmarking.benchmark_backend", mock_benchmark_backend)

    candidate = MagicMock()
    baseline = MagicMock()
    evaluator = MagicMock()

    result = compare_backends(candidate, baseline, seed_network, evaluator)

    assert result["candidate_mean_seconds"] == 1.0
    assert result["baseline_mean_seconds"] == 2.0
    assert result["speedup"] == 2.0
    assert result["candidate_mean_score"] == 0.9
    assert result["baseline_mean_score"] == 0.8

    assert mock_benchmark_backend.call_count == 2

def test_compare_backends_zero_candidate_time(seed_network, monkeypatch):
    mock_benchmark_backend = MagicMock()

    c_res = BenchmarkResult(backend_name="candidate", runs=1, mean_seconds=0.0, mean_score=0.9)
    b_res = BenchmarkResult(backend_name="baseline", runs=1, mean_seconds=2.0, mean_score=0.8)

    mock_benchmark_backend.side_effect = [c_res, b_res]

    monkeypatch.setattr("modern_biojazz.benchmarking.benchmark_backend", mock_benchmark_backend)

    candidate = MagicMock()
    baseline = MagicMock()
    evaluator = MagicMock()

    result = compare_backends(candidate, baseline, seed_network, evaluator)

    assert result["speedup"] == 0.0
