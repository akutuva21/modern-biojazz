import pytest
from unittest.mock import patch

from modern_biojazz.benchmarking import benchmark_backend, compare_backends, BenchmarkConfig, BenchmarkResult

class MockBackend:
    def simulate(self, network, options):
        return {"trajectory": [{"t": options.t_end, "output": 1.0}]}

class MockEvaluator:
    def score(self, simulation_result=None, backend=None, network=None, t_end=20.0, dt=1.0, solver="Rodas5P", initial_conditions=None):
        return 0.95

@patch("time.perf_counter")
def test_benchmark_backend(mock_perf_counter, seed_network):
    mock_perf_counter.side_effect = [0.0, 1.5, 2.0, 3.5, 4.0, 5.5]

    backend = MockBackend()
    evaluator = MockEvaluator()
    config = BenchmarkConfig(runs=3)

    result = benchmark_backend(backend, "mock_backend", seed_network, evaluator, config)

    assert result.backend_name == "mock_backend"
    assert result.runs == 3
    assert result.mean_seconds == 1.5
    assert result.mean_score == pytest.approx(0.95)

@patch("modern_biojazz.benchmarking.benchmark_backend")
def test_compare_backends(mock_benchmark_backend, seed_network):
    mock_benchmark_backend.side_effect = [
        BenchmarkResult("candidate", 5, 2.0, 0.9),
        BenchmarkResult("baseline", 5, 4.0, 0.8)
    ]

    backend = MockBackend()
    evaluator = MockEvaluator()
    config = BenchmarkConfig(runs=5)

    result = compare_backends(backend, backend, seed_network, evaluator, config)

    assert result["candidate_mean_seconds"] == 2.0
    assert result["baseline_mean_seconds"] == 4.0
    assert result["speedup"] == 2.0
    assert result["candidate_mean_score"] == 0.9
    assert result["baseline_mean_score"] == 0.8
