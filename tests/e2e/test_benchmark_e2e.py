from __future__ import annotations

import time

from modern_biojazz.benchmarking import compare_backends
from modern_biojazz.simulation import LocalCatalystEngine, FitnessEvaluator


class SlowLegacyLikeBackend:
    def simulate(self, network, t_end, dt, solver):
        time.sleep(0.5)
        return LocalCatalystEngine().simulate(network, t_end, dt, solver)


def test_benchmark_compare_backends_e2e(seed_network):
    candidate = LocalCatalystEngine()
    baseline = SlowLegacyLikeBackend()
    metrics = compare_backends(
        candidate=candidate,
        baseline=baseline,
        network=seed_network,
        evaluator=FitnessEvaluator(target_output=1.0),
        runs=3,
    )

    assert metrics["speedup"] > 1.0
    assert metrics["candidate_mean_score"] >= 0.0
    assert metrics["baseline_mean_score"] >= 0.0
