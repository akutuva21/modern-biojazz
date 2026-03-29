from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from .site_graph import ReactionNetwork
from .simulation import SimulationBackend, FitnessScorer


@dataclass
class BenchmarkResult:
    backend_name: str
    runs: int
    mean_seconds: float
    mean_score: float


def benchmark_backend(
    backend: SimulationBackend,
    backend_name: str,
    network: ReactionNetwork,
    evaluator: FitnessScorer,
    runs: int = 5,
    t_end: float = 20.0,
    dt: float = 1.0,
    solver: str = "Rodas5P",
) -> BenchmarkResult:
    durations: List[float] = []
    scores: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        sim = backend.simulate(network, t_end=t_end, dt=dt, solver=solver)
        durations.append(time.perf_counter() - start)
        scores.append(
            evaluator.score(
                simulation_result=sim,
                backend=backend,
                network=network,
                t_end=t_end,
                dt=dt,
                solver=solver,
            )
        )

    return BenchmarkResult(
        backend_name=backend_name,
        runs=runs,
        mean_seconds=sum(durations) / max(1, len(durations)),
        mean_score=sum(scores) / max(1, len(scores)),
    )


def compare_backends(
    candidate: SimulationBackend,
    baseline: SimulationBackend,
    network: ReactionNetwork,
    evaluator: FitnessScorer,
    runs: int = 5,
) -> Dict[str, float]:
    c = benchmark_backend(candidate, "candidate", network, evaluator, runs=runs)
    b = benchmark_backend(baseline, "baseline", network, evaluator, runs=runs)
    speedup = b.mean_seconds / c.mean_seconds if c.mean_seconds > 0 else 0.0
    return {
        "candidate_mean_seconds": c.mean_seconds,
        "baseline_mean_seconds": b.mean_seconds,
        "speedup": speedup,
        "candidate_mean_score": c.mean_score,
        "baseline_mean_score": b.mean_score,
    }
