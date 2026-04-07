
import time
from dataclasses import dataclass
from typing import Dict, List

from .site_graph import ReactionNetwork
from .simulation import SimulationBackend, FitnessScorer, SimulationOptions


@dataclass
class BenchmarkConfig:
    runs: int = 5
    t_end: float = 20.0
    dt: float = 1.0
    solver: str = "Rodas5P"


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
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    if config is None:
        config = BenchmarkConfig()

    durations: List[float] = []
    scores: List[float] = []
    for _ in range(config.runs):
        start = time.perf_counter()
        sim = backend.simulate(
            network,
            SimulationOptions(
                t_end=config.t_end,
                dt=config.dt,
                solver=config.solver
            ),
        )
        durations.append(time.perf_counter() - start)
        scores.append(
            evaluator.score(
                simulation_result=sim,
                backend=backend,
                network=network,
                t_end=config.t_end,
                dt=config.dt,
                solver=config.solver,
            )
        )

    return BenchmarkResult(
        backend_name=backend_name,
        runs=config.runs,
        mean_seconds=sum(durations) / max(1, len(durations)),
        mean_score=sum(scores) / max(1, len(scores)),
    )


def compare_backends(
    candidate: SimulationBackend,
    baseline: SimulationBackend,
    network: ReactionNetwork,
    evaluator: FitnessScorer,
    config: BenchmarkConfig | None = None,
) -> Dict[str, float]:
    if config is None:
        config = BenchmarkConfig()
    c = benchmark_backend(candidate, "candidate", network, evaluator, config=config)
    b = benchmark_backend(baseline, "baseline", network, evaluator, config=config)
    speedup = b.mean_seconds / c.mean_seconds if c.mean_seconds > 0 else 0.0
    return {
        "candidate_mean_seconds": c.mean_seconds,
        "baseline_mean_seconds": b.mean_seconds,
        "speedup": speedup,
        "candidate_mean_score": c.mean_score,
        "baseline_mean_score": b.mean_score,
    }
