"""Differential evolution optimizer for reaction network rate constants.

Port of BNG Playground's differentialEvolution.ts (Storn & Price 1997).
Operates on log10-transformed rate constants to handle the wide dynamic
range typical of biochemical systems (1e-6 to 1e3).

Usage:
    from modern_biojazz.rate_optimizer import optimize_rates, DEConfig

    result = optimize_rates(
        network=my_network,
        objective=lambda net: evaluator.score(backend, net, t_end=20.0),
        config=DEConfig(max_eval=500),
    )
    optimized_network = result.best_network
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .site_graph import ReactionNetwork


@dataclass
class DEConfig:
    """Differential Evolution configuration."""
    pop_size: Optional[int] = None  # Default: 10 * n_params
    F: float = 0.8              # Differential weight [0, 2]
    CR: float = 0.9             # Crossover probability [0, 1]
    max_eval: int = 2000        # Maximum objective evaluations
    ftol: float = 1e-6          # Convergence tolerance on best value change
    patience: int = 20          # Stagnant generations before stopping
    log10_lower: float = -6.0   # log10 lower bound for rates
    log10_upper: float = 2.0    # log10 upper bound for rates
    seed: int = 42


@dataclass
class DEResult:
    """Differential evolution result."""
    best_network: ReactionNetwork
    best_rates: List[float]
    best_score: float
    n_eval: int
    generations: int
    converged: bool
    stop_reason: str
    history: List[float] = field(default_factory=list)


def optimize_rates(
    network: ReactionNetwork,
    objective: Callable[[ReactionNetwork], float],
    config: Optional[DEConfig] = None,
) -> DEResult:
    """Optimize rate constants of a ReactionNetwork using Differential Evolution.

    Parameters
    ----------
    network : ReactionNetwork
        The network whose rates will be optimized. Not mutated.
    objective : callable
        Takes a ReactionNetwork, returns a score to MAXIMIZE.
        Typically wraps a FitnessEvaluator.
    config : DEConfig, optional
        DE hyperparameters.
    """
    cfg = config or DEConfig()
    rng = random.Random(cfg.seed)

    # Extract mutable rate indices.
    rate_indices = list(range(len(network.rules)))
    n_dim = len(rate_indices)
    if n_dim == 0:
        return DEResult(
            best_network=network.copy(),
            best_rates=[],
            best_score=0.0,
            n_eval=0,
            generations=0,
            converged=False,
            stop_reason="no_rates",
        )

    pop_size = cfg.pop_size or max(5, 10 * n_dim)
    lb = cfg.log10_lower
    ub = cfg.log10_upper

    # Initialize population in log10 space.
    population: List[List[float]] = []
    for _ in range(pop_size):
        individual = [rng.uniform(lb, ub) for _ in range(n_dim)]
        population.append(individual)

    # Seed the first individual with the current rates (in log10).
    seed_rates = []
    for idx in rate_indices:
        rate = max(1e-12, network.rules[idx].rate)
        seed_rates.append(math.log10(rate))
    population[0] = [max(lb, min(ub, x)) for x in seed_rates]

    def evaluate(log_rates: List[float]) -> float:
        """Build network with given rates and score it."""
        candidate = network.copy()
        for i, idx in enumerate(rate_indices):
            candidate.rules[idx].rate = 10.0 ** max(lb, min(ub, log_rates[i]))
        try:
            return objective(candidate)
        except Exception:
            return 0.0

    # Evaluate initial population.
    scores = [evaluate(ind) for ind in population]
    n_eval = pop_size

    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_x = list(population[best_idx])
    history = [best_score]

    prev_best = best_score
    stagnant = 0

    generation = 0
    while n_eval < cfg.max_eval:
        generation += 1

        for i in range(pop_size):
            if n_eval >= cfg.max_eval:
                break

            # DE/rand/1/bin: pick 3 distinct individuals ≠ i.
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = rng.sample(candidates, 3)

            # Mutation.
            mutant = [
                population[a][d] + cfg.F * (population[b][d] - population[c][d])
                for d in range(n_dim)
            ]
            # Clamp to bounds.
            mutant = [max(lb, min(ub, m)) for m in mutant]

            # Binomial crossover.
            j_rand = rng.randint(0, n_dim - 1)
            trial = [
                mutant[d] if (rng.random() < cfg.CR or d == j_rand) else population[i][d]
                for d in range(n_dim)
            ]

            trial_score = evaluate(trial)
            n_eval += 1

            # Greedy selection (maximize).
            if trial_score >= scores[i]:
                population[i] = trial
                scores[i] = trial_score

                if trial_score > best_score:
                    best_score = trial_score
                    best_x = list(trial)

        history.append(best_score)

        # Convergence check.
        if abs(best_score - prev_best) < cfg.ftol:
            stagnant += 1
        else:
            stagnant = 0
        prev_best = best_score

        if stagnant >= cfg.patience:
            return _build_result(
                network, rate_indices, best_x, best_score, n_eval, generation,
                True, "patience", history, lb, ub,
            )

    stop = "converged_f" if stagnant >= cfg.patience else "maxeval"
    return _build_result(
        network, rate_indices, best_x, best_score, n_eval, generation,
        stop == "converged_f", stop, history, lb, ub,
    )


def _build_result(
    network: ReactionNetwork,
    rate_indices: List[int],
    best_x: List[float],
    best_score: float,
    n_eval: int,
    generations: int,
    converged: bool,
    stop_reason: str,
    history: List[float],
    lb: float,
    ub: float,
) -> DEResult:
    best_network = network.copy()
    best_rates = []
    for i, idx in enumerate(rate_indices):
        rate = 10.0 ** max(lb, min(ub, best_x[i]))
        best_network.rules[idx].rate = rate
        best_rates.append(rate)

    return DEResult(
        best_network=best_network,
        best_rates=best_rates,
        best_score=best_score,
        n_eval=n_eval,
        generations=generations,
        converged=converged,
        stop_reason=stop_reason,
        history=history,
    )
