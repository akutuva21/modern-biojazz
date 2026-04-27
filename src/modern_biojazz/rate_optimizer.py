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
from typing import Callable, List, Optional

from .site_graph import ReactionNetwork, Rule


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
class DEState:
    """Internal state for Differential Evolution."""
    best_x: List[float]
    best_score: float
    n_eval: int
    generations: int
    converged: bool
    stop_reason: str
    history: List[float]


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

    population = _initialize_population(
        network, rate_indices, pop_size, n_dim, lb, ub, rng
    )

    evaluate = _make_evaluator(network, rate_indices, lb, ub, objective)

    state = _run_de_loop(population, n_dim, pop_size, lb, ub, cfg, rng, evaluate)

    return _build_result(network, rate_indices, state, lb, ub)


def _initialize_population(
    network: ReactionNetwork,
    rate_indices: List[int],
    pop_size: int,
    n_dim: int,
    lb: float,
    ub: float,
    rng: random.Random,
) -> List[List[float]]:
    """Initialize population in log10 space with seed from current rates."""
    population: List[List[float]] = []
    for _ in range(pop_size):
        individual = [rng.uniform(lb, ub) for _ in range(n_dim)]
        population.append(individual)

    seed_rates = []
    for idx in rate_indices:
        rate = max(1e-12, network.rules[idx].rate)
        seed_rates.append(math.log10(rate))
    population[0] = [max(lb, min(ub, x)) for x in seed_rates]
    return population


def _run_de_loop(
    population: List[List[float]],
    n_dim: int,
    pop_size: int,
    lb: float,
    ub: float,
    cfg: DEConfig,
    rng: random.Random,
    evaluate: Callable[[List[float]], float],
) -> DEState:
    """Run the main Differential Evolution loop."""
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

            trial = _mutate_and_crossover(
                i, population, n_dim, pop_size, lb, ub, cfg, rng
            )

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
            return DEState(
                best_x=best_x,
                best_score=best_score,
                n_eval=n_eval,
                generations=generation,
                converged=True,
                stop_reason="patience",
                history=history,
            )

    stop = "converged_f" if stagnant >= cfg.patience else "maxeval"
    return DEState(
        best_x=best_x,
        best_score=best_score,
        n_eval=n_eval,
        generations=generation,
        converged=(stop == "converged_f"),
        stop_reason=stop,
        history=history,
    )


def _mutate_and_crossover(
    i: int,
    population: List[List[float]],
    n_dim: int,
    pop_size: int,
    lb: float,
    ub: float,
    cfg: DEConfig,
    rng: random.Random,
) -> List[float]:
    """Generate a trial vector using DE/rand/1/bin."""
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
    return trial


def _make_evaluator(
    network: ReactionNetwork,
    rate_indices: List[int],
    lb: float,
    ub: float,
    objective: Callable[[ReactionNetwork], float],
) -> Callable[[List[float]], float]:
    """Create the evaluation function for DE."""
    def evaluate(log_rates: List[float]) -> float:
        candidate = network.copy()
        for i, idx in enumerate(rate_indices):
            r = candidate.rules[idx]
            candidate.rules[idx] = Rule(
                name=r.name,
                rule_type=r.rule_type,
                reactants=r.reactants,
                products=r.products,
                rate=10.0 ** max(lb, min(ub, log_rates[i]))
            )
        try:
            return objective(candidate)
        except Exception:
            return 0.0
    return evaluate


def _build_result(
    network: ReactionNetwork,
    rate_indices: List[int],
    state: DEState,
    lb: float,
    ub: float,
) -> DEResult:
    best_network = network.copy()
    best_rates = []
    for i, idx in enumerate(rate_indices):
        rate = 10.0 ** max(lb, min(ub, state.best_x[i]))
        r = best_network.rules[idx]
        best_network.rules[idx] = Rule(
            name=r.name,
            rule_type=r.rule_type,
            reactants=r.reactants,
            products=r.products,
            rate=rate
        )
        best_rates.append(rate)

    return DEResult(
        best_network=best_network,
        best_rates=best_rates,
        best_score=state.best_score,
        n_eval=state.n_eval,
        generations=state.generations,
        converged=state.converged,
        stop_reason=state.stop_reason,
        history=state.history,
    )
