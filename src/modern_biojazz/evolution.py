from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, List, Protocol

from .mutation import GraphMutator
from .site_graph import ReactionNetwork
from .simulation import SimulationBackend, FitnessScorer


class LLMProposer(Protocol):
    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        ...


@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 10
    mutations_per_candidate: int = 3
    islands: int = 2
    migration_interval: int = 2
    migration_count: int = 1
    sim_t_end: float = 20.0
    sim_dt: float = 1.0
    sim_solver: str = "Rodas5P"


@dataclass
class EvolutionResult:
    best_network: ReactionNetwork
    best_score: float
    history: List[float]


class LLMEvolutionEngine:
    def __init__(
        self,
        simulation_backend: SimulationBackend,
        fitness_evaluator: FitnessScorer,
        proposer: LLMProposer,
        mutator: GraphMutator | None = None,
        rng: random.Random | None = None,
        candidate_filter: Callable[[ReactionNetwork], bool] | None = None,
    ) -> None:
        self.backend = simulation_backend
        self.fitness = fitness_evaluator
        self.proposer = proposer
        self.mutator = mutator or GraphMutator()
        self.rng = rng or random.Random()
        self.candidate_filter = candidate_filter
        self.filter_rejection_count = 0

    def _cegis_feedback(self, network: ReactionNetwork, score: float, failure_type: str, details: dict) -> None:
        if not hasattr(self.proposer, "record_feedback"):
            return

        # Keep the counterexample message small and structured.
        proto = {
            "cegis": True,
            "failure_type": failure_type,
            "score": float(score),
            "details": details,
            "network_summary": {
                "n_proteins": len(network.proteins),
                "n_rules": len(network.rules),
                "output_species": network.metadata.get("output_species"),
            },
        }
        try:
            self.proposer.record_feedback(score, json.dumps(proto))
        except Exception:
            # Don't break evolution if proposer feedback path is untrusted.
            pass

    def _model_code(self, network: ReactionNetwork) -> str:
        rule_types: dict[str, int] = {}
        for r in network.rules:
            rule_types[r.rule_type] = rule_types.get(r.rule_type, 0) + 1

        protein_names = sorted(network.proteins.keys())
        preview_rules = []
        for r in network.rules[: min(20, len(network.rules))]:
            left = "+".join(r.reactants)
            right = "+".join(r.products)
            preview_rules.append(f"{r.rule_type}:{left}->{right}@{r.rate:.3g}")

        return (
            f"n_proteins={len(network.proteins)};"
            f"n_rules={len(network.rules)};"
            f"rule_types={rule_types};"
            f"proteins={protein_names};"
            f"rules_preview={preview_rules}"
        )

    def _mutate_candidate(self, network: ReactionNetwork, budget: int) -> ReactionNetwork:
        last_child: ReactionNetwork | None = None
        for _ in range(8):
            child = network.copy()
            last_child = child
            actions = self.mutator.action_library(child)
            choices = self.proposer.propose(self._model_code(child), list(actions.keys()), budget)
            for action_name in choices:
                action = actions.get(action_name)
                if action is not None:
                    action.apply(child)
            if self.candidate_filter is None or self.candidate_filter(child):
                return child

        self.filter_rejection_count += 1
        self._cegis_feedback(
            network=last_child or network,
            score=0.0,
            failure_type="filter_rejection",
            details={
                "attempts": 8,
                "total_rejections": self.filter_rejection_count,
                "source_rules": len(network.rules),
            },
        )

        # Try low-risk edits that preserve the existing token alphabet before giving up.
        safe = network.copy()
        safe_actions = self.mutator.action_library(safe)
        for action_name in ["modify_rate", "remove_rule", "remove_site"]:
            action = safe_actions.get(action_name)
            if action is not None:
                action.apply(safe)
                if self.candidate_filter is None or self.candidate_filter(safe):
                    return safe

        if last_child is not None and (self.candidate_filter is None or self.candidate_filter(last_child)):
            return last_child
        return network.copy()

    def _evaluate(self, network: ReactionNetwork, config: EvolutionConfig) -> float:
        try:
            score = self.fitness.score(
                backend=self.backend,
                network=network,
                t_end=config.sim_t_end,
                dt=config.sim_dt,
                solver=config.sim_solver,
            )
        except Exception as error:
            self._cegis_feedback(
                network=network,
                score=0.0,
                failure_type="simulation_exception",
                details={"error": str(error)},
            )
            return 0.0

        if score <= 0.0:
            self._cegis_feedback(
                network=network,
                score=score,
                failure_type="low_fitness",
                details={"desc": "score<=0.0, no improvement"},
            )

        return score

    def run(self, seed: ReactionNetwork, config: EvolutionConfig) -> EvolutionResult:
        islands: List[List[ReactionNetwork]] = []
        for _ in range(max(1, config.islands)):
            island_pop = [seed.copy()] + [
                self._mutate_candidate(seed, config.mutations_per_candidate)
                for _ in range(max(0, config.population_size - 1))
            ]
            islands.append(island_pop)

        all_initial = [n for island in islands for n in island]
        scored_initial = [(self._evaluate(n, config), n) for n in all_initial]
        scored_initial.sort(key=lambda x: x[0], reverse=True)
        best_score, best_network = scored_initial[0]
        best = best_network.copy()
        history: List[float] = [best_score]

        for generation in range(config.generations):
            for island_idx, island_pop in enumerate(islands):
                scored = [(self._evaluate(n, config), n) for n in island_pop]
                scored.sort(key=lambda x: x[0], reverse=True)
                island_best_score, island_best = scored[0]
                if island_best_score > best_score:
                    best_score = island_best_score
                    best = island_best.copy()

                survivors = [n for _, n in scored[: max(2, config.population_size // 3)]]
                new_pop: List[ReactionNetwork] = [island_best.copy()]
                while len(new_pop) < config.population_size:
                    parent = self.rng.choice(survivors)
                    child = self._mutate_candidate(parent, config.mutations_per_candidate)
                    new_pop.append(child)
                islands[island_idx] = new_pop

            if (generation + 1) % max(1, config.migration_interval) == 0 and len(islands) > 1:
                self._migrate(islands, config.migration_count)

            if hasattr(self.proposer, "record_feedback"):
                self.proposer.record_feedback(best_score, f"generation={generation+1}")

            history.append(best_score)

        return EvolutionResult(best_network=best, best_score=best_score, history=history)

    def _migrate(self, islands: List[List[ReactionNetwork]], migration_count: int) -> None:
        migrants_by_island: List[List[ReactionNetwork]] = []
        for pop in islands:
            migrants = [self.rng.choice(pop).copy() for _ in range(max(1, migration_count))]
            migrants_by_island.append(migrants)

        for idx, pop in enumerate(islands):
            source = migrants_by_island[idx - 1]
            for m in source:
                replace_idx = self.rng.randint(0, len(pop) - 1)
                pop[replace_idx] = m


@dataclass
class DeterministicProposer:
    """Test-friendly proposer that cycles over available mutation actions."""

    cursor: int = 0
    feedback_log: List[str] | None = None

    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        if not action_names:
            return []
        out: List[str] = []
        for _ in range(max(1, budget)):
            out.append(action_names[self.cursor % len(action_names)])
            self.cursor += 1
        return out

    def record_feedback(self, score: float, notes: str) -> None:
        if self.feedback_log is None:
            self.feedback_log = []
        self.feedback_log.append(f"score={score:.4f};{notes}")
