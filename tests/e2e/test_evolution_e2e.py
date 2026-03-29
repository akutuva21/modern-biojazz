from __future__ import annotations

import random

from modern_biojazz.evolution import LLMEvolutionEngine, EvolutionConfig, DeterministicProposer
from modern_biojazz.mutation import GraphMutator
from modern_biojazz.site_graph import ReactionNetwork
from modern_biojazz.simulation import LocalCatalystEngine, FitnessEvaluator


def test_llm_guided_evolution_e2e(seed_network):
    engine = LLMEvolutionEngine(
        simulation_backend=LocalCatalystEngine(),
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=DeterministicProposer(),
        mutator=GraphMutator(random.Random(3)),
        rng=random.Random(3),
    )

    result = engine.run(
        seed_network,
        EvolutionConfig(
            population_size=8,
            generations=4,
            mutations_per_candidate=2,
            islands=2,
            migration_interval=2,
            migration_count=1,
        ),
    )

    assert result.best_score >= 0.0
    assert len(result.history) == 5
    assert len(result.best_network.proteins) >= len(seed_network.proteins)


def test_evolution_handles_empty_seed_network():
    engine = LLMEvolutionEngine(
        simulation_backend=LocalCatalystEngine(),
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=DeterministicProposer(),
        mutator=GraphMutator(random.Random(8)),
        rng=random.Random(8),
    )
    empty = ReactionNetwork()

    result = engine.run(
        empty,
        EvolutionConfig(population_size=4, generations=2, mutations_per_candidate=1, islands=1),
    )

    assert result.best_score >= 0.0
    assert len(result.history) == 3


class _RejectingProposer:
    def propose(self, model_code: str, action_names: list[str], budget: int) -> list[str]:
        del model_code
        del budget
        return ["add_phosphorylation"] if "add_phosphorylation" in action_names else action_names[:1]

    def record_feedback(self, score: float, notes: str) -> None:
        del score
        del notes


def test_mutate_candidate_uses_safe_fallback_when_filter_rejects(seed_network):
    initial_rule_count = len(seed_network.rules)
    engine = LLMEvolutionEngine(
        simulation_backend=LocalCatalystEngine(),
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=_RejectingProposer(),
        mutator=GraphMutator(random.Random(9)),
        rng=random.Random(9),
        candidate_filter=lambda net: len(net.rules) <= initial_rule_count,
    )

    candidate = engine._mutate_candidate(seed_network, budget=1)
    assert engine.filter_rejection_count == 1
    assert len(candidate.rules) <= initial_rule_count


def test_cegis_feedback_on_filter_rejection(seed_network):
    class FeedbackProposer:
        def __init__(self):
            self.feedback = []

        def propose(self, model_code: str, action_names: list[str], budget: int) -> list[str]:
            del model_code
            del budget
            return action_names[:1]

        def record_feedback(self, score: float, notes: str) -> None:
            self.feedback.append((score, notes))

    proposer = FeedbackProposer()
    engine = LLMEvolutionEngine(
        simulation_backend=LocalCatalystEngine(),
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=proposer,
        mutator=GraphMutator(random.Random(9)),
        rng=random.Random(9),
        candidate_filter=lambda net: False,
    )

    _ = engine._mutate_candidate(seed_network, budget=1)
    assert any(
        '"failure_type":"filter_rejection"' in notes or '"failure_type": "filter_rejection"' in notes
        for _, notes in proposer.feedback
    )


def test_cegis_feedback_on_simulation_exception(seed_network):
    class ThrowingScorer:
        def score(self, *args, **kwargs):
            raise RuntimeError("simulate fail")

    class FeedbackProposer:
        def __init__(self):
            self.feedback = []

        def propose(self, model_code: str, action_names: list[str], budget: int) -> list[str]:
            del model_code
            del action_names
            del budget
            return []

        def record_feedback(self, score: float, notes: str) -> None:
            self.feedback.append((score, notes))

    proposer = FeedbackProposer()
    engine = LLMEvolutionEngine(
        simulation_backend=LocalCatalystEngine(),
        fitness_evaluator=ThrowingScorer(),
        proposer=proposer,
        mutator=GraphMutator(random.Random(9)),
        rng=random.Random(9),
    )

    score = engine._evaluate(seed_network, EvolutionConfig(population_size=1, generations=1, mutations_per_candidate=1, islands=1))
    assert score == 0.0
    assert any(
        '"failure_type":"simulation_exception"' in notes or '"failure_type": "simulation_exception"' in notes
        for _, notes in proposer.feedback
    )
