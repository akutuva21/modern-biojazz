from __future__ import annotations

import json
import random
from unittest.mock import MagicMock


from modern_biojazz.evolution import (
    EvolutionConfig,
    LLMEvolutionEngine,
    RandomProposer,
    DeterministicProposer,
)
from modern_biojazz.site_graph import ReactionNetwork, Protein
from modern_biojazz.simulation import SimulationBackend, FitnessScorer


def test_random_proposer():
    proposer = RandomProposer(rng=random.Random(42))
    actions = proposer.propose("model", ["a", "b", "c"], budget=2)
    assert len(actions) == 2
    assert set(actions).issubset({"a", "b", "c"})

    assert proposer.propose("model", [], budget=2) == []


def test_deterministic_proposer():
    proposer = DeterministicProposer()
    actions = proposer.propose("model", ["a", "b", "c"], budget=4)
    assert actions == ["a", "b", "c", "a"]

    proposer.record_feedback(0.5, "test")
    assert proposer.feedback_log is not None
    assert len(proposer.feedback_log) == 1
    assert "score=0.5000;test" in proposer.feedback_log[0]

    assert proposer.propose("model", [], budget=4) == []


def test_llm_evolution_engine_evaluate(seed_network: ReactionNetwork):
    backend = MagicMock(spec=SimulationBackend)
    fitness = MagicMock(spec=FitnessScorer)
    fitness.score.return_value = 10.0
    proposer = DeterministicProposer()
    engine = LLMEvolutionEngine(simulation_backend=backend, fitness_evaluator=fitness, proposer=proposer)

    config = EvolutionConfig()
    score = engine._evaluate(seed_network, config)
    assert score == 10.0

    # Test exception handling
    fitness.score.side_effect = Exception("sim error")
    err_network = ReactionNetwork()
    err_network.proteins["err"] = Protein(name="err")
    score_err = engine._evaluate(err_network, config)
    assert score_err == 0.0

    # Test low fitness
    fitness.score.side_effect = None
    fitness.score.return_value = -1.0
    low_network = ReactionNetwork()
    low_network.proteins["low"] = Protein(name="low")
    score_low = engine._evaluate(low_network, config)
    assert score_low == -1.0


def test_llm_evolution_engine_run(seed_network: ReactionNetwork):
    backend = MagicMock(spec=SimulationBackend)
    fitness = MagicMock(spec=FitnessScorer)
    fitness.score.return_value = 5.0
    proposer = DeterministicProposer()

    engine = LLMEvolutionEngine(simulation_backend=backend, fitness_evaluator=fitness, proposer=proposer)

    config = EvolutionConfig(population_size=2, generations=2, islands=1, mutations_per_candidate=1)
    result = engine.run(seed_network, config)

    assert result.best_score == 5.0
    assert len(result.history) == 3  # init + 2 gens
    assert len(result.generation_summary) == 3


def test_llm_evolution_engine_mutate_candidate(seed_network: ReactionNetwork):
    backend = MagicMock(spec=SimulationBackend)
    fitness = MagicMock(spec=FitnessScorer)
    proposer = DeterministicProposer()
    engine = LLMEvolutionEngine(simulation_backend=backend, fitness_evaluator=fitness, proposer=proposer)

    child = engine._mutate_candidate(seed_network, budget=1)
    assert isinstance(child, ReactionNetwork)


def test_llm_evolution_engine_cegis_feedback(seed_network: ReactionNetwork):
    backend = MagicMock(spec=SimulationBackend)
    fitness = MagicMock(spec=FitnessScorer)
    proposer = MagicMock()
    proposer.record_feedback = MagicMock()
    engine = LLMEvolutionEngine(simulation_backend=backend, fitness_evaluator=fitness, proposer=proposer)

    engine._cegis_feedback(seed_network, 0.5, "test", {"info": "data"})
    proposer.record_feedback.assert_called_once()
    call_args = proposer.record_feedback.call_args[0]
    assert call_args[0] == 0.5
    feedback_json = json.loads(call_args[1])
    assert feedback_json["failure_type"] == "test"
    assert feedback_json["details"] == {"info": "data"}

    # test untrusted proposer path (exception)
    proposer.record_feedback.side_effect = Exception("untrusted")
    # Should not raise exception
    engine._cegis_feedback(seed_network, 0.5, "test", {"info": "data"})
