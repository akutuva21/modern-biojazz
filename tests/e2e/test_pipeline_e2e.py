from __future__ import annotations

import random

from modern_biojazz.evolution import LLMEvolutionEngine, EvolutionConfig, DeterministicProposer
from modern_biojazz.grounding import GroundingEngine
from modern_biojazz.mutation import GraphMutator
from modern_biojazz.pipeline import ModernBioJazzPipeline, PipelineConfig
from modern_biojazz.simulation import FitnessEvaluator, LocalCatalystEngine


def test_full_pipeline_e2e(seed_network, grounding_payload):
    evolution_engine = LLMEvolutionEngine(
        simulation_backend=LocalCatalystEngine(),
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=DeterministicProposer(),
        mutator=GraphMutator(random.Random(21)),
        rng=random.Random(21),
    )
    pipeline = ModernBioJazzPipeline(evolution_engine, GroundingEngine())

    result = pipeline.run(
        seed_network=seed_network,
        config=PipelineConfig(
            evolution=EvolutionConfig(
                population_size=8,
                generations=3,
                mutations_per_candidate=2,
                islands=2,
                migration_interval=2,
                migration_count=1,
            ),
            do_grounding=True,
        ),
        grounding_payload={
            "abstract_types": grounding_payload["abstract_types"],
            "real_nodes": grounding_payload["real_nodes"],
            "real_interactions": [tuple(x) for x in grounding_payload["real_interactions"]],
            "confidence_by_pair": grounding_payload["confidence_by_pair"],
        },
    )

    assert result.evolution.best_score >= 0.0
    assert result.grounding is not None
    assert result.grounding.candidates_considered >= 1

    allowed = set(grounding_payload["abstract_types"].keys())
    for rule in result.evolution.best_network.rules:
        for token in [*rule.reactants, *rule.products]:
            if token in allowed:
                continue
            if token.endswith("_P") and token[:-2] in allowed:
                continue
            if token.endswith("_inh") and token[:-4] in allowed:
                continue
            if ":" in token and all(part in allowed for part in token.split(":")):
                continue
            raise AssertionError(f"Found token outside grounding constraints: {token}")
