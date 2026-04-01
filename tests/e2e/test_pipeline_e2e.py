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
        grounding_payload=grounding_payload,
    )

    assert result.evolution.best_score >= 0.0
    assert result.grounding is not None
    assert result.grounding.candidates_considered >= 1

    # Validate all tokens are within the grounding constraint vocabulary.
    # The filter allows: base proteins, _P, _inh, _dupNNNN, colon-complexes.
    allowed = set(grounding_payload["abstract"]["nodes"]) | set(seed_network.proteins.keys())

    def _base(tok: str) -> str:
        if tok.endswith("_P"):
            return tok[:-2]
        if tok.endswith("_inh"):
            return tok[:-4]
        if "_dup" in tok:
            return tok[: tok.index("_dup")]
        return tok

    def _ok(tok: str) -> bool:
        if _base(tok) in allowed:
            return True
        if ":" in tok:
            return all(_base(p) in allowed for p in tok.split(":"))
        return False

    for rule in result.evolution.best_network.rules:
        for token in [*rule.reactants, *rule.products]:
            assert _ok(token), f"Found token outside grounding constraints: {token}"
    for pname in result.evolution.best_network.proteins:
        assert _ok(pname), f"Found protein name outside grounding constraints: {pname}"
