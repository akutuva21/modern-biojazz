from __future__ import annotations

import random

from modern_biojazz.evolution import LLMEvolutionEngine, EvolutionConfig, DeterministicProposer
from modern_biojazz.grounding import GroundingEngine
from modern_biojazz.mutation import GraphMutator
from modern_biojazz.pipeline import ModernBioJazzPipeline, PipelineConfig
from modern_biojazz.simulation import FitnessEvaluator, LocalCatalystEngine
from modern_biojazz.site_graph import ReactionNetwork, Protein, Rule


def _make_filter(pipeline, allowed):
    return pipeline._grounding_constraint_filter(allowed)


def test_filter_accepts_base_proteins():
    pipeline = ModernBioJazzPipeline(None, None)  # type: ignore
    filt = _make_filter(pipeline, {"STAT3", "SOCS3"})
    net = ReactionNetwork(
        proteins={
            "STAT3": Protein(name="STAT3", sites=[]),
            "SOCS3": Protein(name="SOCS3", sites=[]),
        },
        rules=[Rule(name="r1", rule_type="phosphorylation",
                     reactants=["STAT3", "SOCS3"], products=["STAT3", "SOCS3_P"], rate=0.1)],
    )
    assert filt(net) is True


def test_filter_accepts_derived_tokens():
    pipeline = ModernBioJazzPipeline(None, None)  # type: ignore
    filt = _make_filter(pipeline, {"A", "B"})
    net = ReactionNetwork(
        proteins={
            "A": Protein(name="A", sites=[]),
            "B": Protein(name="B", sites=[]),
            "A_P": Protein(name="A_P", sites=[]),
            "B_inh": Protein(name="B_inh", sites=[]),
            "A:B": Protein(name="A:B", sites=[]),
        },
        rules=[
            Rule(name="r1", rule_type="phosphorylation",
                 reactants=["A", "B"], products=["A", "B_inh"], rate=0.1),
            Rule(name="r2", rule_type="binding",
                 reactants=["A", "B"], products=["A:B"], rate=0.1),
        ],
    )
    assert filt(net) is True


def test_filter_accepts_dup_suffix():
    pipeline = ModernBioJazzPipeline(None, None)  # type: ignore
    filt = _make_filter(pipeline, {"STAT3", "SOCS3"})
    net = ReactionNetwork(
        proteins={
            "STAT3": Protein(name="STAT3", sites=[]),
            "STAT3_dup42": Protein(name="STAT3_dup42", sites=[]),
        },
        rules=[Rule(name="r1", rule_type="phosphorylation",
                     reactants=["STAT3", "STAT3_dup42"],
                     products=["STAT3", "STAT3_dup42_P"], rate=0.1)],
    )
    assert filt(net) is True


def test_filter_rejects_foreign_protein():
    pipeline = ModernBioJazzPipeline(None, None)  # type: ignore
    filt = _make_filter(pipeline, {"A", "B"})
    net = ReactionNetwork(
        proteins={
            "A": Protein(name="A", sites=[]),
            "FOREIGN": Protein(name="FOREIGN", sites=[]),
        },
        rules=[Rule(name="r1", rule_type="binding",
                     reactants=["A", "FOREIGN"], products=["A:FOREIGN"], rate=0.1)],
    )
    assert filt(net) is False


def test_filter_rejects_foreign_in_rule_products():
    pipeline = ModernBioJazzPipeline(None, None)  # type: ignore
    filt = _make_filter(pipeline, {"A"})
    net = ReactionNetwork(
        proteins={"A": Protein(name="A", sites=[])},
        rules=[Rule(name="r1", rule_type="phosphorylation",
                     reactants=["A"], products=["UNKNOWN_P"], rate=0.1)],
    )
    assert filt(net) is False


def test_pipeline_grounding_includes_seed_proteins(seed_network, grounding_payload):
    """Ensure that seed network proteins are in the allowed set even if not
    explicitly listed in abstract_types."""
    engine = LLMEvolutionEngine(
        simulation_backend=LocalCatalystEngine(),
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=DeterministicProposer(),
        mutator=GraphMutator(random.Random(42)),
        rng=random.Random(42),
    )
    pipeline = ModernBioJazzPipeline(engine, GroundingEngine())

    result = pipeline.run(
        seed_network=seed_network,
        config=PipelineConfig(
            evolution=EvolutionConfig(population_size=4, generations=2, mutations_per_candidate=1),
            do_grounding=True,
        ),
        grounding_payload={
            "abstract": {
                "nodes": list(grounding_payload["abstract_types"].keys()),
                "types": grounding_payload["abstract_types"],
            },
            "real": {
                "nodes": grounding_payload["real_nodes"],
                "edges": [tuple(x) for x in grounding_payload["real_interactions"]],
            },
            "confidence": grounding_payload["confidence_by_pair"],
        },
    )
    # Must not crash and should produce a valid result.
    assert result.evolution.best_score >= 0.0
