from __future__ import annotations

from modern_biojazz.grounding import GroundingEngine


def test_grounding_engine_e2e(seed_network, grounding_payload):
    engine = GroundingEngine()

    constraints = engine.build_constraint_matrix(
        grounding_payload["abstract_types"],
        grounding_payload["real_nodes"],
    )
    mappings = engine.match_abstract_to_real(
        seed_network,
        constraints,
        [tuple(x) for x in grounding_payload["real_interactions"]],
    )
    result = engine.score_mappings(mappings, grounding_payload["confidence_by_pair"])

    assert "STAT3" in constraints
    assert result.candidates_considered >= 1
    assert result.mapping.get("SOCS3") == "SOCS3_HUMAN"
    assert result.score > 0.0
