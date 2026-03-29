from __future__ import annotations

import pytest

from modern_biojazz.site_graph import ReactionNetwork, ReactionNetworkValidationError


def test_roundtrip_serialization(seed_network):
    payload = seed_network.to_dict()
    reconstructed = ReactionNetwork.from_dict(payload)
    assert reconstructed.to_dict() == payload


def test_copy_metadata_isolation(seed_network):
    seed_network.metadata["nested"] = {"a": 1}
    copied = seed_network.copy()
    copied.metadata["nested"]["a"] = 999
    assert seed_network.metadata["nested"]["a"] == 1


def test_from_dict_missing_key_raises_helpful_error():
    with pytest.raises(ReactionNetworkValidationError, match="missing required key 'name'"):
        ReactionNetwork.from_dict(
            {
                "proteins": {"P1": {"sites": []}},
                "rules": [],
                "metadata": {},
            }
        )


def test_from_dict_invalid_rule_shape_raises_helpful_error():
    with pytest.raises(ReactionNetworkValidationError, match="missing required keys"):
        ReactionNetwork.from_dict(
            {
                "proteins": {"P1": {"name": "P1", "sites": []}},
                "rules": [{"name": "r1", "rule_type": "binding"}],
                "metadata": {},
            }
        )
