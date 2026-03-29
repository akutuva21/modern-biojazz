from __future__ import annotations

import random

from modern_biojazz.mutation import GraphMutator
from modern_biojazz.site_graph import Site


def test_remove_protein_removes_associated_rules(seed_network):
    mutator = GraphMutator(random.Random(1))
    network = seed_network.copy()
    network.rules.append(
        type(network.rules[0])(
            name="bind_temp",
            rule_type="binding",
            reactants=["STAT3", "SOCS3"],
            products=["STAT3:SOCS3"],
            rate=0.1,
        )
    )
    mutator.remove_protein(network, "STAT3")
    assert "STAT3" not in network.proteins
    assert all("STAT3" not in r.reactants for r in network.rules)


def test_action_library_contains_extended_operators(seed_network):
    mutator = GraphMutator(random.Random(2))
    actions = mutator.action_library(seed_network)
    expected = {
        "add_site",
        "add_binding",
        "add_phosphorylation",
        "add_inhibition",
        "remove_rule",
        "modify_rate",
        "remove_site",
        "duplicate_protein",
        "remove_protein",
    }
    assert expected.issubset(set(actions.keys()))


def test_binding_compatibility_requires_both_directions(seed_network):
    mutator = GraphMutator(random.Random(4))
    network = seed_network.copy()
    for site in network.proteins["SOCS3"].sites:
        if site.site_type == "binding":
            site.allowed_partners = ["NOT_STAT3"]
    network.proteins["STAT3"].sites.append(Site(name="b1", site_type="binding", allowed_partners=["SOCS3"]))
    network.proteins["SOCS3"].sites.append(Site(name="b2", site_type="binding", allowed_partners=["NOT_STAT3"]))

    # SOCS3 does not explicitly allow STAT3 here, so compatibility should fail.
    mutator.add_binding_rule(network, "STAT3", "SOCS3")
    assert all(r.rule_type != "binding" for r in network.rules)


def test_remove_protein_cleans_derived_tokens(seed_network):
    mutator = GraphMutator(random.Random(5))
    network = seed_network.copy()
    network.rules.append(
        type(network.rules[0])(
            name="derived_refs",
            rule_type="inhibition",
            reactants=["STAT3_P", "SOCS3"],
            products=["STAT3:SOCS3"],
            rate=0.1,
        )
    )
    mutator.remove_protein(network, "STAT3")
    assert all("STAT3" not in " ".join([*r.reactants, *r.products]) for r in network.rules)
