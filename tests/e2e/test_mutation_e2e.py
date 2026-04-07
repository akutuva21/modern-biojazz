
import random

from modern_biojazz.mutation import GraphMutator


def test_graph_mutation_e2e(seed_network):
    mutator = GraphMutator(random.Random(11))
    network = seed_network.copy()

    mutator.add_protein(network, "JAK1")
    mutator.add_site(network, "JAK1", "bind_STAT3", "binding")
    mutator.add_site(network, "STAT3", "bind_JAK1", "binding")
    mutator.add_site(network, "JAK1", "Ksite", "modification")
    network.proteins["JAK1"].sites[0].allowed_partners.append("STAT3")
    network.proteins["STAT3"].sites[-1].allowed_partners.append("JAK1")
    mutator.add_binding_rule(network, "JAK1", "STAT3")
    mutator.add_phosphorylation_rule(network, "JAK1", "STAT3")
    mutator.duplicate_protein_with_rewiring(network, "STAT3")

    assert "JAK1" in network.proteins
    assert any(s.name == "Ksite" for s in network.proteins["JAK1"].sites)
    assert any(r.rule_type == "binding" for r in network.rules)
    assert any(r.rule_type == "phosphorylation" for r in network.rules)
    assert "JAK1:STAT3" in network.proteins
    assert "STAT3_P" in network.proteins
    assert any(name.startswith("STAT3_dup") for name in network.proteins)

    # Reverse reactions
    mutator.add_dephosphorylation_rule(network, "JAK1", "STAT3_P")
    mutator.add_unbinding_rule(network, "JAK1:STAT3")

    assert any(r.rule_type == "dephosphorylation" for r in network.rules)
    assert any(r.rule_type == "unbinding" for r in network.rules)

    # Inhibition
    mutator.add_inhibition_rule(network, "SOCS3", "JAK1")
    assert "JAK1_inh" in network.proteins
    assert any(r.rule_type == "inhibition" for r in network.rules)
