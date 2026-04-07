
import random

from modern_biojazz.mutation import GraphMutator
from modern_biojazz.site_graph import Site, Protein, Rule, ReactionNetwork


def test_remove_protein_removes_associated_rules(seed_network):
    mutator = GraphMutator(random.Random(1))
    network = seed_network.copy()
    network.rules.append(
        Rule(
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
        "add_dephosphorylation",
        "add_inhibition",
        "add_unbinding",
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
        Rule(
            name="derived_refs",
            rule_type="inhibition",
            reactants=["STAT3_P", "SOCS3"],
            products=["STAT3:SOCS3"],
            rate=0.1,
        )
    )
    mutator.remove_protein(network, "STAT3")
    assert all("STAT3" not in " ".join([*r.reactants, *r.products]) for r in network.rules)


def test_remove_protein_refuses_to_empty_network():
    mutator = GraphMutator(random.Random(6))
    network = ReactionNetwork(
        proteins={"ONLY": Protein(name="ONLY", sites=[])},
        rules=[],
    )
    mutator.remove_protein(network, "ONLY")
    assert "ONLY" in network.proteins


def test_dephosphorylation_creates_reverse_reaction(seed_network):
    mutator = GraphMutator(random.Random(7))
    network = seed_network.copy()
    # Create a phosphorylated species first
    mutator.add_phosphorylation_rule(network, "STAT3", "SOCS3")
    assert "SOCS3_P" in network.proteins

    mutator.add_dephosphorylation_rule(network, "STAT3", "SOCS3_P")
    dephos_rules = [r for r in network.rules if r.rule_type == "dephosphorylation"]
    assert len(dephos_rules) == 1
    assert "SOCS3_P" in dephos_rules[0].reactants
    assert "SOCS3" in dephos_rules[0].products


def test_unbinding_creates_dissociation_rule(seed_network):
    mutator = GraphMutator(random.Random(8))
    network = seed_network.copy()
    # Create binding sites and a complex
    mutator.add_site(network, "STAT3", "b_SOCS3", "binding")
    mutator.add_site(network, "SOCS3", "b_STAT3", "binding")
    network.proteins["STAT3"].sites[-1].allowed_partners.append("SOCS3")
    network.proteins["SOCS3"].sites[-1].allowed_partners.append("STAT3")
    mutator.add_binding_rule(network, "STAT3", "SOCS3")
    assert "STAT3:SOCS3" in network.proteins

    mutator.add_unbinding_rule(network, "STAT3:SOCS3")
    unbind_rules = [r for r in network.rules if r.rule_type == "unbinding"]
    assert len(unbind_rules) == 1
    assert unbind_rules[0].reactants == ["STAT3:SOCS3"]
    assert set(unbind_rules[0].products) == {"STAT3", "SOCS3"}


def test_unbinding_ignores_non_complex():
    mutator = GraphMutator(random.Random(9))
    network = ReactionNetwork(
        proteins={"A": Protein(name="A", sites=[])},
        rules=[],
    )
    mutator.add_unbinding_rule(network, "A")
    assert len(network.rules) == 0


def test_dephosphorylation_ignores_non_phospho():
    mutator = GraphMutator(random.Random(10))
    network = ReactionNetwork(
        proteins={
            "A": Protein(name="A", sites=[]),
            "B": Protein(name="B", sites=[]),
        },
        rules=[],
    )
    mutator.add_dephosphorylation_rule(network, "A", "B")  # B doesn't end in _P
    assert len(network.rules) == 0


def test_replace_species_token_handles_inh_suffix():
    mutator = GraphMutator(random.Random(11))
    assert mutator._replace_species_token("X_inh", "X", "Y") == "Y_inh"
    assert mutator._replace_species_token("X_P", "X", "Y") == "Y_P"
    assert mutator._replace_species_token("X:Z", "X", "Y") == "Y:Z"
    assert mutator._replace_species_token("X", "X", "Y") == "Y"
    assert mutator._replace_species_token("OTHER", "X", "Y") == "OTHER"


def test_modify_rate_clamps_to_bounds():
    mutator = GraphMutator(random.Random(12))
    network = ReactionNetwork(
        proteins={"A": Protein(name="A", sites=[])},
        rules=[Rule(name="r1", rule_type="binding", reactants=["A"], products=["A"], rate=50.0)],
    )
    mutator.modify_rate(network, "r1", 1000.0)
    assert network.rules[0].rate == 100.0

    network.rules[0].rate = 1e-5
    mutator.modify_rate(network, "r1", 0.001)
    assert network.rules[0].rate == 1e-6


def test_add_kinase_cascade():
    mutator = GraphMutator(random.Random(123))
    net = ReactionNetwork()
    mutator.add_kinase_cascade(net)

    # 3 proteins + 2 intermediate phosphos
    assert len(net.proteins) == 5
    assert len(net.rules) == 2
    assert net.rules[0].rule_type == "phosphorylation"
    assert net.rules[1].rule_type == "phosphorylation"


def test_add_negative_feedback_loop():
    mutator = GraphMutator(random.Random(123))
    net = ReactionNetwork()
    mutator.add_negative_feedback_loop(net)

    # 3 proteins + 2 intermediate phosphos + 1 inhibition state
    assert len(net.proteins) == 6
    assert len(net.rules) == 3
    assert net.rules[-1].rule_type == "inhibition"


def test_crossover():
    mutator = GraphMutator(random.Random(123))

    net1 = ReactionNetwork()
    mutator.add_protein(net1, "A")
    mutator.add_protein(net1, "B")
    mutator.add_binding_rule(net1, "A", "B")

    net2 = ReactionNetwork()
    mutator.add_protein(net2, "X")
    mutator.add_protein(net2, "Y")
    mutator.add_negative_feedback_loop(net2)

    child = mutator.crossover(net1, net2)

    # Should contain some nodes from both
    assert "A" in child.proteins
    assert "B" in child.proteins

    # Randomly copied nodes from net2
    assert len(child.proteins) > 2


def test_add_protein_with_name():
    mutator = GraphMutator(random.Random(42))
    network = ReactionNetwork()
    mutator.add_protein(network, "NEW_PROT")
    assert "NEW_PROT" in network.proteins
    assert network.proteins["NEW_PROT"].name == "NEW_PROT"
    assert network.proteins["NEW_PROT"].sites == []


def test_add_protein_without_name():
    mutator = GraphMutator(random.Random(42))
    network = ReactionNetwork()
    mutator.add_protein(network, "A")
    mutator.add_protein(network)
    assert "P2" in network.proteins
    assert network.proteins["P2"].name == "P2"
    assert network.proteins["P2"].sites == []


def test_add_protein_already_exists():
    mutator = GraphMutator(random.Random(42))
    network = ReactionNetwork()
    network.proteins["EXISTING"] = Protein(name="EXISTING", sites=[Site(name="s1", site_type="binding")])
    mutator.add_protein(network, "EXISTING")
    assert "EXISTING" in network.proteins
    assert len(network.proteins["EXISTING"].sites) == 1
    assert network.proteins["EXISTING"].sites[0].name == "s1"
def test_duplicate_protein_with_rewiring_basic():
    mutator = GraphMutator(random.Random(42))
    net = ReactionNetwork()
    mutator.add_protein(net, "A")
    mutator.add_site(net, "A", "b1", "binding")
    net.proteins["A"].sites[0].allowed_partners.append("A")

    mutator.add_binding_rule(net, "A", "A", rate=0.1)

    assert len(net.proteins) == 2  # A and A:A
    assert len(net.rules) == 1

    mutator.duplicate_protein_with_rewiring(net, "A")

    # A duplicate protein should be created
    assert len(net.proteins) == 3  # A, A:A, and A_dup...
    dup_name = [p for p in net.proteins if p != "A" and p != "A:A"][0]
    assert dup_name.startswith("A_dup")

    # It should have rewired the rule A + A -> A:A
    # The duplicate should have allowed partners rewired
    dup_protein = net.proteins[dup_name]
    assert dup_protein.sites[0].allowed_partners == [dup_name]

    # The rules should include the original + 1 rewired rule
    assert len(net.rules) == 2
    rewired_rule = net.rules[1]
    assert rewired_rule.name.startswith("bind_A_A_1_rewired_")
    assert rewired_rule.reactants == [dup_name, dup_name]
    assert rewired_rule.products == [f"{dup_name}:{dup_name}"]


def test_duplicate_protein_missing_protein():
    mutator = GraphMutator(random.Random(42))
    net = ReactionNetwork()
    mutator.add_protein(net, "A")

    mutator.duplicate_protein_with_rewiring(net, "B")

    assert len(net.proteins) == 1
    assert "A" in net.proteins
