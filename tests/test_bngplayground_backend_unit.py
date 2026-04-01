from modern_biojazz.bngplayground_backend import BNGPlaygroundBackend
from modern_biojazz.site_graph import ReactionNetwork, Protein, Site, Rule

def test_network_to_bngl():
    proteins = {
        "A": Protein(name="A", sites=[Site(name="s1", site_type="binding")]),
        "B": Protein(name="B", sites=[Site(name="s1", site_type="binding")])
    }
    rules = [
        Rule(name="r1", rule_type="binding", reactants=["A", "B"], products=["A_B"], rate=0.1)
    ]

    network = ReactionNetwork(proteins=proteins, rules=rules)
    backend = BNGPlaygroundBackend(bngplayground_path="/dummy/path")

    bngl = backend._network_to_bngl(network, t_end=10.0, dt=1.0)

    assert "begin model" in bngl
    assert "r1_rate 0.1" in bngl
    assert "A(s1)" in bngl
    assert "B(s1)" in bngl
    assert "r1: A() + B() -> A_B() r1_rate" in bngl
