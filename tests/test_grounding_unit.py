import pytest
from modern_biojazz.grounding import GroundingEngine, GroundingResult
from modern_biojazz.site_graph import ReactionNetwork, Protein, Rule

def test_normalize_edge_type():
    engine = GroundingEngine()
    assert engine._normalize_edge_type("dephosph") == "dephosphorylation"
    assert engine._normalize_edge_type("phosph") == "phosphorylation"
    assert engine._normalize_edge_type("inhib") == "inhibition"
    assert engine._normalize_edge_type("unbind") == "unbinding"
    assert engine._normalize_edge_type("dissoc") == "unbinding"
    assert engine._normalize_edge_type("bind") == "binding"
    assert engine._normalize_edge_type("complex") == "binding"
    assert engine._normalize_edge_type("unknown") == "unknown"

def test_build_constraint_matrix():
    engine = GroundingEngine()
    abstract_types = {"A": "Kinase", "B": "Phosphatase"}
    real_nodes = [{"name": "node1", "type": "Kinase"}, {"name": "node2", "type": "Kinase"}, {"name": "node3", "type": "Phosphatase"}, {"name": "node4"}]

    constraints = engine.build_constraint_matrix(abstract_types, real_nodes)

    assert constraints == {"A": ["node1", "node2"], "B": ["node3"]}

def test_prune_constraints_by_degree():
    engine = GroundingEngine()
    network = ReactionNetwork(
        proteins={"A": Protein(name="A"), "B": Protein(name="B")},
        rules=[Rule(name="r1", rule_type="binding", reactants=["A", "B"], products=["A_B"], rate=1.0)]
    )
    constraints = {"A": ["real_A1", "real_A2"], "B": ["real_B1"]}
    real_interactions = [("real_A1", "real_B1", "binding")]

    pruned = engine.prune_constraints_by_degree(network, constraints, real_interactions)

    assert pruned == {"A": ["real_A1"], "B": ["real_B1"]}

def test_prune_constraints_by_degree_all_pruned():
    engine = GroundingEngine()
    network = ReactionNetwork(
        proteins={"A": Protein(name="A"), "B": Protein(name="B")},
        rules=[Rule(name="r1", rule_type="binding", reactants=["A", "B"], products=["A_B"], rate=1.0)]
    )
    constraints = {"A": ["real_A2"], "B": ["real_B2"]}
    real_interactions = [("real_A1", "real_B1", "binding")]

    pruned = engine.prune_constraints_by_degree(network, constraints, real_interactions)

    assert pruned == {"A": ["real_A2"], "B": ["real_B2"]}

def test_match_abstract_to_real():
    engine = GroundingEngine()
    network = ReactionNetwork(
        proteins={"A": Protein(name="A"), "B": Protein(name="B")},
        rules=[Rule(name="r1", rule_type="binding", reactants=["A", "B"], products=["A_B"], rate=1.0)]
    )
    constraints = {"A": ["real_A1", "real_A2"], "B": ["real_B1"]}
    real_interactions = [("real_A1", "real_B1", "binding")]

    solutions = engine.match_abstract_to_real(network, constraints, real_interactions)

    assert len(solutions) == 1
    assert solutions[0] == {"A": "real_A1", "B": "real_B1"}

def test_match_abstract_to_real_no_solution():
    engine = GroundingEngine()
    network = ReactionNetwork(
        proteins={"A": Protein(name="A"), "B": Protein(name="B")},
        rules=[Rule(name="r1", rule_type="binding", reactants=["A", "B"], products=["A_B"], rate=1.0)]
    )
    constraints = {"A": ["real_A2"], "B": ["real_B2"]}
    real_interactions = [("real_A1", "real_B1", "binding")]

    solutions = engine.match_abstract_to_real(network, constraints, real_interactions)

    assert len(solutions) == 0

def test_mapping_respects_edges():
    engine = GroundingEngine()
    mapping = {"A": "real_A", "B": "real_B"}
    abstract_edges = {("A", "B", "binding")}
    real_edges = {("real_A", "real_B", "binding")}
    assert engine._mapping_respects_edges(mapping, abstract_edges, real_edges)

def test_mapping_respects_edges_false():
    engine = GroundingEngine()
    mapping = {"A": "real_A", "B": "real_B"}
    abstract_edges = {("A", "B", "binding")}
    real_edges = {("real_A", "real_C", "binding")}
    assert not engine._mapping_respects_edges(mapping, abstract_edges, real_edges)

def test_mapping_respects_edges_missing_mapping():
    engine = GroundingEngine()
    mapping = {"A": "real_A"}
    abstract_edges = {("A", "B", "binding")}
    real_edges = {("real_A", "real_B", "binding")}
    assert not engine._mapping_respects_edges(mapping, abstract_edges, real_edges)

def test_score_mappings():
    engine = GroundingEngine()
    mappings = [{"A": "real_A1", "B": "real_B1"}, {"A": "real_A2", "B": "real_B2"}]
    confidence = {("A", "real_A1"): 0.9, ("B", "real_B1"): 0.8, ("A", "real_A2"): 0.5, ("B", "real_B2"): 0.4}

    result = engine.score_mappings(mappings, confidence)

    assert result.mapping == {"A": "real_A1", "B": "real_B1"}
    assert result.score == pytest.approx(0.85)
    assert result.candidates_considered == 2

def test_score_mappings_empty():
    engine = GroundingEngine()
    result = engine.score_mappings([], {})

    assert result.mapping == {}
    assert result.score == 0.0
    assert result.candidates_considered == 0
