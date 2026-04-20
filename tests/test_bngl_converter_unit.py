from __future__ import annotations

from pathlib import Path
from modern_biojazz.bngl_converter import (
    bngl_to_reaction_network,
    _parse_parameters,
    _parse_molecule_types,
    _parse_mol_pattern,
    _parse_seed_species,
    _parse_reaction_rules,
    _split_rhs_tokens,
    _state_qualified_name,
    _infer_rule_type,
    _reverse_rule_type,
    _add_partner,
    _extract_mol_names,
)
from modern_biojazz.site_graph import Protein, Site


def test_bngl_converter_parses_molecule_types(fixtures_dir: Path):
    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)

    assert "JAK1" in net.proteins
    assert "STAT3" in net.proteins
    assert "SOCS3" in net.proteins

    stat3 = net.proteins["STAT3"]
    site_names = [s.name for s in stat3.sites]
    assert "Y705" in site_names
    y705 = next(s for s in stat3.sites if s.name == "Y705")
    assert y705.site_type == "modification"
    assert "u" in y705.states and "p" in y705.states


def test_bngl_converter_parses_rules(fixtures_dir: Path):
    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)

    assert len(net.rules) >= 4  # 4 forward + 1 reverse from r3
    rule_types = {r.rule_type for r in net.rules}
    assert "phosphorylation" in rule_types
    assert "inhibition" in rule_types

    r3_names = [r.name for r in net.rules if r.name.startswith("r3")]
    assert len(r3_names) == 2  # r3 and r3_rev


def test_bngl_converter_reads_parameters(fixtures_dir: Path):
    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)

    r1 = next(r for r in net.rules if r.name == "r1")
    assert abs(r1.rate - 0.07) < 1e-6


def test_bngl_converter_skips_invalid_parameters():
    bngl_text = """
    begin parameters
    # This is a comment

    valid_param 1.5
    invalid_param not_a_number
    valid_param2 2.0
    bare_param
    end parameters
    """
    params = _parse_parameters(bngl_text)
    assert "valid_param" in params
    assert params["valid_param"] == 1.5
    assert "valid_param2" in params
    assert params["valid_param2"] == 2.0
    assert "invalid_param" not in params
    assert "bare_param" not in params


def test_bngl_converter_reads_seed_species(fixtures_dir: Path):
    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)

    ic = net.metadata.get("initial_concentrations", {})
    assert ic.get("JAK1", 0) == 1.0
    assert ic.get("STAT3__Y705_u", 0) == 1.0


def test_bngl_converter_roundtrip_to_simulation(fixtures_dir: Path):
    from modern_biojazz.simulation import LocalCatalystEngine, FitnessEvaluator, SimulationOptions

    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)
    engine = LocalCatalystEngine()
    result = engine.simulate(net, SimulationOptions(t_end=5.0, dt=1.0, solver="BDF"))
    assert len(result["trajectory"]) == 6
    score = FitnessEvaluator(target_output=1.0).score(simulation_result=result)
    assert score >= 0.0


def test_parse_molecule_types_edge_cases():
    text = """
    begin molecule types
    # A comment

    A
    B()
    C(s1~u~p,)
    end molecule types
    """
    mol_types = _parse_molecule_types(text)
    assert "A" in mol_types
    assert mol_types["A"] == []
    assert "B" in mol_types
    assert mol_types["B"] == []
    assert "C" in mol_types
    assert mol_types["C"][0][0] == "s1"
    assert mol_types["C"][0][1] == ["u", "p"]


def test_parse_mol_pattern_edge_cases():
    assert _parse_mol_pattern("") == ("", [])
    assert _parse_mol_pattern("A") == ("A", [])
    assert _parse_mol_pattern("A  ") == ("A", [])
    assert _parse_mol_pattern("A()") == ("A", [])


def test_parse_seed_species_edge_cases():
    text = """
    begin seed species
    # A comment

    A 1.0
    B invalid_float
    end seed species
    """
    seeds = _parse_seed_species(text)
    assert "A" in seeds
    assert seeds["A"] == 1.0
    assert "B" not in seeds


def test_parse_reaction_rules_edge_cases():
    text = """
    begin reaction rules
    # A comment

    A -> B
    A + B
    C -> D k1 k2
    E -> F k1,k2
    r_label: G -> H k1
    end reaction rules
    """
    params = {"k1": 1.0, "k2": 2.0}
    rules = _parse_reaction_rules(text, params)

    names = [r.name for r in rules]
    assert len(rules) == 4
    assert "rule_1" in names  # A -> B
    assert "rule_3" in names  # C -> D
    assert "rule_4" in names  # E -> F
    assert "r_label" in names # G -> H


def test_parse_reaction_rules_empty_rhs():
    text = """
    begin reaction rules
    A ->
    end reaction rules
    """
    rules = _parse_reaction_rules(text, {})
    assert len(rules) == 0


def test_split_rhs_tokens_edge_cases():
    # Only rates
    assert _split_rhs_tokens(["k1", "k2"], {"k1": 1.0, "k2": 2.0}) == (["k1"], ["k2"])
    # Product and rate
    assert _split_rhs_tokens(["A", "+", "B", "k1"], {"k1": 1.0}) == (["A", "+", "B"], ["k1"])
    # No rates
    assert _split_rhs_tokens(["A", "+", "B"], {"k1": 1.0}) == (["A", "+", "B"], [])

    # Empty product check branch
    # If no product tokens, fallback is first element is product
    assert _split_rhs_tokens(["k1", "k2"], {"k2": 2.0}) == (["k1"], ["k2"])


def test_state_qualified_name_edge_cases():
    assert _state_qualified_name("A") == "A"
    assert _state_qualified_name("A()") == "A"
    assert _state_qualified_name("A(s1~u,)") == "A__s1_u"
    assert _state_qualified_name("A(s1!1)") == "A"


def test_extract_mol_names_empty_tok():
    assert _extract_mol_names("A +  + B") == ["A", "B"]


def test_is_rate_token_valid_float():
    from modern_biojazz.bngl_converter import _is_rate_token
    assert _is_rate_token("1.23", {}) is True


def test_resolve_rates_edge_cases():
    from modern_biojazz.bngl_converter import _resolve_rates
    assert _resolve_rates(["k1, k2"], {"k1": 1.0}) == [1.0, 0.01]
    assert _resolve_rates(["invalid"], {}) == [0.01]
    assert _resolve_rates([",k1,"], {"k1": 1.0}) == [1.0]


def test_infer_rule_type_edge_cases():
    assert _infer_rule_type("A", "B", ["A"], ["B"]) == "reaction"
    assert _infer_rule_type("A~inactive", "A~active", ["A"], ["A"]) == "activation"
    assert _infer_rule_type("A!1.B!1", "A + B", ["A:B"], ["A", "B"]) == "unbinding"
    assert _infer_rule_type("A + B", "C + D", ["A", "B"], ["C", "D"]) == "catalysis"


def test_reverse_rule_type():
    assert _reverse_rule_type("binding") == "unbinding"
    assert _reverse_rule_type("phosphorylation") == "dephosphorylation"
    assert _reverse_rule_type("unknown") == "reaction"


def test_add_partner_edge_cases():
    proteins = {"A": Protein("A", [Site("s1", "binding", [], [])])}

    _add_partner(proteins, "Unknown", "B")
    assert "Unknown" not in proteins

    _add_partner(proteins, "A", "B")
    assert proteins["A"].sites[0].allowed_partners == ["B"]

    # Adding again shouldn't duplicate
    _add_partner(proteins, "A", "B")
    assert proteins["A"].sites[0].allowed_partners == ["B"]
