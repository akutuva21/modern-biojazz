from __future__ import annotations

import json
from pathlib import Path

from modern_biojazz.bngl_converter import bngl_to_reaction_network
from modern_biojazz.e2e_pipeline import run_e2e, E2EConfig, _fallback_assembly
from modern_biojazz.indra_assembly import load_bngl_file
from modern_biojazz.pathway_discovery import (
    PathwayDiscoveryResult,
    load_discovery_snapshot,
    save_discovery_snapshot,
)


# ── BNGL converter ───────────────────────────────────────────────────


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

    # Check that the reversible rule produced both forward and reverse.
    r3_names = [r.name for r in net.rules if r.name.startswith("r3")]
    assert len(r3_names) == 2  # r3 and r3_rev


def test_bngl_converter_reads_parameters(fixtures_dir: Path):
    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)

    r1 = next(r for r in net.rules if r.name == "r1")
    assert abs(r1.rate - 0.07) < 1e-6


def test_bngl_converter_reads_seed_species(fixtures_dir: Path):
    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)

    ic = net.metadata.get("initial_concentrations", {})
    assert ic.get("JAK1", 0) == 1.0
    # STAT3 is represented in state-qualified form now.
    assert ic.get("STAT3__Y705_u", 0) == 1.0


def test_bngl_converter_roundtrip_to_simulation(fixtures_dir: Path):
    """Parsed BNGL network can be simulated without errors."""
    from modern_biojazz.simulation import LocalCatalystEngine, FitnessEvaluator

    bngl = (fixtures_dir / "sample_indra.bngl").read_text()
    net = bngl_to_reaction_network(bngl)
    engine = LocalCatalystEngine()
    result = engine.simulate(net, t_end=5.0, dt=1.0, solver="BDF")
    assert len(result["trajectory"]) == 6
    score = FitnessEvaluator(target_output=1.0).score(simulation_result=result)
    assert score >= 0.0


# ── Pathway discovery snapshots ──────────────────────────────────────


def test_discovery_snapshot_roundtrip(fixtures_dir: Path, tmp_path: Path):
    original = load_discovery_snapshot(str(fixtures_dir / "discovery_snapshot.json"))
    assert len(original.species) == 5
    assert original.source == "snapshot"

    out_path = str(tmp_path / "disc_out.json")
    save_discovery_snapshot(original, out_path)
    reloaded = load_discovery_snapshot(out_path)
    assert reloaded.species == original.species
    assert reloaded.seed_genes == original.seed_genes


# ── BNGL file loading ────────────────────────────────────────────────


def test_load_bngl_file(fixtures_dir: Path):
    result = load_bngl_file(str(fixtures_dir / "sample_indra.bngl"))
    assert result.source == "file"
    assert "begin model" in result.bngl_text


# ── Fallback assembly ────────────────────────────────────────────────


def test_fallback_assembly_generates_valid_bngl():
    result = _fallback_assembly(["A", "B", "C"], "test offline")
    assert "begin model" in result.bngl_text
    assert "A()" in result.bngl_text
    net = bngl_to_reaction_network(result.bngl_text)
    assert "A" in net.proteins
    assert "B" in net.proteins
    assert "C" in net.proteins


# ── Full E2E pipeline (offline, using snapshots) ─────────────────────


def test_e2e_pipeline_from_snapshots(fixtures_dir: Path):
    """Run the full pipeline using offline snapshot + BNGL file — no network needed."""
    config = E2EConfig(
        seed_genes=["IL6", "STAT3", "SOCS3"],
        discovery_snapshot=str(fixtures_dir / "discovery_snapshot.json"),
        bngl_file=str(fixtures_dir / "sample_indra.bngl"),
        evolution=E2EConfig().evolution,
        do_grounding=True,
        random_seed=99,
    )
    # Smaller evolution for test speed.
    config.evolution.population_size = 4
    config.evolution.generations = 2
    config.evolution.mutations_per_candidate = 1

    result = run_e2e(config)

    assert result.baseline_score >= 0.0
    assert result.evolved_score >= 0.0
    assert len(result.baseline_network.proteins) >= 3
    assert len(result.evolved_network.proteins) >= 1
    assert result.discovery.source == "snapshot"


def test_e2e_pipeline_fallback_when_offline():
    """Run with no snapshots and no network — should still produce results via fallbacks."""
    config = E2EConfig(
        seed_genes=["X", "Y"],
        evolution=E2EConfig().evolution,
        do_grounding=False,
        random_seed=123,
    )
    config.evolution.population_size = 4
    config.evolution.generations = 2
    config.evolution.mutations_per_candidate = 1

    result = run_e2e(config)

    assert "fallback" in result.discovery.source
    assert "fallback" in result.assembly.source
    assert result.baseline_score >= 0.0
    assert result.evolved_score >= 0.0


def test_evolution_generation_summary_includes_top_scores():
    config = E2EConfig(
        seed_genes=["IL6", "STAT3", "SOCS3"],
        discovery_snapshot="tests/fixtures/discovery_snapshot.json",
        bngl_file="tests/fixtures/sample_indra.bngl",
        evolution=E2EConfig().evolution,
        do_grounding=False,
        random_seed=42,
    )
    config.evolution.population_size = 4
    config.evolution.generations = 2
    config.evolution.mutations_per_candidate = 1

    result = run_e2e(config)

    assert len(result.evolution.history) == 3
    assert len(result.evolution.generation_summary) == 3
    assert result.evolution.generation_summary[0].best_score == result.evolution.history[0]
    assert isinstance(result.evolution.generation_summary[0].top_scores, list)
    assert isinstance(result.evolution.generation_summary[0].unique_population, int)
    assert result.evolution.generation_summary[0].unique_population >= 1

