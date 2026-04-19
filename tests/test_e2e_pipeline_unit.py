from __future__ import annotations

import json
from pathlib import Path

from modern_biojazz.bngl_converter import bngl_to_reaction_network
from unittest.mock import patch

from modern_biojazz.e2e_pipeline import run_e2e, E2EConfig, _fallback_assembly, _run_discovery
from modern_biojazz.indra_assembly import load_bngl_file
from modern_biojazz.pathway_discovery import (
    PathwayDiscoveryResult,
    load_discovery_snapshot,
    save_discovery_snapshot,
)


# ── Pathway discovery snapshots ──────────────────────────────────────


def test_run_discovery_offline_fallback():
    """Test that _run_discovery correctly falls back to seed genes when OmniPath raises an Exception."""
    config = E2EConfig(seed_genes=["A", "B", "C"])

    with patch("modern_biojazz.e2e_pipeline.OmniPathDiscovery.discover") as mock_discover:
        mock_discover.side_effect = Exception("offline test")

        result = _run_discovery(config)

        assert mock_discover.called
        assert result.seed_genes == ["A", "B", "C"]
        assert result.species == ["A", "B", "C"]
        assert result.interactions == []
        assert result.source == "fallback::offline test"


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


# ── Printing summaries ────────────────────────────────────────────────


def test_print_e2e_summary_basic(capsys):
    from unittest.mock import Mock
    from modern_biojazz.e2e_pipeline import print_e2e_summary

    mock_result = Mock()
    mock_result.config.seed_genes = ["A", "B"]
    mock_result.discovery.source = "test_disc"
    mock_result.discovery.species = ["A", "B"]
    mock_result.discovery.interactions = [{"source_genesymbol": "A", "target_genesymbol": "B"}]
    mock_result.assembly.source = "test_assem"

    print_e2e_summary(mock_result)
    captured = capsys.readouterr()

    assert "E2E Pipeline Summary" in captured.out
    assert "Seed genes:       ['A', 'B']" in captured.out
    assert "Discovery source: test_disc" in captured.out
    assert "Discovered ops:   2 species, 1 interactions" in captured.out
    assert "Assembly source:  test_assem" in captured.out
