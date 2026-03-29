"""End-to-end pipeline: pathway query → INDRA baseline → evolution → comparison.

Usage from CLI:
  python -m modern_biojazz.e2e --seeds IL6,STAT3,TGFB1,SMAD3

Usage from Python:
  from modern_biojazz.e2e_pipeline import run_e2e
  result = run_e2e(seed_genes=["IL6", "STAT3", "TGFB1", "SMAD3"])
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bngl_converter import bngl_to_reaction_network
from .evolution import DeterministicProposer, RandomProposer, EvolutionConfig, EvolutionResult, LLMEvolutionEngine
from .grounding import GroundingEngine
from .indra_assembly import (
    AssemblyResult,
    INDRAAssembler,
    load_assembly_snapshot,
    load_bngl_file,
    save_assembly_snapshot,
)
from .mutation import GraphMutator
from .pathway_discovery import (
    OmniPathDiscovery,
    PathwayDiscoveryResult,
    load_discovery_snapshot,
    save_discovery_snapshot,
)
from .pipeline import ModernBioJazzPipeline, PipelineConfig
from .rate_optimizer import optimize_rates, DEConfig
from .simulation import FitnessEvaluator, LocalCatalystEngine, SimulationBackend
from .site_graph import ReactionNetwork


@dataclass
class E2EConfig:
    """Configuration for the full end-to-end pipeline."""

    # Discovery
    seed_genes: List[str] = field(default_factory=lambda: ["IL6", "STAT3", "TGFB1", "SMAD3"])
    expand_neighborhood: bool = True
    discovery_snapshot: Optional[str] = None  # Load from file instead of live query
    save_discovery_to: Optional[str] = None

    # Assembly
    assembly_snapshot: Optional[str] = None  # Load from file instead of live query
    bngl_file: Optional[str] = None  # Load BNGL directly, skip INDRA
    save_assembly_to: Optional[str] = None

    # Evolution
    evolution: EvolutionConfig = field(default_factory=lambda: EvolutionConfig(
        population_size=20,
        generations=10,
        mutations_per_candidate=3,
    ))
    fitness_target: float = 1.0
    random_seed: int = 42

    # Simulation
    sim_t_end: float = 30.0
    sim_dt: float = 0.5
    output_species: Optional[str] = None  # Which species to track for fitness. Auto-detected if None.

    # Grounding
    do_grounding: bool = True

    # Rate optimization (post-evolution)
    optimize_rates: bool = False
    rate_opt_max_eval: int = 500
    rate_opt_pop_size: Optional[int] = None  # Default: 10 * n_rates


@dataclass
class E2EResult:
    """Result of a full end-to-end run."""

    discovery: PathwayDiscoveryResult
    assembly: AssemblyResult
    baseline_network: ReactionNetwork
    baseline_score: float
    evolved_network: ReactionNetwork
    evolved_score: float
    evolution: EvolutionResult
    optimized_network: Optional[ReactionNetwork] = None
    optimized_score: Optional[float] = None
    improvement: float = 0.0  # best - baseline
    grounding: Any = None


def run_e2e(
    config: Optional[E2EConfig] = None,
    simulation_backend: Optional[SimulationBackend] = None,
) -> E2EResult:
    """Run the full end-to-end pipeline.

    Steps:
      1. Discover pathway species (OmniPath or snapshot)
      2. Assemble INDRA baseline (live, snapshot, or raw BNGL file)
      3. Parse BNGL → ReactionNetwork (baseline)
      4. Score baseline
      5. Build grounding payload from discovery data
      6. Evolve from baseline seed
      7. Score evolved network
      8. Compare
    """
    cfg = config or E2EConfig()
    rng = random.Random(cfg.random_seed)
    backend = simulation_backend or LocalCatalystEngine()
    evaluator = FitnessEvaluator(target_output=cfg.fitness_target)

    # ── Step 1: Discovery ────────────────────────────────────────────
    if cfg.discovery_snapshot:
        discovery = load_discovery_snapshot(cfg.discovery_snapshot)
    else:
        try:
            disc = OmniPathDiscovery()
            discovery = disc.discover(cfg.seed_genes, expand_neighborhood=cfg.expand_neighborhood)
        except Exception as exc:
            # Offline fallback: use seed genes directly.
            discovery = PathwayDiscoveryResult(
                seed_genes=cfg.seed_genes,
                species=list(cfg.seed_genes),
                interactions=[],
                source=f"fallback::{exc}",
            )

    if cfg.save_discovery_to:
        save_discovery_snapshot(discovery, cfg.save_discovery_to)

    # ── Step 2: Assembly ─────────────────────────────────────────────
    if cfg.bngl_file:
        assembly = load_bngl_file(cfg.bngl_file)
        assembly.species = discovery.species
    elif cfg.assembly_snapshot:
        assembly = load_assembly_snapshot(cfg.assembly_snapshot)
    else:
        try:
            assembler = INDRAAssembler()
            assembly = assembler.assemble(discovery.species)
            if not assembly.statements:
                assembly = _fallback_assembly(
                    discovery.species, "INDRA returned no statements"
                )
        except Exception as exc:
            # Offline fallback: generate a minimal BNGL from species names.
            assembly = _fallback_assembly(discovery.species, str(exc))

    if cfg.save_assembly_to:
        save_assembly_snapshot(assembly, cfg.save_assembly_to)

    # ── Step 3: Parse BNGL → ReactionNetwork ─────────────────────────
    baseline_network = bngl_to_reaction_network(assembly.bngl_text)
    baseline_network.metadata["pathway"] = "auto"
    baseline_network.metadata["source"] = assembly.source

    # Set output_species — user-specified, or auto-detect from BNGL observables,
    # or pick the first state-qualified species with nontrivial dynamics.
    if cfg.output_species:
        baseline_network.metadata["output_species"] = cfg.output_species
    elif "output_species" not in baseline_network.metadata:
        # Pick a phosphorylated or activated species as the output target.
        candidates = [name for name in baseline_network.proteins if "__" in name and ("_p" in name or "_active" in name or "_high" in name)]
        if candidates:
            baseline_network.metadata["output_species"] = sorted(candidates)[0]

    # ── Step 4: Score baseline ───────────────────────────────────────
    baseline_score = evaluator.score(
        backend=backend,
        network=baseline_network,
        t_end=cfg.sim_t_end,
        dt=cfg.sim_dt,
    )

    # ── Step 5: Build grounding payload ──────────────────────────────
    grounding_payload = _build_grounding_from_discovery(discovery, baseline_network)

    # ── Step 6: Evolve ───────────────────────────────────────────────
    engine = LLMEvolutionEngine(
        simulation_backend=backend,
        fitness_evaluator=evaluator,
        proposer=RandomProposer(random.Random(cfg.random_seed)),
        mutator=GraphMutator(rng),
        rng=rng,
    )
    pipeline = ModernBioJazzPipeline(engine, GroundingEngine())

    pipeline_result = pipeline.run(
        seed_network=baseline_network,
        config=PipelineConfig(
            evolution=cfg.evolution,
            do_grounding=cfg.do_grounding,
        ),
        grounding_payload=grounding_payload if cfg.do_grounding else None,
    )

    evolved_network = pipeline_result.evolution.best_network
    evolved_score = pipeline_result.evolution.best_score

    # ── Step 7: Rate optimization (optional) ─────────────────────────
    optimized_network = None
    optimized_score = None

    if cfg.optimize_rates and evolved_network.rules:
        def _objective(candidate: ReactionNetwork) -> float:
            return evaluator.score(
                backend=backend,
                network=candidate,
                t_end=cfg.sim_t_end,
                dt=cfg.sim_dt,
            )

        de_result = optimize_rates(
            network=evolved_network,
            backend=backend,
            objective=_objective,
            config=DEConfig(
                max_eval=cfg.rate_opt_max_eval,
                pop_size=cfg.rate_opt_pop_size,
                seed=cfg.random_seed + 1000,
            ),
        )
        optimized_network = de_result.best_network
        optimized_score = de_result.best_score

    best_score = optimized_score if optimized_score is not None else evolved_score

    return E2EResult(
        discovery=discovery,
        assembly=assembly,
        baseline_network=baseline_network,
        baseline_score=baseline_score,
        evolved_network=evolved_network,
        evolved_score=evolved_score,
        evolution=pipeline_result.evolution,
        optimized_network=optimized_network,
        optimized_score=optimized_score,
        improvement=best_score - baseline_score,
        grounding=pipeline_result.grounding,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _build_grounding_from_discovery(
    discovery: PathwayDiscoveryResult,
    baseline_network: ReactionNetwork,
) -> Dict[str, Any]:
    """Build grounding payload from discovery interactions + baseline protein types."""
    abstract_types: Dict[str, str] = {}
    for pname in baseline_network.proteins:
        has_mod = any(s.site_type == "modification" for s in baseline_network.proteins[pname].sites)
        has_bind = any(s.site_type == "binding" for s in baseline_network.proteins[pname].sites)
        if has_mod:
            abstract_types[pname] = "kinase_substrate"
        elif has_bind:
            abstract_types[pname] = "receptor_adaptor"
        else:
            abstract_types[pname] = "signaling"

    real_nodes = [{"name": sp, "type": "signaling"} for sp in discovery.species]

    real_interactions = []
    for row in discovery.interactions:
        src = row.get("source_genesymbol", "")
        dst = row.get("target_genesymbol", "")
        is_stim = row.get("is_stimulation", 0)
        is_inhib = row.get("is_inhibition", 0)
        if is_inhib:
            edge_type = "inhibition"
        elif is_stim:
            edge_type = "phosphorylation"
        else:
            edge_type = "binding"
        if src and dst:
            real_interactions.append((src, dst, edge_type))

    confidence_by_pair: Dict[str, float] = {}
    for abstract in abstract_types:
        for node in real_nodes:
            if node["name"] == abstract:
                confidence_by_pair[f"{abstract}->{node['name']}"] = 0.99
            elif node["name"].startswith(abstract):
                confidence_by_pair[f"{abstract}->{node['name']}"] = 0.8

    return {
        "abstract_types": abstract_types,
        "real_nodes": real_nodes,
        "real_interactions": real_interactions,
        "confidence_by_pair": confidence_by_pair,
    }


def _fallback_assembly(species: List[str], reason: str) -> AssemblyResult:
    """Generate a minimal BNGL model from species names when INDRA is unavailable."""
    lines = ["begin model", "", "begin parameters", "  kf 0.01", "end parameters", ""]
    lines.append("begin molecule types")
    for sp in species:
        lines.append(f"  {sp}()")
    lines.append("end molecule types")
    lines.append("")
    lines.append("begin seed species")
    for sp in species:
        lines.append(f"  {sp}() 1.0")
    lines.append("end seed species")
    lines.append("")
    lines.append("begin reaction rules")
    lines.append("end reaction rules")
    lines.append("")
    lines.append("begin observables")
    for sp in species:
        lines.append(f"  Molecules {sp}_obs {sp}()")
    lines.append("end observables")
    lines.append("")
    lines.append("end model")

    return AssemblyResult(
        species=species,
        statements=[],
        bngl_text="\n".join(lines),
        source=f"fallback::{reason}",
    )


def print_e2e_summary(result: E2EResult) -> None:
    """Print a human-readable summary of the E2E run."""
    print(f"Discovery: {len(result.discovery.species)} species from {result.discovery.source}")
    print(f"Assembly:  {len(result.assembly.statements)} statements from {result.assembly.source}")
    print(f"Baseline:  {len(result.baseline_network.proteins)} proteins, "
          f"{len(result.baseline_network.rules)} rules, score={result.baseline_score:.4f}")
    print(f"Evolved:   {len(result.evolved_network.proteins)} proteins, "
          f"{len(result.evolved_network.rules)} rules, score={result.evolved_score:.4f}")
    if result.optimized_network is not None and result.optimized_score is not None:
        print(f"Optimized: {len(result.optimized_network.rules)} rules, "
              f"score={result.optimized_score:.4f} "
              f"(+{result.optimized_score - result.evolved_score:.4f} from rate tuning)")
    sign = "+" if result.improvement >= 0 else ""
    print(f"Delta:     {sign}{result.improvement:.4f}")
    if result.grounding:
        print(f"Grounding: {result.grounding.candidates_considered} candidates, "
              f"best score={result.grounding.score:.4f}")
        if result.grounding.mapping:
            for abstract, real in sorted(result.grounding.mapping.items()):
                print(f"  {abstract} → {real}")


def print_evolution_summary(result: E2EResult) -> None:
    print("Evolution generation details:")
    for gen in result.evolution.generation_summary:
        print(
            f"gen {gen.generation}: best={gen.best_score:.4f}, "
            f"proteins={gen.best_n_proteins}, rules={gen.best_n_rules}, "
            f"unique_population={gen.unique_population}, "
            f"top_scores={gen.top_scores}"
        )
