# AGENTS.md

## Project

modern-biojazz: evolutionary design of biochemical signaling networks using
rule-based modeling. Reimplements BioJazz (Feng et al., NAR 2015) in Python
with scipy BDF solver, LLM-guided mutation proposals, OmniPath/INDRA biological
grounding, and differential evolution rate optimization.

## Architecture

```
pathway_discovery.py  → OmniPath species enumeration
indra_assembly.py     → INDRA statements → BNGL text
bngl_converter.py     → BNGL text → ReactionNetwork
site_graph.py         → ReactionNetwork / Protein / Site / Rule dataclasses
mutation.py           → 11 graph-level mutation operators
simulation.py         → BDF solver (scipy), HTTP client (Catalyst.jl), fitness evaluators
evolution.py          → Island-model GA with LLM proposer protocol + CEGIS feedback
grounding.py          → Backtracking subgraph matcher with edge-type normalization
pipeline.py           → Orchestrator: grounding constraints → evolution → grounding
rate_optimizer.py     → Differential evolution for rate constants (log10 space)
e2e_pipeline.py       → Full pipeline: discovery → assembly → baseline → evolve → optimize → compare
bngplayground_backend.py → MCP server adapter for BNG Playground (CVODE, BNGL parser)
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/test_*.py -v          # unit tests
pytest tests/e2e/ -v               # e2e tests (offline, uses mock HTTP servers)
pytest -v                          # everything
```

All tests run offline. No API keys or network access needed. The e2e pipeline
uses snapshot fixtures in `tests/fixtures/`.

## Key conventions

- `ReactionNetwork` is the central data structure. Always use `.copy()` before mutation.
- `SimulationBackend` is a Protocol. `LocalCatalystEngine` uses scipy BDF; `CatalystHTTPClient` posts to Julia; `BNGPlaygroundBackend` calls the MCP server.
- `FitnessScorer` is a Protocol with `score(simulation_result=..., backend=..., network=..., ...)`.
- Mutation operators in `GraphMutator` must create product species via `_add_species_if_missing`.
- The grounding constraint filter in `pipeline.py` uses `_base_protein()` to strip `_P`, `_inh`, `_dupNNNN` suffixes iteratively.
- Rate optimizer works in log10 space. Bounds default to [1e-6, 100].
- CEGIS feedback is sent via `proposer.record_feedback()` (optional method, checked with `hasattr`).

## File layout

```
src/modern_biojazz/     # Source modules
tests/                  # Unit tests (test_*.py)
tests/e2e/              # End-to-end tests
tests/fixtures/         # JSON/BNGL fixtures for offline testing
docs/                   # OpenAPI spec, architecture comparison
.github/workflows/      # CI: fast-tests, type-check, bngl-parity, e2e-pipeline, integration
```

## Dependencies

Runtime: `scipy>=1.10` (BDF solver). Dev: `pytest>=8.0`.
Optional: `indra` (live BNGL assembly), `omnipath` (live pathway queries), BNG Playground + Node.js (MCP backend).

## Common tasks

- **Add a mutation operator**: Add method to `GraphMutator`, add random wrapper in `action_library()`, update `test_mutation_unit.py` expected set.
- **Add a fitness evaluator**: Implement `FitnessScorer` protocol in `simulation.py`, add to `__init__.py` exports.
- **Add a simulation backend**: Implement `SimulationBackend` protocol. Return `{"solver": str, "trajectory": [{t, output, species}], "stats": {n_rules, n_species, stiff}}`.
- **Run the full pipeline offline**: `python -c "from modern_biojazz.e2e_pipeline import run_e2e, E2EConfig, print_e2e_summary; r = run_e2e(E2EConfig(discovery_snapshot='tests/fixtures/discovery_snapshot.json', bngl_file='tests/fixtures/sample_indra.bngl')); print_e2e_summary(r)"`
