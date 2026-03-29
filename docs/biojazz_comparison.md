# Architectural Comparison: BioJazz → modern-biojazz

## Lineage

This project is a ground-up reimplementation of the ideas in
[BioJazz](https://github.com/OSS-Lab/biojazz) (Feng, Ollivier, Swain & Soyer,
*Nucleic Acids Research* 2015). No code is shared. BioJazz is Perl + MATLAB;
modern-biojazz is Python + scipy (with a Julia/Catalyst.jl HTTP adapter).

## Module Mapping

| BioJazz (Perl) | modern-biojazz (Python) | Notes |
|---|---|---|
| `Genome.pm` / `Gene.pm` / `ProtoDomain.pm` — binary bit-field genome | `site_graph.py` — `ReactionNetwork` / `Protein` / `Site` / `Rule` | BioJazz encoded networks as binary strings. modern-biojazz uses named proteins with typed sites. |
| `ProtoDomainInstance.pm` — `create_canbindrules()` via Hamming distance | `mutation.py` — `_binding_is_compatible()` with `allowed_partners` | BioJazz: complementarity = LR-flip + NOT + Hamming ≤ radius. modern-biojazz: explicit partner allowlists. |
| `BindingProfile.pm` — `mismatch()` with polarity/conformation masks | `mutation.py` — `_ensure_binding_sites()` | BioJazz masks enabled allosteric effects. modern-biojazz does not yet have allosteric encoding. |
| `Network.pm` — adjacency matrix + msite state pruning | `mutation.py` — `action_library()` with 11 graph operators | BioJazz built static adjacency matrices. modern-biojazz mutates graphs directly. |
| `GenAlg.pm` — Kimura / population-based GA | `evolution.py` — island-model GA + LLM proposals + CEGIS feedback | BioJazz: random bit mutations. modern-biojazz: named operators proposed by LLM. |
| `Scoring.pm` + `MatlabDriver.pm` — MATLAB ODE via Expect | `simulation.py` — `scipy.integrate.solve_ivp(BDF)` + Catalyst HTTP | BioJazz spawned MATLAB. modern-biojazz uses scipy or Julia HTTP. |
| `custom/Ultrasensitive.pm` — Hill coefficient scoring | `simulation.py` — `UltrasensitiveFitnessEvaluator` | Both estimate Hill coefficients; BioJazz used time-domain stimulus, modern-biojazz uses steady-state dose-response. |
| `custom/Oscillator.pm` | *(not yet implemented)* | |
| `custom/Adaptive.pm` | *(not yet implemented)* | |
| `custom/Multistable.pm` | *(not yet implemented)* | |
| ANC (Allosteric Network Compiler) | *(not needed — rules are native)* | |
| Facile (ANC → MATLAB ODE) | *(not needed — scipy/Catalyst reads network directly)* | |
| *(no equivalent)* | `grounding.py` — subgraph matcher via OmniPath/INDRA | Novel: maps abstract topologies to real proteins. |
| *(no equivalent)* | `pipeline.py` — grounding constraint filter during evolution | Novel: biological constraints restrict search space in-loop. |

## Mutation Operators

| BioJazz | modern-biojazz |
|---|---|
| Point mutation (bit flip) | `modify_rate` (log-uniform jitter, clamped) |
| Gene duplication | `duplicate_protein` (with rule rewiring) |
| Gene deletion | `remove_protein` (cleans derived tokens) |
| Domain duplication | `add_site` |
| Domain deletion | `remove_site` |
| Recombination | *(not yet implemented)* |
| *(implicit via bits)* | `add_binding` / `add_unbinding` |
| *(implicit via bits)* | `add_phosphorylation` / `add_dephosphorylation` |
| *(implicit via bits)* | `add_inhibition` |
| *(no equivalent)* | `remove_rule` |

## Key Architectural Differences

**Encoding**: BioJazz used indirect encoding (binary genome → compiled network) enabling neutral drift. modern-biojazz uses direct encoding — every mutation immediately changes the network.

**Binding specificity**: BioJazz used Hamming-distance complementarity with configurable radius. modern-biojazz uses explicit `allowed_partners` lists — more interpretable but less emergent.

**Allosteric regulation**: BioJazz encoded allostery via domain-level conformation masks (XOR on binding profiles). modern-biojazz has `Site.states` but no allosteric coupling yet.

## What modern-biojazz Adds

1. **Biological grounding** via OmniPath/INDRA with confidence scoring
2. **In-loop grounding constraints** rejecting candidates outside the grounding vocabulary
3. **LLM-guided mutation proposals** instead of random bit operations
4. **CEGIS-style feedback** from simulation failures to the proposer
5. **Typed site model** with states, types, and partner constraints as first-class data
6. **HTTP service architecture** for swappable simulation backends
