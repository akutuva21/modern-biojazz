# BioJazz to modern-biojazz Architecture Comparison

| BioJazz (Perl) | modern-biojazz (Python) |
|---|---|
| `Genome.pm` / `Gene.pm` / `ProtoDomain.pm` - binary bit-field encoding | `site_graph.py` - direct `ReactionNetwork` dataclasses |
| `BindingProfile.pm` - Hamming distance complementarity | `mutation.py` - `_binding_is_compatible` with `allowed_partners` |
| `Network.pm` - adjacency matrix from `create_canbindrules` | `mutation.py` - `action_library` with 9 graph operators |
| `GenAlg.pm` - Kimura selection GA | `evolution.py` - island-model GA with LLM proposer |
| `Scoring.pm` + `MatlabDriver.pm` - MATLAB ODE fitness | `simulation.py` - `scipy.integrate.solve_ivp(BDF)` + Catalyst.jl HTTP |
| `custom/Ultrasensitive.pm` - Hill coefficient scoring | `simulation.py` - `UltrasensitiveFitnessEvaluator` |
| (no equivalent) | `grounding.py` - OmniPath/INDRA biological grounding |
