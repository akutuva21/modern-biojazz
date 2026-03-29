# Modern BioJazz

This package provides a practical modernization scaffold for BioJazz:

- direct site-graph network representation
- graph-level mutation operators
- Catalyst.jl simulation adapter interface
- simulation benchmarking and backend comparison helpers
- LLM-guided evolutionary search loop
- OpenAI-compatible LLM proposer adapter with safety filtering
- OmniPath/INDRA-style biological grounding layer
- snapshot-based grounding payload ingestion for reproducible experiments
- end-to-end pipeline orchestration and tests

## Lineage

This project is a ground-up reimplementation of the ideas in
[BioJazz](https://github.com/OSS-Lab/biojazz) (Feng et al., NAR 2015).
No code is shared. For a detailed architectural comparison, see
[docs/biojazz_comparison.md](docs/biojazz_comparison.md).

## Quick start

```powershell
cd modern_biojazz
python -m pip install -e .[dev]
pytest
```

## Run the pipeline

Local simulation + deterministic proposer:

```powershell
python -m modern_biojazz.cli --seed tests/fixtures/seed_network.json --grounding tests/fixtures/grounding_payload.json
```

HTTP simulation backend + OpenAI-compatible proposer:

```powershell
$env:OPENAI_API_KEY="your_key"
python -m modern_biojazz.cli --seed tests/fixtures/seed_network.json --grounding tests/fixtures/grounding_payload.json --sim-backend http --sim-base-url http://127.0.0.1:8000 --llm-provider openai_compatible --llm-base-url https://api.openai.com/v1 --llm-model gpt-4o-mini
```

## Benchmark helper

Use `modern_biojazz.benchmarking.compare_backends(...)` to compare a candidate backend (e.g., Catalyst HTTP service) against a baseline backend and report mean runtime and speedup.

## Catalyst service contract

The expected simulation service request/response schema is defined in:

- `docs/catalyst_service_openapi.yaml`

## Fitness options

- `FitnessEvaluator`: target end-point matching
- `UltrasensitiveFitnessEvaluator`: effective Hill-coefficient style steepness scoring over dose sweeps

## Notes

The package defaults to dependency-light implementations and injectable adapters so tests run without requiring Julia, OmniPath, or INDRA to be installed locally.
