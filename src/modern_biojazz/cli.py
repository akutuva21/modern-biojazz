from __future__ import annotations

import argparse
import json
import os
import random

from .evolution import LLMEvolutionEngine, EvolutionConfig, DeterministicProposer
from .grounding import GroundingEngine
from .llm_proposer import OpenAICompatibleProposer, SafeActionFilterProposer
from .mutation import GraphMutator
from .pipeline import ModernBioJazzPipeline, PipelineConfig
from .site_graph import ReactionNetwork
from .simulation import LocalCatalystEngine, FitnessEvaluator, CatalystHTTPClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Modern BioJazz pipeline runner")
    parser.add_argument("--seed", required=True, help="Path to seed network JSON")
    parser.add_argument("--grounding", help="Path to grounding payload JSON")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--sim-t-end", type=float, default=20.0)
    parser.add_argument("--sim-dt", type=float, default=1.0)
    parser.add_argument("--sim-solver", default="Rodas5P")
    parser.add_argument("--sim-backend", choices=["local", "http"], default="local")
    parser.add_argument("--sim-base-url", help="Base URL of Catalyst simulation service")
    parser.add_argument("--llm-provider", choices=["deterministic", "openai_compatible"], default="deterministic")
    parser.add_argument("--llm-base-url", help="Base URL for OpenAI-compatible API")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    with open(args.seed, "r", encoding="utf-8") as f:
        seed_payload = json.load(f)
    seed_network = ReactionNetwork.from_dict(seed_payload)

    grounding_payload = None
    if args.grounding:
        with open(args.grounding, "r", encoding="utf-8") as f:
            grounding_payload = json.load(f)

    if args.sim_backend == "http":
        if not args.sim_base_url:
            raise ValueError("--sim-base-url is required when --sim-backend=http")
        simulation_backend = CatalystHTTPClient(base_url=args.sim_base_url)
    else:
        simulation_backend = LocalCatalystEngine()

    if args.llm_provider == "openai_compatible":
        if not args.llm_base_url:
            raise ValueError("--llm-base-url is required when --llm-provider=openai_compatible")
        api_key = os.environ.get(args.llm_api_key_env, "")
        if not api_key:
            raise ValueError(f"Environment variable {args.llm_api_key_env} must be set for llm provider")
        proposer = SafeActionFilterProposer(
            OpenAICompatibleProposer(
                base_url=args.llm_base_url,
                api_key=api_key,
                model=args.llm_model,
            )
        )
    else:
        proposer = DeterministicProposer()

    engine = LLMEvolutionEngine(
        simulation_backend=simulation_backend,
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=proposer,
        mutator=GraphMutator(random.Random(7)),
        rng=random.Random(7),
    )
    pipeline = ModernBioJazzPipeline(engine, GroundingEngine())

    result = pipeline.run(
        seed_network,
        PipelineConfig(
            evolution=EvolutionConfig(
                population_size=args.population,
                generations=args.generations,
                sim_t_end=args.sim_t_end,
                sim_dt=args.sim_dt,
                sim_solver=args.sim_solver,
            ),
            do_grounding=grounding_payload is not None,
        ),
        grounding_payload=grounding_payload,
    )

    output = {
        "best_score": result.evolution.best_score,
        "history": result.evolution.history,
        "best_network": result.evolution.best_network.to_dict(),
        "grounding": None
        if result.grounding is None
        else {
            "mapping": result.grounding.mapping,
            "score": result.grounding.score,
            "candidates_considered": result.grounding.candidates_considered,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
