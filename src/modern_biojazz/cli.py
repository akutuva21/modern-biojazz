from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

from .evolution import LLMEvolutionEngine, EvolutionConfig, DeterministicProposer, RandomProposer
from .grounding import GroundingEngine
from .llm_proposer import OpenAICompatibleProposer, SafeActionFilterProposer
from .mutation import GraphMutator
from .pipeline import ModernBioJazzPipeline, PipelineConfig
from .site_graph import ReactionNetwork
from .simulation import LocalCatalystEngine, FitnessEvaluator, CatalystHTTPClient


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--llm-provider", choices=["deterministic", "random", "openai_compatible"], default="random")
    parser.add_argument("--llm-base-url", help="Base URL for OpenAI-compatible API")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def get_simulation_backend(args: argparse.Namespace) -> Any:
    if args.sim_backend == "http":
        if not args.sim_base_url:
            raise ValueError("--sim-base-url is required when --sim-backend=http")
        return CatalystHTTPClient(base_url=args.sim_base_url)
    return LocalCatalystEngine()


def get_llm_proposer(args: argparse.Namespace) -> Any:
    if args.llm_provider == "openai_compatible":
        if not args.llm_base_url:
            raise ValueError("--llm-base-url is required when --llm-provider=openai_compatible")
        api_key = os.environ.get(args.llm_api_key_env, "")
        if not api_key:
            raise ValueError(f"Environment variable {args.llm_api_key_env} must be set for llm provider")
        return SafeActionFilterProposer(
            OpenAICompatibleProposer(
                base_url=args.llm_base_url,
                api_key=api_key,
                model=args.llm_model,
            )
        )
    if args.llm_provider == "random":
        return RandomProposer(random.Random(7))
    return DeterministicProposer()


def load_payload(filepath: str | None) -> dict[str, Any] | None:
    if not filepath:
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pipeline(args: argparse.Namespace) -> ModernBioJazzPipeline:
    engine = LLMEvolutionEngine(
        simulation_backend=get_simulation_backend(args),
        fitness_evaluator=FitnessEvaluator(target_output=1.0),
        proposer=get_llm_proposer(args),
        mutator=GraphMutator(random.Random(7)),
        rng=random.Random(7),
    )
    return ModernBioJazzPipeline(engine, GroundingEngine())


def execute_run(
    pipeline: ModernBioJazzPipeline,
    seed_network: ReactionNetwork,
    grounding_payload: dict[str, Any] | None,
    args: argparse.Namespace,
) -> Any:
    return pipeline.run(
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


def format_output(result: Any) -> dict[str, Any]:
    return {
        "best_score": result.evolution.best_score,
        "history": result.evolution.history,
        "best_network": result.evolution.best_network.to_dict(),
        "grounding": (
            None
            if result.grounding is None
            else {
                "mapping": result.grounding.mapping,
                "score": result.grounding.score,
                "candidates_considered": result.grounding.candidates_considered,
            }
        ),
    }


def main() -> None:
    args = parse_args()

    seed_payload = load_payload(args.seed)
    if seed_payload is None:
        raise ValueError("Seed payload is required")
    seed_network = ReactionNetwork.from_dict(seed_payload)

    grounding_payload = load_payload(args.grounding)

    pipeline = build_pipeline(args)
    result = execute_run(pipeline, seed_network, grounding_payload, args)

    print(json.dumps(format_output(result), indent=2))


if __name__ == "__main__":
    main()
