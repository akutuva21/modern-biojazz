from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .evolution import LLMEvolutionEngine, EvolutionConfig, EvolutionResult
from .grounding import GroundingEngine, GroundingResult
from .site_graph import ReactionNetwork


@dataclass
class PipelineConfig:
    evolution: EvolutionConfig
    do_grounding: bool = True


@dataclass
class PipelineResult:
    evolution: EvolutionResult
    grounding: GroundingResult | None


class ModernBioJazzPipeline:
    def __init__(self, evolution_engine: LLMEvolutionEngine, grounding_engine: GroundingEngine) -> None:
        self.evolution_engine = evolution_engine
        self.grounding_engine = grounding_engine

    def run(
        self,
        seed_network: ReactionNetwork,
        config: PipelineConfig,
        grounding_payload: Dict[str, Any] | None = None,
    ) -> PipelineResult:
        if config.do_grounding and grounding_payload is not None:
            allowed = set(grounding_payload.get("abstract_types", {}).keys())
            self.evolution_engine.candidate_filter = self._grounding_constraint_filter(allowed)

        evolution = self.evolution_engine.run(seed_network, config.evolution)
        grounding_result: GroundingResult | None = None

        if config.do_grounding and grounding_payload is not None:
            constraints = self.grounding_engine.build_constraint_matrix(
                grounding_payload["abstract_types"],
                grounding_payload["real_nodes"],
            )
            mappings = self.grounding_engine.match_abstract_to_real(
                evolution.best_network,
                constraints,
                grounding_payload["real_interactions"],
            )
            grounding_result = self.grounding_engine.score_mappings(
                mappings,
                grounding_payload["confidence_by_pair"],
            )

        return PipelineResult(evolution=evolution, grounding=grounding_result)

    def _grounding_constraint_filter(self, allowed_symbols: set[str]):
        def _is_allowed_token(token: str) -> bool:
            if token in allowed_symbols:
                return True
            if token.endswith("_P") and token[:-2] in allowed_symbols:
                return True
            if token.endswith("_inh") and token[:-4] in allowed_symbols:
                return True
            if ":" in token:
                parts = token.split(":")
                return all(part in allowed_symbols for part in parts)
            return False

        def _filter(network: ReactionNetwork) -> bool:
            for rule in network.rules:
                for token in [*rule.reactants, *rule.products]:
                    if not _is_allowed_token(token):
                        return False
            return True

        return _filter
