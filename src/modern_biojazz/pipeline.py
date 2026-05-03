import re
from dataclasses import dataclass
from typing import Any

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
        grounding_payload: dict[str, Any] | None = None,
    ) -> PipelineResult:
        if config.do_grounding and grounding_payload is not None:
            # Build the allowed set from both grounding abstract types AND the seed
            # network's existing proteins so that evolution doesn't reject the starting
            # state or its natural derivatives (phospho, inhibited, complex, duplicated).
            abstract_keys = set(grounding_payload["abstract"]["nodes"])
            seed_keys = set(seed_network.proteins)
            allowed = abstract_keys | seed_keys
            self.evolution_engine.candidate_filter = self._grounding_constraint_filter(allowed)

        evolution = self.evolution_engine.run(seed_network, config.evolution)
        grounding_result: GroundingResult | None = None

        if config.do_grounding and grounding_payload is not None:
            constraints = self.grounding_engine.build_constraint_matrix(
                grounding_payload["abstract"]["types"],
                grounding_payload["real"]["nodes"],
            )
            mappings = self.grounding_engine.match_abstract_to_real(
                evolution.best_network,
                constraints,
                grounding_payload["real"]["edges"],
            )
            grounding_result = self.grounding_engine.score_mappings(
                mappings,
                grounding_payload.get("confidence", {}),
            )

        return PipelineResult(evolution=evolution, grounding=grounding_result)

    def _grounding_constraint_filter(self, allowed_symbols: set[str]):
        # Regex to strip trailing _P or _inh iteratively
        suffix_pattern = re.compile(r"(_P|_inh)+$")

        _base_cache: dict[str, str] = {}

        def _base_protein(token: str) -> str:
            """Strip derived suffixes to find the base protein name.
            Handles compound suffixes like STAT3_dup42_P."""
            if token in _base_cache:
                return _base_cache[token]

            orig_token = token
            if "_dup" in token:
                token = token[: token.index("_dup")]
            else:
                token = suffix_pattern.sub("", token)

            _base_cache[orig_token] = token
            return token

        _allowed_cache: dict[str, bool] = {}

        def _is_allowed_token(token: str) -> bool:
            if token in _allowed_cache:
                return _allowed_cache[token]

            result = False
            if token in allowed_symbols:
                result = True
            else:
                base = _base_protein(token)
                if base in allowed_symbols:
                    result = True
                elif ":" in token:
                    parts = token.split(":")
                    result = all(_base_protein(part) in allowed_symbols for part in parts)

            _allowed_cache[token] = result
            return result

        def _filter(network: ReactionNetwork) -> bool:
            for pname in network.proteins:
                if not _is_allowed_token(pname):
                    return False
            for rule in network.rules:
                for token in itertools.chain(rule.reactants, rule.products):
                    if not _is_allowed_token(token):
                        return False
            return True

        return _filter
