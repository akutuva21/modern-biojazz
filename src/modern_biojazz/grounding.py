from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from .site_graph import ReactionNetwork


@dataclass
class GroundingResult:
    mapping: Dict[str, str]
    score: float
    candidates_considered: int


@dataclass
class GroundingEngine:
    """
    A practical grounding engine that supports OmniPath/INDRA-style inputs.

    Expected references:
    - abstract_types: protein -> type label
    - real_nodes: list[dict(name, type)]
    - interactions: list[dict(src, dst, interaction_type, confidence)]
    """

    def _normalize_edge_type(self, edge_type: str) -> str:
        raw = edge_type.lower().strip()
        if "phosph" in raw:
            return "phosphorylation"
        if "inhib" in raw:
            return "inhibition"
        if "bind" in raw or "complex" in raw:
            return "binding"
        return raw

    def build_constraint_matrix(
        self,
        abstract_types: Dict[str, str],
        real_nodes: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        by_type: Dict[str, List[str]] = {}
        for node in real_nodes:
            by_type.setdefault(node.get("type", "unknown"), []).append(node["name"])

        constraints: Dict[str, List[str]] = {}
        for abstract_node, required_type in abstract_types.items():
            constraints[abstract_node] = list(by_type.get(required_type, []))
        return constraints

    def prune_constraints_by_degree(
        self,
        network: ReactionNetwork,
        constraints: Dict[str, List[str]],
        real_interactions: List[Tuple[str, str, str]],
    ) -> Dict[str, List[str]]:
        abstract_degree: Dict[str, int] = {k: 0 for k in constraints.keys()}
        for r in network.rules:
            if len(r.reactants) >= 2:
                s, t = r.reactants[0], r.reactants[-1]
                if s in abstract_degree:
                    abstract_degree[s] += 1
                if t in abstract_degree:
                    abstract_degree[t] += 1

        real_degree: Dict[str, int] = {}
        for src, dst, _ in real_interactions:
            real_degree[src] = real_degree.get(src, 0) + 1
            real_degree[dst] = real_degree.get(dst, 0) + 1

        pruned: Dict[str, List[str]] = {}
        for abstract, candidates in constraints.items():
            threshold = abstract_degree.get(abstract, 0)
            survivors = [c for c in candidates if real_degree.get(c, 0) >= threshold]
            pruned[abstract] = survivors or list(candidates)
        return pruned

    def match_abstract_to_real(
        self,
        network: ReactionNetwork,
        constraints: Dict[str, List[str]],
        real_interactions: List[Tuple[str, str, str]],
    ) -> List[Dict[str, str]]:
        abstract_nodes = [n for n in network.proteins.keys() if n in constraints]
        edge_types = {
            (r.reactants[0], r.reactants[-1], self._normalize_edge_type(r.rule_type))
            for r in network.rules
            if len(r.reactants) >= 2 and r.reactants[0] in constraints and r.reactants[-1] in constraints
        }
        normalized_real_interactions = [
            (src, dst, self._normalize_edge_type(edge_t)) for src, dst, edge_t in real_interactions
        ]
        constraints = self.prune_constraints_by_degree(network, constraints, normalized_real_interactions)
        real_edge_set = set(normalized_real_interactions)

        solutions: List[Dict[str, str]] = []

        def backtrack(i: int, used: set[str], mapping: Dict[str, str]) -> None:
            if i == len(abstract_nodes):
                if self._mapping_respects_edges(mapping, edge_types, real_edge_set):
                    solutions.append(dict(mapping))
                return

            abstract = abstract_nodes[i]
            for candidate in constraints.get(abstract, []):
                if candidate in used:
                    continue
                mapping[abstract] = candidate
                used.add(candidate)
                backtrack(i + 1, used, mapping)
                used.remove(candidate)
                del mapping[abstract]

        backtrack(0, set(), {})
        return solutions

    def _mapping_respects_edges(
        self,
        mapping: Dict[str, str],
        abstract_edges: set[Tuple[str, str, str]],
        real_edges: set[Tuple[str, str, str]],
    ) -> bool:
        for src_a, dst_a, edge_t in abstract_edges:
            src_r = mapping.get(src_a)
            dst_r = mapping.get(dst_a)
            if not src_r or not dst_r:
                return False
            if (src_r, dst_r, edge_t) not in real_edges:
                return False
        return True

    def score_mappings(
        self,
        mappings: List[Dict[str, str]],
        confidence_by_pair: Dict[Any, float],
    ) -> GroundingResult:
        if not mappings:
            return GroundingResult(mapping={}, score=0.0, candidates_considered=0)

        def score_mapping(m: Dict[str, str]) -> float:
            values = []
            for a, r in m.items():
                direct = confidence_by_pair.get((a, r), None)
                if direct is not None:
                    values.append(direct)
                    continue
                values.append(confidence_by_pair.get(f"{a}->{r}", 0.0))
            return sum(values) / max(1, len(values))

        scored = [(score_mapping(m), m) for m in mappings]
        scored.sort(key=lambda x: x[0], reverse=True)
        return GroundingResult(mapping=scored[0][1], score=scored[0][0], candidates_considered=len(mappings))
