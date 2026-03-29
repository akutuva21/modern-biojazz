"""Discover signaling pathway species from seed genes via OmniPath.

Provides the species enumeration step for the end-to-end pipeline:
  pathway query → species set → INDRA assembly → BNGL → evolution
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class PathwayDiscoveryResult:
    seed_genes: List[str]
    species: List[str]
    interactions: List[Dict[str, Any]]
    source: str  # "omnipath", "snapshot", etc.


@dataclass
class OmniPathDiscovery:
    """Query OmniPath for directed signaling interactions around seed genes.

    Expands a small set of seed genes (e.g. ["IL6", "STAT3", "TGFB1", "SMAD3"])
    into the full interacting neighborhood, returning both the species list and
    the interaction edges (used later for grounding constraints).
    """

    base_url: str = "https://omnipathdb.org"
    timeout_seconds: float = 60.0
    resources: List[str] = field(default_factory=lambda: ["SignaLink3", "SIGNOR", "Reactome", "KEGG"])
    max_neighborhood_size: int = 80

    def discover(
        self,
        seed_genes: List[str],
        expand_neighborhood: bool = True,
    ) -> PathwayDiscoveryResult:
        """Fetch interactions involving seed genes and optionally expand to their neighbors."""
        interactions = self._fetch_interactions(seed_genes)

        if expand_neighborhood:
            species = self._extract_species(interactions, seed_genes)
        else:
            species = sorted(set(seed_genes))

        return PathwayDiscoveryResult(
            seed_genes=list(seed_genes),
            species=species,
            interactions=interactions,
            source="omnipath",
        )

    def discover_from_pathway(
        self,
        pathway_query: str,
    ) -> PathwayDiscoveryResult:
        """Discover species from a Reactome/KEGG pathway name substring.

        Queries OmniPath annotations to find proteins annotated with the given
        pathway, then expands their interaction neighborhood.
        """
        genes = self._fetch_pathway_genes(pathway_query)
        if not genes:
            return PathwayDiscoveryResult(
                seed_genes=[],
                species=[],
                interactions=[],
                source="omnipath_pathway",
            )
        return self.discover(genes)

    def _fetch_interactions(self, genes: List[str]) -> List[Dict[str, Any]]:
        gene_csv = ",".join(sorted(set(genes)))
        params = {
            "genesymbols": "1",
            "fields": "sources,references,type,curation_effort",
            "format": "json",
            "source": gene_csv,
            "target": gene_csv,
        }
        if self.resources:
            params["databases"] = ",".join(self.resources)

        query = urllib.parse.urlencode(params)
        url = f"{self.base_url.rstrip('/')}/interactions?{query}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_species(
        self,
        interactions: List[Dict[str, Any]],
        seed_genes: List[str],
    ) -> List[str]:
        seed_set = set(seed_genes)
        species: Set[str] = set(seed_set)
        for row in interactions:
            src = row.get("source_genesymbol") or row.get("source", "")
            dst = row.get("target_genesymbol") or row.get("target", "")
            if src in seed_set or dst in seed_set:
                if src:
                    species.add(src)
                if dst:
                    species.add(dst)

        # Trim to max size, keeping seeds and sorting remainder by interaction count.
        if len(species) > self.max_neighborhood_size:
            counts: Dict[str, int] = {}
            for row in interactions:
                src = row.get("source_genesymbol", "")
                dst = row.get("target_genesymbol", "")
                counts[src] = counts.get(src, 0) + 1
                counts[dst] = counts.get(dst, 0) + 1
            extras = sorted(species - seed_set, key=lambda g: counts.get(g, 0), reverse=True)
            species = seed_set | set(extras[: self.max_neighborhood_size - len(seed_set)])

        return sorted(species)

    def _fetch_pathway_genes(self, pathway_query: str) -> List[str]:
        params = {
            "resources": "Reactome",
            "entity_type": "protein",
            "genesymbols": "1",
            "format": "json",
        }
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url.rstrip('/')}/annotations?{query}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        needle = pathway_query.lower()
        genes: Set[str] = set()
        for row in data:
            value = str(row.get("value", "")).lower()
            if needle in value:
                gene = row.get("genesymbol") or row.get("uniprot", "")
                if gene:
                    genes.add(gene)
        return sorted(genes)


def load_discovery_snapshot(path: str) -> PathwayDiscoveryResult:
    """Load a cached discovery result from JSON for offline/reproducible use."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PathwayDiscoveryResult(
        seed_genes=data["seed_genes"],
        species=data["species"],
        interactions=data["interactions"],
        source=data.get("source", "snapshot"),
    )


def save_discovery_snapshot(result: PathwayDiscoveryResult, path: str) -> None:
    """Save a discovery result to JSON for offline/reproducible use."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed_genes": result.seed_genes,
                "species": result.species,
                "interactions": result.interactions,
                "source": result.source,
            },
            f,
            indent=2,
        )
