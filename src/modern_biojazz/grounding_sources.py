from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class OmniPathClient:
    base_url: str = "https://omnipathdb.org"
    timeout_seconds: float = 45.0

    def fetch_interactions(self, genes: List[str]) -> List[Dict[str, Any]]:
        gene_csv = ",".join(sorted(set(genes)))
        params = {
            "genesymbols": "1",
            "fields": "sources,references,curation_effort,dorothea_level,type",
            "format": "json",
            "sources": gene_csv,
            "targets": gene_csv,
        }
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/interactions/?{query}",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return payload


@dataclass
class INDRAClient:
    base_url: str = "https://api.indra.bio"
    timeout_seconds: float = 45.0

    def fetch_statements(self, genes: List[str], stmt_type: str = "Phosphorylation") -> List[Dict[str, Any]]:
        payload = {
            "subject": genes,
            "object": genes,
            "type": stmt_type,
            "format": "json",
        }
        req = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/statements/from_agents",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("statements", [])


def load_grounding_snapshot(snapshot_path: str | Path) -> Dict[str, Any]:
    path = Path(snapshot_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_grounding_payload_from_sources(
    abstract_types: Dict[str, str],
    omnipath_rows: List[Dict[str, Any]],
    indra_statements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Construct a grounding payload from live or cached source rows/statements."""

    normalized_type_to_abstract: Dict[str, List[str]] = {}
    for a, t in abstract_types.items():
        normalized_type_to_abstract.setdefault(t.lower(), []).append(a)

    def infer_node_type(node_name: str) -> str:
        upper_name = node_name.upper()
        for t, aliases in normalized_type_to_abstract.items():
            for alias in aliases:
                if upper_name.startswith(alias.upper()):
                    return t
        return "unknown"

    def confidence_for_pair(abstract_name: str, node_name: str) -> float:
        if node_name.upper() == abstract_name.upper() or node_name.upper().startswith(f"{abstract_name.upper()}_"):
            return 0.95
        if node_name.upper().startswith(abstract_name.upper()):
            return 0.8
        return 0.2

    real_nodes_map: Dict[str, Dict[str, Any]] = {}
    real_interactions: List[List[str]] = []
    confidence_by_pair: Dict[str, float] = {}

    for row in omnipath_rows:
        src = row.get("source_genesymbol") or row.get("source")
        dst = row.get("target_genesymbol") or row.get("target")
        if not src or not dst:
            continue
        real_nodes_map.setdefault(src, {"name": src, "type": infer_node_type(src)})
        real_nodes_map.setdefault(dst, {"name": dst, "type": infer_node_type(dst)})

    for stmt in indra_statements:
        agents = stmt.get("agents", [])
        if len(agents) < 2:
            continue
        src = agents[0].get("name")
        dst = agents[1].get("name")
        if not src or not dst:
            continue
        stype = str(stmt.get("type", "binding")).lower()
        real_interactions.append([src, dst, stype])

    real_nodes = list(real_nodes_map.values())

    for abstract in abstract_types.keys():
        for node in real_nodes:
            confidence_by_pair[f"{abstract}->{node['name']}"] = confidence_for_pair(abstract, node["name"])

    return {
        "abstract_types": abstract_types,
        "real_nodes": real_nodes,
        "real_interactions": real_interactions,
        "confidence_by_pair": confidence_by_pair,
    }
