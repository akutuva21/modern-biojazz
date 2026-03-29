"""Assemble BNGL models from species lists via INDRA.

Two modes:
  1. Live: query INDRA DB REST API (no key needed) → statements → BNGL text
  2. Offline: load pre-assembled BNGL from file
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AssemblyResult:
    species: List[str]
    statements: List[Dict[str, Any]]
    bngl_text: str
    source: str  # "indra_live", "indra_cached", "file"


@dataclass
class INDRAAssembler:
    """Query INDRA DB REST and assemble statements into BNGL.

    The public INDRA DB REST API (https://db.indra.bio) requires no API key.
    Statements are fetched for the given species, then assembled into BNGL
    using INDRA's BnglAssembler (requires `indra` pip package locally).
    """

    db_rest_url: str = "https://db.indra.bio"
    timeout_seconds: float = 120.0
    statement_types: List[str] = field(
        default_factory=lambda: [
            "Phosphorylation",
            "Dephosphorylation",
            "Inhibition",
            "Activation",
            "IncreaseAmount",
            "DecreaseAmount",
            "Complex",
        ]
    )
    max_statements_per_type: int = 100

    def assemble(self, species: List[str]) -> AssemblyResult:
        """Fetch statements from INDRA DB REST and assemble into BNGL."""
        all_stmts = self._fetch_statements(species)
        bngl_text = self._assemble_bngl(all_stmts, species)
        return AssemblyResult(
            species=species,
            statements=all_stmts,
            bngl_text=bngl_text,
            source="indra_live",
        )

    def _fetch_statements(self, species: List[str]) -> List[Dict[str, Any]]:
        """Fetch statements from INDRA DB REST for all configured statement types."""
        all_statements: List[Dict[str, Any]] = []

        for stmt_type in self.statement_types:
            try:
                stmts = self._query_db_rest(species, stmt_type)
                all_statements.extend(stmts[: self.max_statements_per_type])
            except Exception:
                # Some statement types may not return results; continue.
                continue

        return all_statements

    def _query_db_rest(self, species: List[str], stmt_type: str) -> List[Dict[str, Any]]:
        payload = json.dumps({
            "subject": species,
            "object": species,
            "type": stmt_type,
            "ev_limit": 5,
        }).encode("utf-8")

        req = urllib.request.Request(
            url=f"{self.db_rest_url.rstrip('/')}/statements/from_agents",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("statements", data) if isinstance(data, dict) else data

    def _assemble_bngl(
        self,
        raw_statements: List[Dict[str, Any]],
        species: List[str],
    ) -> str:
        """Try INDRA's Python BnglAssembler first; fall back to manual assembly."""
        try:
            return self._assemble_via_indra_lib(raw_statements)
        except ImportError:
            return self._assemble_manual(raw_statements, species)

    def _assemble_via_indra_lib(self, raw_statements: List[Dict[str, Any]]) -> str:
        """Use indra.assemblers.bngl if the indra package is installed."""
        from indra.statements import stmts_from_json  # type: ignore
        from indra.assemblers.bngl import BnglAssembler  # type: ignore

        stmts = stmts_from_json(raw_statements)
        ba = BnglAssembler()
        ba.add_statements(stmts)
        return ba.make_model()

    def _assemble_manual(
        self,
        raw_statements: List[Dict[str, Any]],
        species: List[str],
    ) -> str:
        """Lightweight BNGL assembly when the indra package is not installed.

        Produces a valid BNGL file from raw INDRA JSON statements by extracting
        agent names, modification types, and generating mass-action rules.
        """
        mol_types: Dict[str, Dict[str, List[str]]] = {}  # protein -> {site -> [states]}
        rules: List[str] = []
        params: Dict[str, float] = {}
        param_counter = 0

        for s in species:
            mol_types.setdefault(s, {})

        for stmt in raw_statements:
            stype = stmt.get("type", "").lower()
            agents = stmt.get("agents") or stmt.get("members") or []
            if not agents:
                # Try sub/obj format
                sub = stmt.get("sub") or stmt.get("subj") or stmt.get("enz")
                obj = stmt.get("obj")
                if sub and obj:
                    agents = [sub, obj]

            agent_names = []
            for a in agents:
                if isinstance(a, dict):
                    name = a.get("name", a.get("db_refs", {}).get("HGNC", ""))
                elif isinstance(a, str):
                    name = a
                else:
                    continue
                if name:
                    agent_names.append(name)
                    mol_types.setdefault(name, {})

            if len(agent_names) < 2:
                continue

            param_counter += 1
            kinase, substrate = agent_names[0], agent_names[1]

            if "phosphorylation" in stype:
                site = _extract_site(stmt) or "phospho"
                mol_types.setdefault(substrate, {})[site] = ["u", "p"]
                pname = f"kp_{param_counter}"
                params[pname] = _extract_belief(stmt) * 0.1
                rules.append(
                    f"  r{param_counter}: "
                    f"{kinase}() + {substrate}({site}~u) -> "
                    f"{kinase}() + {substrate}({site}~p) {pname}"
                )

            elif "dephosphorylation" in stype:
                site = _extract_site(stmt) or "phospho"
                mol_types.setdefault(substrate, {})[site] = ["u", "p"]
                pname = f"kdp_{param_counter}"
                params[pname] = _extract_belief(stmt) * 0.05
                rules.append(
                    f"  r{param_counter}: "
                    f"{kinase}() + {substrate}({site}~p) -> "
                    f"{kinase}() + {substrate}({site}~u) {pname}"
                )

            elif "complex" in stype or "bind" in stype:
                mol_types.setdefault(kinase, {}).setdefault(f"b_{substrate}", [])
                mol_types.setdefault(substrate, {}).setdefault(f"b_{kinase}", [])
                pname_f = f"kf_{param_counter}"
                pname_r = f"kr_{param_counter}"
                params[pname_f] = _extract_belief(stmt) * 0.01
                params[pname_r] = 0.1
                rules.append(
                    f"  r{param_counter}: "
                    f"{kinase}(b_{substrate}) + {substrate}(b_{kinase}) <-> "
                    f"{kinase}(b_{substrate}!1).{substrate}(b_{kinase}!1) {pname_f}, {pname_r}"
                )

            elif "inhibition" in stype or "decreaseamount" in stype:
                mol_types.setdefault(substrate, {}).setdefault("activity", ["active", "inactive"])
                pname = f"ki_{param_counter}"
                params[pname] = _extract_belief(stmt) * 0.05
                rules.append(
                    f"  r{param_counter}: "
                    f"{kinase}() + {substrate}(activity~active) -> "
                    f"{kinase}() + {substrate}(activity~inactive) {pname}"
                )

            elif "activation" in stype or "increaseamount" in stype:
                mol_types.setdefault(substrate, {}).setdefault("activity", ["active", "inactive"])
                pname = f"ka_{param_counter}"
                params[pname] = _extract_belief(stmt) * 0.1
                rules.append(
                    f"  r{param_counter}: "
                    f"{kinase}() + {substrate}(activity~inactive) -> "
                    f"{kinase}() + {substrate}(activity~active) {pname}"
                )

        return _render_bngl(mol_types, params, rules, species)


def load_assembly_snapshot(path: str | Path) -> AssemblyResult:
    """Load a cached assembly result."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AssemblyResult(
        species=data["species"],
        statements=data.get("statements", []),
        bngl_text=data["bngl_text"],
        source=data.get("source", "file"),
    )


def save_assembly_snapshot(result: AssemblyResult, path: str | Path) -> None:
    """Save an assembly result for offline use."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "species": result.species,
                "statements": result.statements,
                "bngl_text": result.bngl_text,
                "source": result.source,
            },
            f,
            indent=2,
        )


def load_bngl_file(path: str | Path) -> AssemblyResult:
    """Load a raw .bngl file as an assembly result."""
    text = Path(path).read_text(encoding="utf-8")
    return AssemblyResult(species=[], statements=[], bngl_text=text, source="file")


# ── Helpers ──────────────────────────────────────────────────────────


def _extract_site(stmt: Dict[str, Any]) -> Optional[str]:
    """Try to extract a residue/position site name from an INDRA statement."""
    residue = stmt.get("residue", "")
    position = stmt.get("position", "")
    if residue and position:
        return f"{residue}{position}"
    if position:
        return f"site{position}"
    return None


def _extract_belief(stmt: Dict[str, Any]) -> float:
    return float(stmt.get("belief", 0.7))


def _render_bngl(
    mol_types: Dict[str, Dict[str, List[str]]],
    params: Dict[str, float],
    rules: List[str],
    species: List[str],
) -> str:
    lines = ["begin model", "", "begin parameters"]
    for pname, pval in sorted(params.items()):
        lines.append(f"  {pname} {pval:.6g}")
    lines.append("end parameters")
    lines.append("")

    lines.append("begin molecule types")
    for mol_name in sorted(mol_types.keys()):
        sites = mol_types[mol_name]
        if not sites:
            lines.append(f"  {mol_name}()")
        else:
            site_strs = []
            for sname, states in sorted(sites.items()):
                if states:
                    site_strs.append(f"{sname}~{'~'.join(states)}")
                else:
                    site_strs.append(sname)
            lines.append(f"  {mol_name}({','.join(site_strs)})")
    lines.append("end molecule types")
    lines.append("")

    lines.append("begin seed species")
    for mol_name in sorted(mol_types.keys()):
        if mol_name not in species:
            continue
        sites = mol_types[mol_name]
        if not sites:
            lines.append(f"  {mol_name}() 1.0")
        else:
            init_sites = []
            for sname, states in sorted(sites.items()):
                if states:
                    init_sites.append(f"{sname}~{states[0]}")
                else:
                    init_sites.append(sname)
            lines.append(f"  {mol_name}({','.join(init_sites)}) 1.0")
    lines.append("end seed species")
    lines.append("")

    lines.append("begin reaction rules")
    lines.extend(rules)
    lines.append("end reaction rules")
    lines.append("")

    lines.append("begin observables")
    for mol_name in sorted(mol_types.keys()):
        if mol_name in species:
            lines.append(f"  Molecules {mol_name}_obs {mol_name}()")
    lines.append("end observables")
    lines.append("")
    lines.append("end model")
    return "\n".join(lines)
