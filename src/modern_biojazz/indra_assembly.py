"""Assemble BNGL models from species lists via INDRA.

Two modes:
  1. Live: query INDRA DB REST API (no key needed) → statements → BNGL text
  2. Offline: load pre-assembled BNGL from file
"""
from __future__ import annotations

import json
import logging
import urllib.request
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import save_json_snapshot

logger = logging.getLogger(__name__)


@dataclass
class AssemblyState:
    mol_types: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    params: Dict[str, float] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    param_counter: int = 0


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
        url = f"{self.db_rest_url.rstrip('/')}/statements/from_agents?format=json"

        all_stmts = []
        for s in species:
            payload = json.dumps({
                "agent0": s,
                "stmt_type": stmt_type,
                "ev_limit": 5,
            }).encode("utf-8")

            req = urllib.request.Request(
                url=url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    stmts = data.get("statements", data) if isinstance(data, dict) else data
                    if isinstance(stmts, list):
                        all_stmts.extend(stmts)
            except Exception:
                continue

        return all_stmts

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
        state = AssemblyState()

        for s in species:
            state.mol_types.setdefault(s, {})

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
                    state.mol_types.setdefault(name, {})

            if len(agent_names) < 2:
                continue

            state.param_counter += 1
            kinase, substrate = agent_names[0], agent_names[1]

            if "phosphorylation" in stype:
                _handle_phosphorylation(stmt, kinase, substrate, state)

            elif "dephosphorylation" in stype:
                _handle_dephosphorylation(stmt, kinase, substrate, state)

            elif "complex" in stype or "bind" in stype:
                _handle_complex(stmt, kinase, substrate, state)

            elif "inhibition" in stype or "decreaseamount" in stype:
                _handle_inhibition(stmt, kinase, substrate, state)

            elif "activation" in stype or "increaseamount" in stype:
                _handle_activation(stmt, kinase, substrate, state)

        return _render_bngl(state.mol_types, state.params, state.rules, species)


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
    save_json_snapshot(
        {
            "species": result.species,
            "statements": result.statements,
            "bngl_text": result.bngl_text,
            "source": result.source,
        },
        path,
    )


def load_bngl_file(path: str | Path) -> AssemblyResult:
    """Load a raw .bngl file as an assembly result."""
    text = Path(path).read_text(encoding="utf-8")
    return AssemblyResult(species=[], statements=[], bngl_text=text, source="file")


@dataclass
class INDRAGraphProposer:
    """An LLM-free proposer that queries INDRA DB to generate valid network additions.

    Acts as a bridge between real biology (via INDRA) and graph mutation actions.
    When asked to propose, it randomly picks a protein from the network, asks INDRA what
    interacts with it, and translates that into 'add_protein', 'add_phosphorylation', etc.
    """

    assembler: INDRAAssembler = field(default_factory=INDRAAssembler)
    rng: random.Random = field(default_factory=random.Random)

    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        # Parse available proteins from model_code (format: '...proteins=['A', 'B'];...')
        proteins = []
        try:
            import re
            m = re.search(r"proteins=\[(.*?)\]", model_code)
            if m:
                raw = m.group(1).replace("'", "").replace('"', "").split(",")
                proteins = [p.strip() for p in raw if p.strip() and not p.strip().startswith("M_")]
        except Exception as e:
            logger.warning("Failed to parse proteins from model_code: %s", e)

        proteins = [p for p in proteins if p not in ['0', 'Trash']]

        if not proteins:
            return [self.rng.choice(action_names) for _ in range(max(1, budget))]

        target = self.rng.choice(proteins)

        # Query INDRA for statements involving this target
        # For speed, we just grab 5 statements of a random type
        stmt_type = self.rng.choice(["Phosphorylation", "Complex", "Activation"])
        statements = self.assembler._query_db_rest([target], stmt_type)

        if not statements:
            # Fallback
            return [self.rng.choice(action_names) for _ in range(max(1, budget))]

        stmt = self.rng.choice(statements)

        # Extract agents from statement
        agents = stmt.get("agents") or stmt.get("members") or []
        if not agents:
            sub = stmt.get("sub") or stmt.get("subj") or stmt.get("enz")
            obj = stmt.get("obj")
            if sub and obj:
                agents = [sub, obj]

        new_names = []
        for a in agents:
            if isinstance(a, dict):
                n = a.get("name", a.get("db_refs", {}).get("HGNC", ""))
                if n:
                    new_names.append(n)
            elif isinstance(a, str):
                new_names.append(a)

        if len(new_names) < 2:
            return [self.rng.choice(action_names) for _ in range(max(1, budget))]

        # We now have a real interaction (e.g. ['JAK2', 'STAT3']).
        # Because the proposer API expects *action names*, we would ideally return
        # a specialized action string, but since we are constrained to the current mutator API
        # which acts randomly, we can signal the pipeline to run a specific mutation.
        # However, to fit cleanly into the LLMEvolutionEngine pipeline which calls:
        # actions.get(action_name).apply(child)
        # We must return an action name from `action_names`.

        # Since standard BioJazz mutator actions are entirely random (e.g. `add_phosphorylation`
        # picks random nodes), we can instead return standard actions but bias toward growth.
        # To truly inject INDRA knowledge, we would need to dynamically inject an action into the
        # library. For simplicity in this interface constraint, we just map the INDRA
        # statement type to the closest BioJazz mutation action.

        proposals = []
        if stmt_type == "Phosphorylation" and "add_phosphorylation" in action_names:
            proposals.append("add_phosphorylation")
        elif stmt_type == "Complex" and "add_binding" in action_names:
            proposals.append("add_binding")
        elif stmt_type in ("Activation", "Inhibition") and "add_site" in action_names:
            proposals.append("add_site")

        while len(proposals) < budget:
            proposals.append(self.rng.choice(action_names))

        return proposals[:budget]

    def record_feedback(self, score: float, notes: str) -> None:
        pass


# ── Helpers ──────────────────────────────────────────────────────────


def _handle_phosphorylation(
    stmt: Dict[str, Any],
    kinase: str,
    substrate: str,
    state: AssemblyState
) -> None:
    site = _extract_site(stmt) or "phospho"
    state.mol_types.setdefault(substrate, {})[site] = ["u", "p"]
    pname = f"kp_{state.param_counter}"
    state.params[pname] = _extract_belief(stmt) * 0.1
    state.rules.append(
        f"  r{state.param_counter}: "
        f"{kinase}() + {substrate}({site}~u) -> "
        f"{kinase}() + {substrate}({site}~p) {pname}"
    )


def _handle_dephosphorylation(
    stmt: Dict[str, Any],
    kinase: str,
    substrate: str,
    state: AssemblyState
) -> None:
    site = _extract_site(stmt) or "phospho"
    state.mol_types.setdefault(substrate, {})[site] = ["u", "p"]
    pname = f"kdp_{state.param_counter}"
    state.params[pname] = _extract_belief(stmt) * 0.05
    state.rules.append(
        f"  r{state.param_counter}: "
        f"{kinase}() + {substrate}({site}~p) -> "
        f"{kinase}() + {substrate}({site}~u) {pname}"
    )


def _handle_complex(
    stmt: Dict[str, Any],
    kinase: str,
    substrate: str,
    state: AssemblyState
) -> None:
    state.mol_types.setdefault(kinase, {}).setdefault(f"b_{substrate}", [])
    state.mol_types.setdefault(substrate, {}).setdefault(f"b_{kinase}", [])
    pname_f = f"kf_{state.param_counter}"
    pname_r = f"kr_{state.param_counter}"
    state.params[pname_f] = _extract_belief(stmt) * 0.01
    state.params[pname_r] = 0.1
    state.rules.append(
        f"  r{state.param_counter}: "
        f"{kinase}(b_{substrate}) + {substrate}(b_{kinase}) <-> "
        f"{kinase}(b_{substrate}!1).{substrate}(b_{kinase}!1) {pname_f}, {pname_r}"
    )


def _handle_inhibition(
    stmt: Dict[str, Any],
    kinase: str,
    substrate: str,
    state: AssemblyState
) -> None:
    state.mol_types.setdefault(substrate, {}).setdefault("activity", ["active", "inactive"])
    pname = f"ki_{state.param_counter}"
    state.params[pname] = _extract_belief(stmt) * 0.05
    state.rules.append(
        f"  r{state.param_counter}: "
        f"{kinase}() + {substrate}(activity~active) -> "
        f"{kinase}() + {substrate}(activity~inactive) {pname}"
    )


def _handle_activation(
    stmt: Dict[str, Any],
    kinase: str,
    substrate: str,
    state: AssemblyState
) -> None:
    state.mol_types.setdefault(substrate, {}).setdefault("activity", ["active", "inactive"])
    pname = f"ka_{state.param_counter}"
    state.params[pname] = _extract_belief(stmt) * 0.1
    state.rules.append(
        f"  r{state.param_counter}: "
        f"{kinase}() + {substrate}(activity~inactive) -> "
        f"{kinase}() + {substrate}(activity~active) {pname}"
    )


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
