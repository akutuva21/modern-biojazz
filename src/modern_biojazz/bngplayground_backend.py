"""BNG Playground MCP server adapter for simulation and BNGL validation.

Uses the BNG Playground MCP server (packages/mcp-server) as a simulation
backend. Communicates via JSON-RPC over subprocess stdio.

This gives modern-biojazz access to:
  - CVODE stiff ODE solver (SUNDIALS via WASM)
  - Full BNGL parser with ANTLR4 grammar
  - Network expansion (rule-based → species-based)
  - Parameter scanning and fitting
  - Contact map generation

Requirements:
  - Node.js >= 18
  - BNG Playground repo cloned with `npm install` done
  - Set BNGPLAYGROUND_PATH env var to the repo root

Usage:
    backend = BNGPlaygroundBackend(bngplayground_path="/path/to/bngplayground")
    options = SimulationOptions(t_end=20.0, dt=1.0, solver="cvode")
    result = backend.simulate(network, options)
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .site_graph import ReactionNetwork
from .simulation import SimulationOptions


class BNGLParsingError(Exception):
    """Exception raised when BNGL parsing fails."""
    pass


@lru_cache(maxsize=16384)
def _format_rule(name: str, reactants: tuple, products: tuple) -> str:
    """Format and cache the string representation of a reaction rule."""
    reactant_str = " + ".join([r + "()" for r in reactants])
    product_str = " + ".join([p + "()" for p in products])
    return f"  {name}: {reactant_str} -> {product_str} {name}_rate"


@dataclass
class BNGPlaygroundBackend:
    """Simulation backend that calls the BNG Playground MCP server."""

    bngplayground_path: str = ""
    node_command: str = "node"
    timeout_seconds: float = 120.0
    _request_id: int = 0

    def __post_init__(self) -> None:
        if not self.bngplayground_path:
            self.bngplayground_path = os.environ.get("BNGPLAYGROUND_PATH", "")

    def simulate(
        self,
        network: ReactionNetwork,
        options: SimulationOptions,
    ) -> Dict[str, Any]:
        """Simulate by converting ReactionNetwork → BNGL text → MCP simulate call."""
        bngl_code = self._network_to_bngl(
            network, options.t_end, options.dt, options.initial_conditions
        )

        mcp_result = self._call_mcp_tool("simulate", {
            "code": bngl_code,
            "method": "ode" if options.solver in ("cvode", "auto", "BDF", "FBDF") else "ode",
            "t_end": options.t_end,
            "n_steps": max(1, int(options.t_end / options.dt)),
            "solver": options.solver if options.solver in ("cvode", "cvode_sparse", "rosenbrock23", "rk45") else "cvode",
            "include_species_data": True,
        })

        return self._convert_mcp_result(mcp_result, network, options.solver)

    def parse_bngl(self, bngl_code: str) -> Dict[str, Any]:
        """Parse BNGL code and return structured model."""
        result = self._call_mcp_tool("parse_bngl", {"bngl": bngl_code})
        if "error" in result:
            raise BNGLParsingError(result["error"])
        return result

    def validate_model(self, bngl_code: str) -> Dict[str, Any]:
        """Validate a BNGL model."""
        return self._call_mcp_tool("validate_model", {"code": bngl_code})

    def get_contact_map(self, bngl_code: str) -> Dict[str, Any]:
        """Get the contact map for a BNGL model."""
        return self._call_mcp_tool("get_contact_map", {"code": bngl_code})

    def fit_parameters(
        self,
        bngl_code: str,
        experimental_data: Dict[str, Any],
        parameters_to_fit: List[str],
    ) -> Dict[str, Any]:
        """Fit model parameters to experimental data."""
        return self._call_mcp_tool("fit_parameters", {
            "code": bngl_code,
            "experimental_data": experimental_data,
            "parameters_to_fit": parameters_to_fit,
        })

    def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via subprocess JSON-RPC."""
        if not self.bngplayground_path:
            raise RuntimeError(
                "BNGPLAYGROUND_PATH not set. Point it to your bngplayground repo root."
            )

        server_script = os.path.join(
            self.bngplayground_path, "packages", "mcp-server", "src", "index.ts"
        )

        self._request_id += 1

        # JSON-RPC initialize → tool call → shutdown
        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "modern-biojazz", "version": "0.1.0"},
            },
        })

        self._request_id += 1
        tool_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })

        # Send both messages separated by newlines
        stdin_data = init_msg + "\n" + tool_msg + "\n"

        try:
            # Try tsx first (TypeScript loader), fall back to compiled JS
            cmd = self._build_command(server_script)
            proc = subprocess.run(  # nosec B603
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env={**os.environ, "MCP_SERVER_RUN": "true"},
                cwd=self.bngplayground_path,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Could not find node/tsx. Ensure Node.js is installed."
            )

        if proc.returncode != 0:
            stderr = proc.stderr[:500] if proc.stderr else ""
            raise RuntimeError(f"MCP server failed (rc={proc.returncode}): {stderr}")

        return self._parse_mcp_response(proc.stdout)

    def _build_command(self, server_script: str) -> List[str]:
        """Build the command to launch the MCP server."""
        # Resolve executables to absolute paths to prevent cwd hijacking
        node_exec = shutil.which(self.node_command)
        if not node_exec:
            raise FileNotFoundError(f"Could not find executable for '{self.node_command}'")

        # Check if dist/index.js exists (pre-compiled)
        dist_path = os.path.join(
            self.bngplayground_path, "packages", "mcp-server", "dist", "index.js"
        )
        if os.path.exists(dist_path):
            return [node_exec, dist_path]

        # Fall back to tsx for TypeScript
        npx_exec = shutil.which("npx")
        if not npx_exec:
            raise FileNotFoundError("Could not find executable for 'npx'")

        return [npx_exec, "tsx", server_script]

    def _extract_text_content(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract JSON or raw text from the content array of an MCP result."""
        for item in result.get("content", []):
            if item.get("type") == "text":
                try:
                    return json.loads(item["text"])
                except (json.JSONDecodeError, TypeError):
                    return {"raw_text": item.get("text", "")}
        return None

    def _parse_mcp_response(self, stdout: str) -> Dict[str, Any]:
        """Parse the JSON-RPC response(s) from stdout."""
        # The server may emit multiple JSON objects (one per request).
        # We want the tool result (the last one, typically).
        lines = stdout.strip().split("\n")
        last_response: Dict[str, Any] = {}
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict) or "result" not in obj:
                continue

            result = obj["result"]

            if isinstance(result, dict) and "content" in result:
                text_content = self._extract_text_content(result)
                if text_content is not None:
                    return text_content

            return result

        return last_response

    def _generate_parameters(self, network: ReactionNetwork) -> List[str]:
        lines = ["begin parameters"]
        lines.extend(f"  {rule.name}_rate {rule.rate:.6g}" for rule in network.rules)
        lines.extend(["end parameters", ""])
        return lines

    def _generate_molecule_types(self, network: ReactionNetwork) -> List[str]:
        lines = ["begin molecule types"]
        for protein in network.proteins.values():
            if not protein.sites:
                lines.append(f"  {protein.name}()")
            else:
                site_strs = [
                    f"{site.name}~{'~'.join(site.states)}" if site.states else site.name
                    for site in protein.sites
                ]
                lines.append(f"  {protein.name}({','.join(site_strs)})")
        lines.extend(["end molecule types", ""])
        return lines

    def _generate_seed_species(self, network: ReactionNetwork, initial_conditions: Dict[str, float] | None) -> List[str]:
        ic = initial_conditions or {}
        lines = ["begin seed species"]
        for protein in network.proteins.values():
            conc = ic.get(protein.name, 1.0)
            if not protein.sites:
                lines.append(f"  {protein.name}() {conc}")
            else:
                init_sites = [
                    f"{site.name}~{site.states[0]}" if site.states else site.name
                    for site in protein.sites
                ]
                lines.append(f"  {protein.name}({','.join(init_sites)}) {conc}")
        lines.extend(["end seed species", ""])
        return lines

    def _generate_rules(self, network: ReactionNetwork) -> List[str]:
        lines = ["begin reaction rules"]
        for rule in network.rules:
            reactant_str = " + ".join(f"{r}()" for r in rule.reactants)
            product_str = " + ".join(f"{p}()" for p in rule.products)
            lines.append(f"  {rule.name}: {reactant_str} -> {product_str} {rule.name}_rate")
        lines.extend(["end reaction rules", ""])
        return lines

    def _generate_observables(self, network: ReactionNetwork) -> List[str]:
        lines = ["begin observables"]
        lines.extend(f"  Molecules {protein.name}_obs {protein.name}()" for protein in network.proteins.values())
        lines.extend(["end observables", ""])
        return lines

    def _network_to_bngl(
        self,
        network: ReactionNetwork,
        t_end: float,
        dt: float,
        initial_conditions: Dict[str, float] | None = None,
    ) -> str:
        """Convert a ReactionNetwork to BNGL text for the MCP server."""
        lines = ["begin model", ""]

        lines.extend(self._generate_parameters(network))
        lines.extend(self._generate_molecule_types(network))
        lines.extend(self._generate_seed_species(network, initial_conditions))
        lines.extend(self._generate_rules(network))
        lines.extend(self._generate_observables(network))

        lines.extend([
            "end model",
            "",
            "generate_network({max_iter=>100})",
            f"simulate({{method=>\"ode\", t_end=>{t_end}, n_steps=>{max(1, int(t_end/dt))}}})"
        ])

        return "\n".join(lines)

    def _convert_mcp_result(
        self,
        mcp_result: Dict[str, Any],
        network: ReactionNetwork,
        solver: str,
    ) -> Dict[str, Any]:
        """Convert MCP simulation result to the standard modern-biojazz format."""
        # MCP simulate returns {time: [...], observables: {name: [...]}, ...}
        time_points = mcp_result.get("time", mcp_result.get("t", []))
        observables = mcp_result.get("observables", {})
        species_data = mcp_result.get("species", {})

        output_species = network.metadata.get("output_species")
        if output_species and f"{output_species}_obs" in observables:
            output_key = f"{output_species}_obs"
        elif observables:
            output_key = next(iter(observables))
        else:
            output_key = ""

        trajectory = []
        for i, t in enumerate(time_points):
            species_map = {}
            for name, values in (species_data or observables).items():
                if i < len(values):
                    species_map[name] = float(values[i])

            output_val = 0.0
            if output_key and output_key in observables and i < len(observables[output_key]):
                output_val = float(observables[output_key][i])

            trajectory.append({"t": float(t), "output": output_val, "species": species_map})

        return {
            "solver": mcp_result.get("solver", solver),
            "trajectory": trajectory,
            "stats": {
                "n_rules": len(network.rules),
                "n_species": mcp_result.get("n_species", len(network.proteins)),
                "stiff": True,  # CVODE handles stiff systems
            },
        }
