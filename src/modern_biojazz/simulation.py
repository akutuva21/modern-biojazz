from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Protocol

from .site_graph import ReactionNetwork


class SimulationBackend(Protocol):
    def simulate(
        self,
        network: ReactionNetwork,
        t_end: float,
        dt: float,
        solver: str,
        initial_conditions: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        ...


class FitnessScorer(Protocol):
    def score(
        self,
        simulation_result: Dict[str, Any] | None = None,
        *,
        backend: SimulationBackend | None = None,
        network: ReactionNetwork | None = None,
        t_end: float = 20.0,
        dt: float = 1.0,
        solver: str = "Rodas5P",
        initial_conditions: Dict[str, float] | None = None,
    ) -> float:
        ...


@dataclass
class CatalystHTTPClient:
    base_url: str
    timeout_seconds: float = 30.0
    retry_count: int = 2

    def simulate(
        self,
        network: ReactionNetwork,
        t_end: float,
        dt: float,
        solver: str = "FBDF",
        initial_conditions: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "network": network.to_dict(),
            "t_end": t_end,
            "dt": dt,
            "solver": solver,
            "initial_conditions": initial_conditions or {},
        }
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                req = urllib.request.Request(
                    url=f"{self.base_url.rstrip('/')}/simulate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    status = getattr(response, "status", 200)
                    if status >= 400:
                        raise RuntimeError(f"Catalyst service returned HTTP {status}")
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_count:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break
        raise RuntimeError(f"Failed to simulate network via Catalyst service: {last_error}") from last_error


@dataclass
class LocalCatalystEngine:
    """Mass-action stepping engine for local integration tests and baseline scoring."""

    def simulate(
        self,
        network: ReactionNetwork,
        t_end: float,
        dt: float,
        solver: str = "FBDF",
        initial_conditions: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        species_order = list(network.proteins.keys())
        for rule in network.rules:
            for token in [*rule.reactants, *rule.products]:
                if token not in species_order:
                    species_order.append(token)

        index = {name: i for i, name in enumerate(species_order)}
        y0 = [1.0 for _ in species_order]
        for i, name in enumerate(species_order):
            if name not in network.proteins:
                y0[i] = 0.0
        if initial_conditions:
            for name, value in initial_conditions.items():
                if name not in index:
                    species_order.append(name)
                    index[name] = len(y0)
                    y0.append(0.0)
                y0[index[name]] = float(value)

        rates = [max(1e-8, float(r.rate)) for r in network.rules] or [1e-8]
        stiffness_proxy = (max(rates) / min(rates)) > 100.0
        t_eval = [i * dt for i in range(int(t_end / dt) + 1)]

        def rhs(_t: float, y: list[float]) -> list[float]:
            dydt = [0.0 for _ in y]
            for rule in network.rules:
                flux = max(0.0, float(rule.rate))
                for reactant in rule.reactants:
                    flux *= max(0.0, y[index[reactant]])
                for reactant in rule.reactants:
                    dydt[index[reactant]] -= flux
                for product in rule.products:
                    dydt[index[product]] += flux
            return dydt

        trajectory = []
        y_series = None
        used_solver = solver

        solve_ivp = None
        try:
            from scipy.integrate import solve_ivp as _solve_ivp  # type: ignore

            solve_ivp = _solve_ivp
        except ImportError:
            solve_ivp = None

        if solve_ivp is not None:
            solved = solve_ivp(
                fun=rhs,
                t_span=(0.0, t_end),
                y0=y0,
                method="BDF",
                t_eval=t_eval,
                vectorized=False,
                rtol=1e-6,
                atol=1e-9,
            )
            if not solved.success or solved.y is None:
                raise RuntimeError(f"BDF solve failed: {solved.message}")
            y_series = solved.y
            used_solver = "BDF"
        else:
            # Fallback keeps local execution available when SciPy is not present.
            current = list(y0)
            snapshots = [list(current)]
            for _ in range(1, len(t_eval)):
                deriv = rhs(0.0, current)
                current = [max(0.0, c + dt * dc) for c, dc in zip(current, deriv)]
                snapshots.append(list(current))
            # Shape contract for both solver paths: y_series[species_index][time_index].
            y_series = [list(col) for col in zip(*snapshots)]
            used_solver = "EulerFallback"

        output_species = network.metadata.get("output_species")
        if output_species not in index:
            output_species = species_order[0] if species_order else ""

        for ti, tval in enumerate(t_eval):
            species_map = {name: max(0.0, float(y_series[index[name]][ti])) for name in species_order}
            trajectory.append(
                {
                    "t": tval,
                    "output": species_map.get(output_species, 0.0),
                    "species": species_map,
                }
            )

        return {
            "solver": used_solver,
            "trajectory": trajectory,
            "stats": {
                "n_rules": len(network.rules),
                "n_species": len(species_order),
                "stiff": stiffness_proxy,
            },
        }


@dataclass
class FitnessEvaluator:
    target_output: float = 1.0

    def score(
        self,
        simulation_result: Dict[str, Any] | None = None,
        *,
        backend: SimulationBackend | None = None,
        network: ReactionNetwork | None = None,
        t_end: float = 20.0,
        dt: float = 1.0,
        solver: str = "Rodas5P",
        initial_conditions: Dict[str, float] | None = None,
    ) -> float:
        if simulation_result is None:
            if backend is None or network is None:
                raise ValueError("Either simulation_result or both backend and network must be provided.")
            simulation_result = backend.simulate(
                network,
                t_end=t_end,
                dt=dt,
                solver=solver,
                initial_conditions=initial_conditions,
            )

        trajectory = simulation_result.get("trajectory", [])
        if not trajectory:
            return 0.0
        final_output = float(trajectory[-1].get("output", 0.0))
        error = abs(self.target_output - final_output)
        base = max(0.0, 1.0 - error)
        return base


@dataclass
class UltrasensitiveFitnessEvaluator:
    """Scores dose-response steepness by estimating an effective Hill coefficient."""

    input_species: str
    output_species: str
    doses: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)

    def score(
        self,
        _simulation_result: Dict[str, Any] | None = None,
        *,
        backend: SimulationBackend | None = None,
        network: ReactionNetwork | None = None,
        t_end: float = 30.0,
        dt: float = 0.5,
        solver: str = "Rodas5P",
        _initial_conditions: Dict[str, float] | None = None,
    ) -> float:
        if backend is None or network is None:
            raise ValueError("UltrasensitiveFitnessEvaluator requires backend and network.")

        responses = []
        for dose in self.doses:
            result = backend.simulate(
                network,
                t_end=t_end,
                dt=dt,
                solver=solver,
                initial_conditions={self.input_species: dose},
            )
            series = result.get("trajectory", [])
            final = 0.0
            if series:
                final = float(series[-1].get("species", {}).get(self.output_species, series[-1].get("output", 0.0)))
            responses.append(max(1e-8, final))

        lo = responses[0]
        hi = responses[-1]
        if hi <= lo:
            return 0.0

        y10 = lo + 0.1 * (hi - lo)
        y90 = lo + 0.9 * (hi - lo)
        d10 = self._interpolate_dose(self.doses, responses, y10)
        d90 = self._interpolate_dose(self.doses, responses, y90)
        if d10 is None or d90 is None or d10 <= 0 or d90 <= 0:
            return 0.0
        if abs(d90 - d10) <= 1e-9 or d90 / max(1e-12, d10) <= 1.0001:
            return 0.0
        n_h = math.log10(81.0) / math.log10(d90 / d10)
        return max(0.0, min(10.0, n_h))

    def _interpolate_dose(self, doses: tuple[float, ...], responses: list[float], target: float) -> float | None:
        for i in range(1, len(doses)):
            y0, y1 = responses[i - 1], responses[i]
            if (y0 <= target <= y1) or (y1 <= target <= y0):
                x0, x1 = doses[i - 1], doses[i]
                if y1 == y0:
                    return x0
                frac = (target - y0) / (y1 - y0)
                return x0 + frac * (x1 - x0)
        return None
