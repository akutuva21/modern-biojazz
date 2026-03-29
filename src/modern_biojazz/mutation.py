from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict

from .site_graph import ReactionNetwork, Protein, Site, Rule


@dataclass
class MutationAction:
    name: str
    apply: Callable[[ReactionNetwork], None]


class GraphMutator:
    """Graph-level mutation operators with lightweight biological constraints."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def _add_species_if_missing(self, network: ReactionNetwork, species_name: str) -> None:
        if species_name not in network.proteins:
            network.proteins[species_name] = Protein(name=species_name, sites=[])

    def _replace_species_token(self, token: str, old: str, new: str) -> str:
        token = token.replace(f"{old}:", f"{new}:")
        token = token.replace(f":{old}", f":{new}")
        if token == old:
            return new
        if token == f"{old}_P":
            return f"{new}_P"
        return token

    def _binding_is_compatible(self, network: ReactionNetwork, a: str, b: str) -> bool:
        a_sites = [s for s in network.proteins[a].sites if s.site_type == "binding"]
        b_sites = [s for s in network.proteins[b].sites if s.site_type == "binding"]
        if not a_sites or not b_sites:
            return False
        a_ok = any((not a_site.allowed_partners) or (b in a_site.allowed_partners) for a_site in a_sites)
        b_ok = any((not b_site.allowed_partners) or (a in b_site.allowed_partners) for b_site in b_sites)
        return a_ok and b_ok

    def _token_references_protein(self, token: str, protein_name: str) -> bool:
        if token == protein_name:
            return True
        if token == f"{protein_name}_P" or token == f"{protein_name}_inh":
            return True
        if ":" in token and protein_name in token.split(":"):
            return True
        return False

    def add_protein(self, network: ReactionNetwork, protein_name: str | None = None) -> None:
        protein_name = protein_name or f"P{len(network.proteins) + 1}"
        if protein_name in network.proteins:
            return
        network.proteins[protein_name] = Protein(name=protein_name, sites=[])

    def remove_protein(self, network: ReactionNetwork, protein_name: str) -> None:
        if protein_name not in network.proteins:
            return
        del network.proteins[protein_name]
        network.rules = [
            r
            for r in network.rules
            if not any(self._token_references_protein(tok, protein_name) for tok in [*r.reactants, *r.products])
        ]

    def add_site(self, network: ReactionNetwork, protein_name: str, site_name: str, site_type: str) -> None:
        protein = network.proteins.get(protein_name)
        if not protein:
            return
        if any(s.name == site_name for s in protein.sites):
            return
        protein.sites.append(Site(name=site_name, site_type=site_type, states=["u", "p"] if site_type == "modification" else []))

    def remove_site(self, network: ReactionNetwork, protein_name: str, site_name: str) -> None:
        protein = network.proteins.get(protein_name)
        if not protein:
            return
        protein.sites = [s for s in protein.sites if s.name != site_name]

    def add_binding_rule(self, network: ReactionNetwork, a: str, b: str, rate: float = 0.1) -> None:
        if a not in network.proteins or b not in network.proteins:
            return
        if not self._binding_is_compatible(network, a, b):
            return
        rname = f"bind_{a}_{b}_{len(network.rules)+1}"
        complex_species = f"{a}:{b}"
        self._add_species_if_missing(network, complex_species)
        network.rules.append(
            Rule(name=rname, rule_type="binding", reactants=[a, b], products=[complex_species], rate=rate)
        )

    def add_phosphorylation_rule(self, network: ReactionNetwork, kinase: str, substrate: str, rate: float = 0.2) -> None:
        if kinase not in network.proteins or substrate not in network.proteins:
            return
        rname = f"phos_{kinase}_{substrate}_{len(network.rules)+1}"
        phospho = f"{substrate}_P"
        self._add_species_if_missing(network, phospho)
        network.rules.append(
            Rule(
                name=rname,
                rule_type="phosphorylation",
                reactants=[kinase, substrate],
                products=[kinase, phospho],
                rate=rate,
            )
        )

    def add_inhibition_rule(self, network: ReactionNetwork, inhibitor: str, target: str, rate: float = 0.05) -> None:
        if inhibitor not in network.proteins or target not in network.proteins:
            return
        rname = f"inh_{inhibitor}_{target}_{len(network.rules)+1}"
        inhibited = f"{target}_inh"
        self._add_species_if_missing(network, inhibited)
        network.rules.append(
            Rule(
                name=rname,
                rule_type="inhibition",
                reactants=[inhibitor, target],
                products=[inhibitor, inhibited],
                rate=rate,
            )
        )

    def remove_rule(self, network: ReactionNetwork, rule_name: str) -> None:
        network.rules = [r for r in network.rules if r.name != rule_name]

    def modify_rate(self, network: ReactionNetwork, rule_name: str, multiplier: float) -> None:
        for r in network.rules:
            if r.name == rule_name:
                new_rate = r.rate * multiplier
                r.rate = min(100.0, max(1e-6, new_rate))
                return

    def duplicate_protein_with_rewiring(self, network: ReactionNetwork, protein_name: str) -> None:
        if protein_name not in network.proteins:
            return
        source = network.proteins[protein_name]
        duplicate_name = f"{protein_name}_dup{self.rng.randint(1, 9999)}"
        network.proteins[duplicate_name] = Protein(
            name=duplicate_name,
            sites=[
                Site(
                    s.name,
                    s.site_type,
                    states=list(s.states),
                    allowed_partners=[duplicate_name if p == protein_name else p for p in s.allowed_partners],
                )
                for s in source.sites
            ],
        )
        affected = [r for r in network.rules if protein_name in r.reactants or protein_name in r.products]
        for rule in affected[:2]:
            new_rule = Rule(
                name=f"{rule.name}_rewired_{duplicate_name}",
                rule_type=rule.rule_type,
                reactants=[self._replace_species_token(x, protein_name, duplicate_name) for x in rule.reactants],
                products=[self._replace_species_token(x, protein_name, duplicate_name) for x in rule.products],
                rate=rule.rate,
            )
            network.rules.append(new_rule)

    def _ensure_binding_sites(self, network: ReactionNetwork, a: str, b: str) -> None:
        for p, partner in [(a, b), (b, a)]:
            protein = network.proteins[p]
            bind_sites = [s for s in protein.sites if s.site_type == "binding"]
            if not bind_sites:
                protein.sites.append(Site(name=f"b_{partner}", site_type="binding", allowed_partners=[partner]))
            else:
                if partner not in bind_sites[0].allowed_partners:
                    bind_sites[0].allowed_partners.append(partner)

    def action_library(self, network: ReactionNetwork) -> Dict[str, MutationAction]:
        def random_add_site(net: ReactionNetwork) -> None:
            if not net.proteins:
                self.add_protein(net)
            target = self.rng.choice(list(net.proteins.keys()))
            site_type = self.rng.choice(["binding", "modification"])
            self.add_site(net, target, f"s{self.rng.randint(1, 999)}", site_type)

        def random_bind(net: ReactionNetwork) -> None:
            if len(net.proteins) < 2:
                self.add_protein(net)
                self.add_protein(net)
            names = list(net.proteins.keys())
            a, b = self.rng.sample(names, 2)
            self._ensure_binding_sites(net, a, b)
            self.add_binding_rule(net, a, b)

        def random_phos(net: ReactionNetwork) -> None:
            if len(net.proteins) < 2:
                self.add_protein(net)
                self.add_protein(net)
            names = list(net.proteins.keys())
            k, s = self.rng.sample(names, 2)
            self.add_phosphorylation_rule(net, k, s)

        def random_inhibit(net: ReactionNetwork) -> None:
            if len(net.proteins) < 2:
                self.add_protein(net)
                self.add_protein(net)
            names = list(net.proteins.keys())
            i, t = self.rng.sample(names, 2)
            self.add_inhibition_rule(net, i, t)

        def random_remove_rule(net: ReactionNetwork) -> None:
            if not net.rules:
                return
            target = self.rng.choice(net.rules)
            self.remove_rule(net, target.name)

        def random_modify_rate(net: ReactionNetwork) -> None:
            if not net.rules:
                return
            target = self.rng.choice(net.rules)
            # Log-uniform jitter around 1.0 keeps multiplicative updates stable.
            multiplier = 10 ** self.rng.uniform(-0.2, 0.2)
            self.modify_rate(net, target.name, multiplier)

        def random_remove_site(net: ReactionNetwork) -> None:
            candidates = [(p.name, s.name) for p in net.proteins.values() for s in p.sites]
            if not candidates:
                return
            pname, sname = self.rng.choice(candidates)
            self.remove_site(net, pname, sname)

        def random_duplicate(net: ReactionNetwork) -> None:
            if not net.proteins:
                self.add_protein(net)
            target = self.rng.choice(list(net.proteins.keys()))
            self.duplicate_protein_with_rewiring(net, target)

        def random_remove_protein(net: ReactionNetwork) -> None:
            if not net.proteins:
                return
            target = self.rng.choice(list(net.proteins.keys()))
            self.remove_protein(net, target)

        return {
            "add_site": MutationAction("add_site", random_add_site),
            "add_binding": MutationAction("add_binding", random_bind),
            "add_phosphorylation": MutationAction("add_phosphorylation", random_phos),
            "add_inhibition": MutationAction("add_inhibition", random_inhibit),
            "remove_rule": MutationAction("remove_rule", random_remove_rule),
            "modify_rate": MutationAction("modify_rate", random_modify_rate),
            "remove_site": MutationAction("remove_site", random_remove_site),
            "duplicate_protein": MutationAction("duplicate_protein", random_duplicate),
            "remove_protein": MutationAction("remove_protein", random_remove_protein),
        }
