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
        if token == old:
            return new
        if token == f"{old}_P":
            return f"{new}_P"
        if token == f"{old}_inh":
            return f"{new}_inh"
        if ":" in token:
            parts = token.split(":")
            return ":".join(new if p == old else p for p in parts)
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
        if len(network.proteins) <= 1:
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
        new_sites = list(protein.sites)
        new_sites.append(Site(name=site_name, site_type=site_type, states=["u", "p"] if site_type == "modification" else []))
        network.proteins[protein_name] = Protein(name=protein.name, sites=new_sites)

    def remove_site(self, network: ReactionNetwork, protein_name: str, site_name: str) -> None:
        protein = network.proteins.get(protein_name)
        if not protein:
            return
        new_sites = [s for s in protein.sites if s.name != site_name]
        network.proteins[protein_name] = Protein(name=protein.name, sites=new_sites)

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

    def _add_simple_rule(self, network: ReactionNetwork, rule_type: str, r1: str, r2: str, p2_suffix: str, rate: float) -> None:
        if r1 not in network.proteins or r2 not in network.proteins:
            return
        prefix = "phos" if rule_type == "phosphorylation" else "inh"
        rname = f"{prefix}_{r1}_{r2}_{len(network.rules)+1}"
        modified_r2 = f"{r2}{p2_suffix}"
        self._add_species_if_missing(network, modified_r2)
        network.rules.append(
            Rule(
                name=rname,
                rule_type=rule_type,
                reactants=[r1, r2],
                products=[r1, modified_r2],
                rate=rate,
            )
        )

    def add_phosphorylation_rule(self, network: ReactionNetwork, kinase: str, substrate: str, rate: float = 0.2) -> None:
        self._add_simple_rule(network, "phosphorylation", kinase, substrate, "_P", rate)

    def add_inhibition_rule(self, network: ReactionNetwork, inhibitor: str, target: str, rate: float = 0.05) -> None:
        self._add_simple_rule(network, "inhibition", inhibitor, target, "_inh", rate)

    def remove_rule(self, network: ReactionNetwork, rule_name: str) -> None:
        network.rules = [r for r in network.rules if r.name != rule_name]

    def add_dephosphorylation_rule(
        self, network: ReactionNetwork, phosphatase: str, substrate_phospho: str, rate: float = 0.15,
    ) -> None:
        """Add a reverse phosphorylation rule: phosphatase + substrate_P -> phosphatase + substrate."""
        if phosphatase not in network.proteins or substrate_phospho not in network.proteins:
            return
        if not substrate_phospho.endswith("_P"):
            return
        substrate = substrate_phospho[:-2]
        self._add_species_if_missing(network, substrate)
        rname = f"dephos_{phosphatase}_{substrate}_{len(network.rules)+1}"
        network.rules.append(
            Rule(
                name=rname,
                rule_type="dephosphorylation",
                reactants=[phosphatase, substrate_phospho],
                products=[phosphatase, substrate],
                rate=rate,
            )
        )

    def add_unbinding_rule(self, network: ReactionNetwork, complex_name: str, rate: float = 0.3) -> None:
        """Add a dissociation rule: A:B -> A + B."""
        if complex_name not in network.proteins:
            return
        if ":" not in complex_name:
            return
        parts = complex_name.split(":")
        for part in parts:
            self._add_species_if_missing(network, part)
        rname = f"unbind_{complex_name}_{len(network.rules)+1}"
        network.rules.append(
            Rule(
                name=rname,
                rule_type="unbinding",
                reactants=[complex_name],
                products=parts,
                rate=rate,
            )
        )

    def modify_rate(self, network: ReactionNetwork, rule_name: str, multiplier: float) -> None:
        for i, r in enumerate(network.rules):
            if r.name == rule_name:
                new_rate = r.rate * multiplier
                network.rules[i] = Rule(
                    name=r.name,
                    rule_type=r.rule_type,
                    reactants=r.reactants,
                    products=r.products,
                    rate=min(100.0, max(1e-6, new_rate))
                )
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
                # Add a new site
                new_sites = list(protein.sites)
                new_sites.append(Site(name=f"b_{partner}", site_type="binding", allowed_partners=[partner]))
                network.proteins[p] = Protein(name=protein.name, sites=new_sites)
            else:
                if partner not in bind_sites[0].allowed_partners:
                    # Replace the site rather than mutating allowed_partners in place
                    new_sites = list(protein.sites)
                    target_site = bind_sites[0]
                    new_partners = list(target_site.allowed_partners)
                    new_partners.append(partner)

                    for i, s in enumerate(new_sites):
                        if s.name == target_site.name:
                            new_sites[i] = Site(name=s.name, site_type=s.site_type, states=s.states, allowed_partners=new_partners)
                            break
                    network.proteins[p] = Protein(name=protein.name, sites=new_sites)

    def add_kinase_cascade(self, network: ReactionNetwork) -> None:
        """Adds a 3-tier kinase cascade A -> B -> C."""
        p1 = f"M_K1_{self.rng.randint(1, 9999)}"
        p2 = f"M_K2_{self.rng.randint(1, 9999)}"
        p3 = f"M_K3_{self.rng.randint(1, 9999)}"
        for p in [p1, p2, p3]:
            self.add_protein(network, p)
        self.add_phosphorylation_rule(network, p1, p2, rate=0.5)
        self.add_phosphorylation_rule(network, p2, p3, rate=0.5)

    def add_negative_feedback_loop(self, network: ReactionNetwork) -> None:
        """Adds a feedback loop A -> B -> C -| A."""
        p1 = f"M_F1_{self.rng.randint(1, 9999)}"
        p2 = f"M_F2_{self.rng.randint(1, 9999)}"
        p3 = f"M_F3_{self.rng.randint(1, 9999)}"
        for p in [p1, p2, p3]:
            self.add_protein(network, p)
        self.add_phosphorylation_rule(network, p1, p2, rate=0.5)
        self.add_phosphorylation_rule(network, p2, p3, rate=0.5)
        self.add_inhibition_rule(network, p3, p1, rate=0.5)

    def crossover(self, net1: ReactionNetwork, net2: ReactionNetwork) -> ReactionNetwork:
        """Structural crossover of two networks. Returns a new child network."""
        import copy
        child = net1.copy()
        if not net2.proteins:
            return child

        # Add a random subset of proteins from net2
        proteins_to_copy = self.rng.sample(list(net2.proteins.keys()), k=max(1, len(net2.proteins) // 2))
        for p_name in proteins_to_copy:
            if p_name not in child.proteins:
                child.proteins[p_name] = copy.deepcopy(net2.proteins[p_name])

        def all_bases_present(tokens: list[str]) -> bool:
            for t in tokens:
                if ":" in t:
                    if not all(p in child.proteins for p in t.split(":")):
                        return False
                else:
                    base = t[:-2] if t.endswith("_P") else (t[:-4] if t.endswith("_inh") else t)
                    if base not in child.proteins and t not in child.proteins:
                        return False
            return True

        # Copy over rules whose proteins are present in child
        for rule in net2.rules:
            if self.rng.random() < 0.5 and all_bases_present(rule.reactants) and all_bases_present(rule.products):
                if not any(r.name == rule.name for r in child.rules):
                    child.rules.append(copy.deepcopy(rule))

        return child

    def random_add_site(self, net: ReactionNetwork) -> None:
        if not net.proteins:
            self.add_protein(net)
        target = self.rng.choice(list(net.proteins.keys()))
        site_type = self.rng.choice(["binding", "modification"])
        self.add_site(net, target, f"s{self.rng.randint(1, 999)}", site_type)

    def random_bind(self, net: ReactionNetwork) -> None:
        if len(net.proteins) < 2:
            self.add_protein(net)
            self.add_protein(net)
        names = list(net.proteins.keys())
        a, b = self.rng.sample(names, 2)
        self._ensure_binding_sites(net, a, b)
        self.add_binding_rule(net, a, b)

    def random_phos(self, net: ReactionNetwork) -> None:
        if len(net.proteins) < 2:
            self.add_protein(net)
            self.add_protein(net)
        names = list(net.proteins.keys())
        k, s = self.rng.sample(names, 2)
        self.add_phosphorylation_rule(net, k, s)

    def random_inhibit(self, net: ReactionNetwork) -> None:
        if len(net.proteins) < 2:
            self.add_protein(net)
            self.add_protein(net)
        names = list(net.proteins.keys())
        i, t = self.rng.sample(names, 2)
        self.add_inhibition_rule(net, i, t)

    def random_remove_rule(self, net: ReactionNetwork) -> None:
        if not net.rules:
            return
        target = self.rng.choice(net.rules)
        self.remove_rule(net, target.name)

    def random_modify_rate(self, net: ReactionNetwork) -> None:
        if not net.rules:
            return
        target = self.rng.choice(net.rules)
        # Log-uniform jitter around 1.0 keeps multiplicative updates stable.
        multiplier = 10 ** self.rng.uniform(-0.2, 0.2)
        self.modify_rate(net, target.name, multiplier)

    def random_remove_site(self, net: ReactionNetwork) -> None:
        candidates = [(p.name, s.name) for p in net.proteins.values() for s in p.sites]
        if not candidates:
            return
        pname, sname = self.rng.choice(candidates)
        self.remove_site(net, pname, sname)

    def random_duplicate(self, net: ReactionNetwork) -> None:
        if not net.proteins:
            self.add_protein(net)
        target = self.rng.choice(list(net.proteins.keys()))
        self.duplicate_protein_with_rewiring(net, target)

    def random_remove_protein(self, net: ReactionNetwork) -> None:
        if len(net.proteins) <= 1:
            return
        target = self.rng.choice(list(net.proteins.keys()))
        self.remove_protein(net, target)

    def random_dephos(self, net: ReactionNetwork) -> None:
        phospho_species = [name for name in net.proteins.keys() if name.endswith("_P")]
        if not phospho_species:
            return
        substrate_p = self.rng.choice(phospho_species)
        candidates = [n for n in net.proteins.keys() if n != substrate_p and not n.endswith("_P")]
        if not candidates:
            return
        phosphatase = self.rng.choice(candidates)
        self.add_dephosphorylation_rule(net, phosphatase, substrate_p)

    def random_unbind(self, net: ReactionNetwork) -> None:
        complexes = [name for name in net.proteins.keys() if ":" in name]
        if not complexes:
            return
        target = self.rng.choice(complexes)
        self.add_unbinding_rule(net, target)

    def action_library(self, network: ReactionNetwork) -> Dict[str, MutationAction]:
        return {
            "add_site": MutationAction("add_site", self.random_add_site),
            "add_kinase_cascade": MutationAction("add_kinase_cascade", self.add_kinase_cascade),
            "add_feedback_loop": MutationAction("add_feedback_loop", self.add_negative_feedback_loop),
            "add_binding": MutationAction("add_binding", self.random_bind),
            "add_phosphorylation": MutationAction("add_phosphorylation", self.random_phos),
            "add_dephosphorylation": MutationAction("add_dephosphorylation", self.random_dephos),
            "add_inhibition": MutationAction("add_inhibition", self.random_inhibit),
            "add_unbinding": MutationAction("add_unbinding", self.random_unbind),
            "remove_rule": MutationAction("remove_rule", self.random_remove_rule),
            "modify_rate": MutationAction("modify_rate", self.random_modify_rate),
            "remove_site": MutationAction("remove_site", self.random_remove_site),
            "duplicate_protein": MutationAction("duplicate_protein", self.random_duplicate),
            "remove_protein": MutationAction("remove_protein", self.random_remove_protein),
        }
