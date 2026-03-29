from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Any


class ReactionNetworkValidationError(ValueError):
    """Raised when a serialized reaction network payload is malformed."""


@dataclass
class Site:
    name: str
    site_type: str  # binding | modification
    states: List[str] = field(default_factory=list)
    allowed_partners: List[str] = field(default_factory=list)


@dataclass
class Protein:
    name: str
    sites: List[Site] = field(default_factory=list)


@dataclass
class Rule:
    name: str
    rule_type: str  # binding | phosphorylation | inhibition
    reactants: List[str]
    products: List[str]
    rate: float


@dataclass
class ReactionNetwork:
    proteins: Dict[str, Protein] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ReactionNetwork":
        proteins = {
            pname: Protein(
                name=p.name,
                sites=[
                    Site(
                        name=s.name,
                        site_type=s.site_type,
                        states=list(s.states),
                        allowed_partners=list(s.allowed_partners),
                    )
                    for s in p.sites
                ],
            )
            for pname, p in self.proteins.items()
        }
        rules = [
            Rule(
                name=r.name,
                rule_type=r.rule_type,
                reactants=list(r.reactants),
                products=list(r.products),
                rate=r.rate,
            )
            for r in self.rules
        ]
        return ReactionNetwork(proteins=proteins, rules=rules, metadata=deepcopy(self.metadata))

    def validate(self) -> None:
        for protein_name, protein in self.proteins.items():
            if protein.name != protein_name:
                raise ReactionNetworkValidationError(
                    f"Protein key '{protein_name}' does not match protein.name '{protein.name}'."
                )
            for site in protein.sites:
                if site.site_type not in {"binding", "modification"}:
                    raise ReactionNetworkValidationError(
                        f"Protein '{protein_name}' has site '{site.name}' with invalid site_type '{site.site_type}'."
                    )

        for idx, rule in enumerate(self.rules):
            if not rule.reactants:
                raise ReactionNetworkValidationError(f"Rule at index {idx} has no reactants.")
            if not rule.products:
                raise ReactionNetworkValidationError(f"Rule at index {idx} has no products.")
            if rule.rate < 0:
                raise ReactionNetworkValidationError(f"Rule '{rule.name}' has negative rate {rule.rate}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proteins": {
                pname: {
                    "name": p.name,
                    "sites": [
                        {
                            "name": s.name,
                            "site_type": s.site_type,
                            "states": s.states,
                            "allowed_partners": s.allowed_partners,
                        }
                        for s in p.sites
                    ],
                }
                for pname, p in self.proteins.items()
            },
            "rules": [
                {
                    "name": r.name,
                    "rule_type": r.rule_type,
                    "reactants": r.reactants,
                    "products": r.products,
                    "rate": r.rate,
                }
                for r in self.rules
            ],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ReactionNetwork":
        proteins_raw = payload.get("proteins", {})
        rules_raw = payload.get("rules", [])

        if not isinstance(proteins_raw, dict):
            raise ReactionNetworkValidationError("'proteins' must be a dictionary of protein definitions.")
        if not isinstance(rules_raw, list):
            raise ReactionNetworkValidationError("'rules' must be a list of rule definitions.")

        proteins: Dict[str, Protein] = {}
        for pname, payload_p in proteins_raw.items():
            if "name" not in payload_p:
                raise ReactionNetworkValidationError(f"Protein '{pname}' is missing required key 'name'.")
            sites: List[Site] = []
            for s in payload_p.get("sites", []):
                if "name" not in s or "site_type" not in s:
                    raise ReactionNetworkValidationError(
                        f"Protein '{pname}' has a site missing 'name' or 'site_type'."
                    )
                sites.append(
                    Site(
                        name=s["name"],
                        site_type=s["site_type"],
                        states=list(s.get("states", [])),
                        allowed_partners=list(s.get("allowed_partners", [])),
                    )
                )
            proteins[pname] = Protein(name=payload_p["name"], sites=sites)

        rules: List[Rule] = []
        for idx, r in enumerate(rules_raw):
            missing = [k for k in ["name", "rule_type", "reactants", "products", "rate"] if k not in r]
            if missing:
                raise ReactionNetworkValidationError(
                    f"Rule at index {idx} is missing required keys: {', '.join(missing)}"
                )
            rules.append(
                Rule(
                    name=r["name"],
                    rule_type=r["rule_type"],
                    reactants=list(r["reactants"]),
                    products=list(r["products"]),
                    rate=float(r["rate"]),
                )
            )

        network = ReactionNetwork(proteins=proteins, rules=rules, metadata=deepcopy(payload.get("metadata", {})))
        network.validate()
        return network
