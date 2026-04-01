from __future__ import annotations

from modern_biojazz.rate_optimizer import optimize_rates, DEConfig
from modern_biojazz.simulation import LocalCatalystEngine, FitnessEvaluator
from modern_biojazz.site_graph import ReactionNetwork, Protein, Rule, Site


def _simple_network() -> ReactionNetwork:
    return ReactionNetwork(
        proteins={
            "A": Protein(name="A", sites=[Site("mod", "modification", states=["u", "p"])]),
            "B": Protein(name="B", sites=[]),
            "A_P": Protein(name="A_P", sites=[]),
        },
        rules=[
            Rule(name="phos", rule_type="phosphorylation", reactants=["B", "A"], products=["B", "A_P"], rate=0.001),
            Rule(name="decay", rule_type="reaction", reactants=["A_P"], products=["A"], rate=0.5),
        ],
        metadata={"output_species": "A_P"},
    )


def test_de_improves_on_bad_initial_rates():
    net = _simple_network()
    backend = LocalCatalystEngine()
    evaluator = FitnessEvaluator(target_output=0.5)

    def objective(candidate: ReactionNetwork) -> float:
        return evaluator.score(backend=backend, network=candidate, t_end=10.0, dt=1.0)

    initial_score = objective(net)

    result = optimize_rates(
        network=net,
        objective=objective,
        config=DEConfig(max_eval=200, pop_size=10, seed=7),
    )

    assert result.n_eval > 0
    assert result.generations > 0
    assert result.best_score >= initial_score
    assert len(result.best_rates) == 2
    assert result.best_network is not net  # Should be a copy


def test_de_does_not_mutate_input():
    net = _simple_network()
    original_rates = [r.rate for r in net.rules]
    backend = LocalCatalystEngine()

    result = optimize_rates(
        network=net,
        objective=lambda n: FitnessEvaluator(1.0).score(backend=backend, network=n, t_end=5.0, dt=1.0),
        config=DEConfig(max_eval=50, pop_size=5, seed=1),
    )

    assert [r.rate for r in net.rules] == original_rates


def test_de_handles_zero_rules():
    net = ReactionNetwork(proteins={"A": Protein(name="A", sites=[])}, rules=[])
    backend = LocalCatalystEngine()

    result = optimize_rates(
        network=net,
        objective=lambda n: 0.0,
        config=DEConfig(max_eval=10, seed=1),
    )

    assert result.stop_reason == "no_rates"
    assert result.n_eval == 0


def test_optimize_rates_empty_network():
    net = ReactionNetwork(proteins={}, rules=[])
    backend = LocalCatalystEngine()

    result = optimize_rates(
        network=net,
        objective=lambda n: 0.0,
    )

    assert result.best_network is not net
    assert result.best_network.proteins == net.proteins
    assert result.best_network.rules == net.rules


def test_de_history_is_monotonic():
    net = _simple_network()
    backend = LocalCatalystEngine()

    result = optimize_rates(
        network=net,
        objective=lambda n: FitnessEvaluator(1.0).score(backend=backend, network=n, t_end=5.0, dt=1.0),
        config=DEConfig(max_eval=100, pop_size=8, seed=42),
    )

    # History should be non-decreasing (tracking best).
    for i in range(1, len(result.history)):
        assert result.history[i] >= result.history[i - 1] - 1e-12, (
            f"History decreased at index {i}: {result.history[i-1]} -> {result.history[i]}"
        )


def test_de_rates_within_bounds():
    net = _simple_network()
    backend = LocalCatalystEngine()
    cfg = DEConfig(max_eval=100, pop_size=8, seed=3, log10_lower=-4.0, log10_upper=1.0)

    result = optimize_rates(
        network=net,
        objective=lambda n: FitnessEvaluator(1.0).score(backend=backend, network=n, t_end=5.0, dt=1.0),
        config=cfg,
    )

    for rate in result.best_rates:
        assert 1e-4 <= rate <= 10.0, f"Rate {rate} outside bounds [1e-4, 10]"
