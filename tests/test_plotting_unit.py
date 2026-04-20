import os
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import MagicMock, call

from modern_biojazz.plotting import (
    save_fig,
    plot_fitness_trajectory,
    plot_parameter_trajectory,
    plot_network_topology,
    plot_simulation_dynamics,
    plot_efficiency_bars,
    plot_mutational_effects,
    plot_evolutionary_space,
)
from modern_biojazz.site_graph import ReactionNetwork, Protein, Site, Rule
from modern_biojazz.evolution import EvolutionResult, GenerationSummary

@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")

def test_save_fig(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])

    filename = "test_plot.png"
    out_dir = str(tmp_path)

    save_fig(fig, filename, formats=["png", "svg"], out_dir=out_dir)

    assert os.path.exists(os.path.join(out_dir, "test_plot.png"))
    assert os.path.exists(os.path.join(out_dir, "test_plot.svg"))

def test_save_fig_mocked_calls(tmp_path):
    fig = MagicMock(spec=plt.Figure)
    filename = "test_plot.png"
    out_dir = str(tmp_path)

    save_fig(fig, filename, formats=["png", "svg"], out_dir=out_dir)

    expected_calls = [
        call(os.path.join(out_dir, "test_plot.png"), bbox_inches="tight", dpi=300, transparent=False),
        call(os.path.join(out_dir, "test_plot.svg"), bbox_inches="tight", dpi=300, transparent=False),
    ]
    fig.savefig.assert_has_calls(expected_calls, any_order=False)
    assert fig.savefig.call_count == 2

def test_plot_fitness_trajectory():
    gen_sum1 = GenerationSummary(generation=0, best_score=0.1, best_n_proteins=2, best_n_rules=1, top_scores=[0.1], unique_population=10)
    gen_sum2 = GenerationSummary(generation=1, best_score=0.5, best_n_proteins=2, best_n_rules=2, top_scores=[0.5], unique_population=5)

    evo_res = EvolutionResult(best_network=ReactionNetwork(), best_score=0.5, history=[0.1, 0.5], generation_summary=[gen_sum1, gen_sum2])

    fig = plot_fitness_trajectory([evo_res], labels=["Run 1"])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0

def test_plot_parameter_trajectory():
    fig = plot_parameter_trajectory(
        k1_values=[0.1, 0.2],
        k2_values=[1.0, 2.0],
        response_amplitudes=[0.5, 0.8],
        ultrasensitivity_scores=[0.1, 0.9],
        generations=[0, 1]
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0

def test_plot_network_topology():
    network = ReactionNetwork(
        proteins={
            "A": Protein(name="A", sites=[Site(name="s", site_type="binding")]),
            "B": Protein(name="B", sites=[Site(name="s", site_type="modification", states=["U", "P"])])
        },
        rules=[
            Rule(name="bind", rule_type="binding", reactants=["A(s)", "B(s~U)"], products=["A(s!1).B(s~U!1)"], rate=1.0)
        ]
    )
    fig = plot_network_topology(network)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0

def test_plot_simulation_dynamics():
    sim_res = {
        "trajectory": [
            {"t": 0.0, "species": {"A": 1.0, "B": 0.0}, "output": 0.0},
            {"t": 1.0, "species": {"A": 0.5, "B": 0.5}, "output": 0.5}
        ]
    }
    fig = plot_simulation_dynamics(sim_res)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0

def test_plot_efficiency_bars():
    fig = plot_efficiency_bars(
        labels=["Group 1", "Group 2"],
        avg_mutants_sampled=[10.5, 15.2],
        avg_wall_clock_time=[1.2, 1.8]
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2

def test_plot_mutational_effects():
    grid = plot_mutational_effects(
        delta_ultrasensitivity=[0.1, -0.2, 0.5],
        delta_fitness=[0.05, -0.1, 0.3],
        labels=["A", "B", "A"]
    )
    assert isinstance(grid, sns.JointGrid)

def test_plot_evolutionary_space():
    gen_sum1 = GenerationSummary(generation=0, best_score=0.1, best_n_proteins=2, best_n_rules=1, top_scores=[0.1], unique_population=10)
    gen_sum2 = GenerationSummary(generation=1, best_score=0.5, best_n_proteins=3, best_n_rules=4, top_scores=[0.5], unique_population=5)

    evo_res = EvolutionResult(best_network=ReactionNetwork(), best_score=0.5, history=[0.1, 0.5], generation_summary=[gen_sum1, gen_sum2])

    fig = plot_evolutionary_space([evo_res], labels=["Run 1"])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
