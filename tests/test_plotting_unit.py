import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from modern_biojazz.plotting import (
    save_fig,
    plot_fitness_trajectory,
    plot_parameter_trajectory,
    plot_network_topology,
    plot_simulation_dynamics,
    plot_efficiency_bars,
    plot_mutational_effects,
    plot_evolutionary_space
)
from modern_biojazz.evolution import EvolutionResult, GenerationSummary
from modern_biojazz.site_graph import ReactionNetwork

def test_save_fig(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    save_fig(fig, "test_plot", formats=["png"], out_dir=str(tmp_path))
    assert (tmp_path / "test_plot.png").exists()

def test_plot_fitness_trajectory():
    gen1 = GenerationSummary(generation=0, best_score=0.1, best_n_proteins=2, best_n_rules=1, top_scores=[0.1], unique_population=10)
    gen2 = GenerationSummary(generation=1, best_score=0.5, best_n_proteins=2, best_n_rules=1, top_scores=[0.5], unique_population=5)
    res = EvolutionResult(best_network=ReactionNetwork({}, []), best_score=0.5, history=[0.1, 0.5], generation_summary=[gen1, gen2])
    fig = plot_fitness_trajectory([res, res], labels=["run1", "run2"])
    assert isinstance(fig, plt.Figure)

def test_plot_parameter_trajectory():
    fig = plot_parameter_trajectory([1.0, 2.0], [3.0, 4.0], [0.1, 0.2], [0.5, 0.6], [1, 2])
    assert isinstance(fig, plt.Figure)

def test_plot_network_topology():
    net = ReactionNetwork({}, [])
    fig = plot_network_topology(net)
    assert isinstance(fig, plt.Figure)

def test_plot_simulation_dynamics():
    sim_res = {
        "trajectory": [
            {"t": 0.0, "species": {"A": 1.0, "B": 0.0}, "output": 0.0},
            {"t": 1.0, "species": {"A": 0.0, "B": 1.0}, "output": 1.0}
        ]
    }
    fig = plot_simulation_dynamics(sim_res)
    assert isinstance(fig, plt.Figure)

def test_plot_efficiency_bars():
    fig = plot_efficiency_bars(["A", "B"], [10.0, 20.0], [1.0, 2.0])
    assert isinstance(fig, plt.Figure)

def test_plot_mutational_effects():
    g = plot_mutational_effects([0.1, -0.1], [0.2, -0.2])
    assert isinstance(g, sns.JointGrid)

def test_plot_evolutionary_space():
    gen1 = GenerationSummary(generation=0, best_score=0.1, best_n_proteins=2, best_n_rules=1, top_scores=[0.1], unique_population=10)
    gen2 = GenerationSummary(generation=1, best_score=0.5, best_n_proteins=3, best_n_rules=2, top_scores=[0.5], unique_population=5)
    res = EvolutionResult(best_network=ReactionNetwork({}, []), best_score=0.5, history=[0.1, 0.5], generation_summary=[gen1, gen2])
    fig = plot_evolutionary_space([res], labels=["run1"])
    assert isinstance(fig, plt.Figure)