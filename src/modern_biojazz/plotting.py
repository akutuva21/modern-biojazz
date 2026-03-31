"""
Plotting utilities to generate manuscript-quality figures for modern_biojazz.
Designed to mimic and modernize figures from the original BioJazz paper (Feng et al. 2015).
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx

from .site_graph import ReactionNetwork

# Apply basic paper styling defaults
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})


def save_fig(fig: plt.Figure, filename: str, formats: List[str] = ["png", "svg"], out_dir: str = ".") -> None:
    """Helper to save a figure in multiple formats."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]
    for fmt in formats:
        path = os.path.join(out_dir, f"{base}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=300, transparent=False)


def plot_fitness_trajectory(
    evolution_results: List[Any],
    labels: Optional[List[str]] = None,
    target_fitness: float = 0.8,
) -> plt.Figure:
    """
    Plots the fitness trajectory over sampled mutations (Figure 4A analog).
    X-axis: total number of mutations sampled (log scale).
    Y-axis: fitness score [0, 1].
    Spacing between dots naturally encodes rejected mutations if we use sampled count.

    Args:
        evolution_results: List of EvolutionResult objects.
        labels: Optional labels for each run (or runs grouped by omega_c).
        target_fitness: Horizontal line indicating the target fitness.

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, res in enumerate(evolution_results):
        label = labels[idx] if labels else f"Run {idx + 1}"

        # In a real implementation mapping precisely to the paper, the EvolutionResult
        # would need to log the exact `mutations_sampled` for each acceptance.
        # Since our GenerationSummary tracks per-generation bests, we approximate
        # the X-axis by assuming a constant sampling rate per generation or using
        # `unique_population` cumulatively as a proxy for total sampled mutations.

        # Reconstruct cumulative sampled mutations as a proxy
        cumulative_sampled = []
        best_scores = []
        total_so_far = 0

        for gen in res.generation_summary:
            # We assume generation 0 is the seed or initial pop
            if gen.generation == 0:
                total_so_far += gen.unique_population
            else:
                # Approximate new sampled mutations
                total_so_far += gen.unique_population

            cumulative_sampled.append(max(1, total_so_far))
            best_scores.append(gen.best_score)

        ax.plot(cumulative_sampled, best_scores, marker='o', linestyle='-', markersize=4, alpha=0.8, label=label)

    ax.axhline(target_fitness, color="gray", linestyle="--", linewidth=1.5, label="Target Fitness")

    ax.set_xscale("log")
    ax.set_xlabel("Total Mutations Sampled (log scale)")
    ax.set_ylabel("Fitness Score")
    ax.set_title("Fitness Trajectory over Mutations")
    ax.set_ylim(-0.05, 1.05)

    if len(evolution_results) > 1 or labels:
        ax.legend()

    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_parameter_trajectory(
    k1_values: List[float],
    k2_values: List[float],
    response_amplitudes: List[float],
    ultrasensitivity_scores: List[float],
    generations: List[int],
) -> plt.Figure:
    """
    Plots the parameter trajectory in K1-K2 space (Figure 6 analog).
    X-axis: K1 (kinase catalytic efficiency, log scale).
    Y-axis: K2 (phosphatase catalytic efficiency, log scale).
    Dot size: response amplitude.
    Dot color: ultrasensitivity score.
    Arrows: indicate evolutionary direction.

    Args:
        k1_values: List of K1 values over time.
        k2_values: List of K2 values over time.
        response_amplitudes: List of response amplitudes to determine dot size.
        ultrasensitivity_scores: List of scores to determine dot color.
        generations: Generation number for each point to label them.

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Scale sizes for visual clarity
    # Base size + scaled response amplitude
    sizes = [40 + 200 * (r / max(response_amplitudes)) if max(response_amplitudes) > 0 else 50
             for r in response_amplitudes]

    scatter = ax.scatter(
        k1_values,
        k2_values,
        s=sizes,
        c=ultrasensitivity_scores,
        cmap='viridis',
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )

    # Add colorbar for ultrasensitivity
    cbar = fig.colorbar(scatter, ax=ax, label="Ultrasensitivity Score")

    # Draw arrows connecting consecutive points
    for i in range(len(k1_values) - 1):
        ax.annotate(
            "",
            xy=(k1_values[i+1], k2_values[i+1]),
            xytext=(k1_values[i], k2_values[i]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, linestyle="dashed")
        )

    # Label points with their generation number
    for i, gen in enumerate(generations):
        # Slightly offset the label so it doesn't overlap the dot center
        ax.text(k1_values[i], k2_values[i], f" Gen {gen}", fontsize=8, alpha=0.8,
                verticalalignment='bottom', horizontalalignment='left')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K1 (Kinase efficiency, log scale)")
    ax.set_ylabel("K2 (Phosphatase efficiency, log scale)")
    ax.set_title("Parameter Trajectory in K1-K2 Space")

    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_network_topology(network: ReactionNetwork, title: str = "Evolved Network Topology") -> plt.Figure:
    """
    Plots the network graph of a ReactionNetwork (Figure 5A analog).
    Nodes are proteins. Edges are interactions (rules).
    Edge colors:
      - black: binding
      - red: phosphorylation
      - blue: dephosphorylation
      - gray: other

    Args:
        network: ReactionNetwork object.
        title: Title of the plot.

    Returns:
        plt.Figure
    """
    G = nx.MultiDiGraph()

    # Add nodes (proteins)
    for pname in network.proteins:
        # Determine if protein has an allosteric/modification site (green border analog in paper)
        has_mod = any(s.site_type == "modification" for s in network.proteins[pname].sites)
        G.add_node(pname, has_mod=has_mod)

    # Extract edges from rules
    # For simplicity, we define an edge from the first reactant to the first product/target
    # A more rigorous parsing of BNGL rules might be needed for complex graphs,
    # but this heuristic works for basic BioJazz-style graphs.
    def extract_base_protein(token: str) -> str:
        """Helper to get base protein name from a string like A(site~p)."""
        return token.split('(')[0].split(':')[0]

    for rule in network.rules:
        if not rule.reactants:
            continue

        # Try to identify source and target
        # e.g., A + B -> A + B_p (A modifies B)
        # e.g., A + B -> A:B (binding)
        src = extract_base_protein(rule.reactants[0])
        target = src
        if len(rule.reactants) > 1:
            target = extract_base_protein(rule.reactants[1])

        edge_color = "gray"
        if rule.rule_type == "binding":
            edge_color = "black"
        elif rule.rule_type == "phosphorylation":
            edge_color = "red"
        elif rule.rule_type == "dephosphorylation":
            edge_color = "blue"

        # For modification, source is usually the modifier, target is the substrate
        if rule.rule_type in ("phosphorylation", "dephosphorylation") and len(rule.reactants) > 1:
            # First reactant is usually the enzyme, second is substrate
            src = extract_base_protein(rule.reactants[0])
            target = extract_base_protein(rule.reactants[1])
        elif rule.rule_type == "binding" and len(rule.reactants) > 1:
            src = extract_base_protein(rule.reactants[0])
            target = extract_base_protein(rule.reactants[1])

        if src in G.nodes and target in G.nodes:
            G.add_edge(src, target, color=edge_color, rule_type=rule.rule_type)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Layout
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # Draw nodes
    node_colors = ["#e0e0e0" for _ in G.nodes()]
    node_edgecolors = ["green" if G.nodes[n].get("has_mod") else "black" for n in G.nodes()]
    node_linewidths = [2.0 if G.nodes[n].get("has_mod") else 1.0 for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
        node_size=800
    )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_family="sans-serif")

    # Draw edges
    edges = G.edges(data=True)
    for color in ["black", "red", "blue", "gray"]:
        color_edges = [(u, v) for u, v, d in edges if d.get("color") == color]
        if color_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=color_edges,
                edge_color=color,
                ax=ax,
                arrows=True,
                arrowsize=15,
                connectionstyle="arc3,rad=0.2"
            )

    # Custom legend for edges and nodes
    legend_elements = [
        mpatches.Patch(color='black', label='Binding'),
        mpatches.Patch(color='red', label='Phosphorylation'),
        mpatches.Patch(color='blue', label='Dephosphorylation'),
        plt.Line2D([0], [0], marker='o', color='w', label='Allosteric/Mod Site',
                   markerfacecolor='#e0e0e0', markeredgecolor='green', markersize=10, markeredgewidth=2),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1))

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    return fig


def plot_simulation_dynamics(
    simulation_result: Dict[str, Any],
    species_to_plot: Optional[List[str]] = None,
    title: str = "Simulation Dynamics",
) -> plt.Figure:
    """
    Plots the time-series dynamics of a simulation result (Figure 3/5B analog).

    Args:
        simulation_result: The dictionary returned by `SimulationBackend.simulate`.
        species_to_plot: List of specific species names to plot. If None, plots top species or output.
        title: Title of the plot.

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    trajectory = simulation_result.get("trajectory", [])
    if not trajectory:
        ax.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
        return fig

    times = [pt["t"] for pt in trajectory]

    # Extract species data
    all_species_data: Dict[str, List[float]] = {}
    for pt in trajectory:
        species_map = pt.get("species", {})
        for sp, val in species_map.items():
            if sp not in all_species_data:
                all_species_data[sp] = []
            all_species_data[sp].append(val)

    # Decide what to plot
    if species_to_plot:
        plot_keys = [sp for sp in species_to_plot if sp in all_species_data]
    else:
        # Default to plotting output or top varying species
        # BioJazz specifically tracks an "output"
        outputs = [pt.get("output", 0.0) for pt in trajectory]
        if any(v > 0 for v in outputs):
            ax.plot(times, outputs, label="Output", color="green", linewidth=2)

        # Plot up to 5 other species with largest variance to avoid clutter
        variances = {sp: np.var(vals) for sp, vals in all_species_data.items()}
        top_sp = sorted(variances, key=variances.get, reverse=True)[:5]
        plot_keys = top_sp

    # Plot the selected species
    for sp in plot_keys:
        if sp in all_species_data:
            ax.plot(times, all_species_data[sp], label=sp, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration / Activity")
    ax.set_title(title)

    if len(ax.get_lines()) > 0:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_efficiency_bars(
    labels: List[str],
    avg_mutants_sampled: List[float],
    avg_wall_clock_time: List[float],
) -> plt.Figure:
    """
    Plots the efficiency bar charts (Figure 4B analog).
    Left panel: average number of mutants sampled before one is accepted.
    Right panel: average wall-clock simulation time per mutant evaluation.

    Args:
        labels: Group labels (e.g. complexity weights omega_c).
        avg_mutants_sampled: List of average mutants sampled values.
        avg_wall_clock_time: List of average wall-clock times.

    Returns:
        plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    x = np.arange(len(labels))
    width = 0.6

    # Left: Mutants sampled
    ax1.bar(x, avg_mutants_sampled, width, color="skyblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Average mutants sampled")
    ax1.set_title("Search Efficiency")

    # Right: Wall-clock time
    ax2.bar(x, avg_wall_clock_time, width, color="salmon")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Average evaluation time (s)")
    ax2.set_title("Computational Cost")

    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_mutational_effects(
    delta_ultrasensitivity: List[float],
    delta_fitness: List[float],
    labels: Optional[List[str]] = None,
) -> sns.JointGrid:
    """
    Plots the joint distribution of mutational effects (Figure 4D analog).
    Center: scatter of ΔFitness (y-axis) vs. ΔUltrasensitivity (x-axis)
    for all accepted mutations.
    Marginals: KDE on top and right edges.

    Args:
        delta_ultrasensitivity: List of ΔUltrasensitivity values.
        delta_fitness: List of ΔFitness values.
        labels: Optional group labels for hue grouping.

    Returns:
        sns.JointGrid
    """
    # Create a DataFrame for seaborn
    import pandas as pd

    data = {
        "ΔUltrasensitivity": delta_ultrasensitivity,
        "ΔFitness": delta_fitness
    }

    if labels and len(labels) == len(delta_fitness):
        data["Group"] = labels
        df = pd.DataFrame(data)
        g = sns.JointGrid(data=df, x="ΔUltrasensitivity", y="ΔFitness", hue="Group")
    else:
        df = pd.DataFrame(data)
        g = sns.JointGrid(data=df, x="ΔUltrasensitivity", y="ΔFitness")

    g.plot_joint(sns.scatterplot, s=30, alpha=0.6, edgecolor="white", linewidth=0.5)
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.5)

    # Add zero-lines to emphasize beneficial vs deleterious
    g.refline(y=0, color="gray", linestyle="--", alpha=0.5)
    g.refline(x=0, color="gray", linestyle="--", alpha=0.5)

    return g


def plot_evolutionary_space(
    evolution_results: List[Any],
    labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plots the complexity evolution (Figure 4C analog).
    X-axis: number of proteins (proxy for reactive sites).
    Y-axis: number of rules (interactions).
    Dot size: proportional to generation.

    Args:
        evolution_results: List of EvolutionResult objects.
        labels: Optional labels for each run.

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Define a color palette
    palette = sns.color_palette("husl", len(evolution_results))

    for idx, res in enumerate(evolution_results):
        label = labels[idx] if labels else f"Run {idx + 1}"
        color = palette[idx]

        generations = [max(1, gen.generation) for gen in res.generation_summary]
        proteins = [gen.best_n_proteins for gen in res.generation_summary]
        rules = [gen.best_n_rules for gen in res.generation_summary]

        # Scale sizes based on generation to show progression visually
        # Generation 0 might be small, so we scale it starting from a base size
        sizes = [30 + 5 * g for g in generations]

        scatter = ax.scatter(
            proteins,
            rules,
            s=sizes,
            c=[color] * len(proteins),
            alpha=0.6,
            edgecolors='white',
            linewidth=0.5,
            label=label
        )

        # Optional: draw an arrow or line connecting consecutive points
        for i in range(len(proteins) - 1):
            ax.annotate(
                "",
                xy=(proteins[i+1], rules[i+1]),
                xytext=(proteins[i], rules[i]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.3)
            )

    ax.set_xlabel("Number of Proteins (Nodes)")
    ax.set_ylabel("Number of Rules (Interactions)")
    ax.set_title("Evolutionary Space Trajectory")

    # Only use basic legend to show groups, avoiding big dots everywhere in legend
    if len(evolution_results) > 1 or labels:
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(handles, lbls, scatterpoints=1)

    sns.despine(fig)
    fig.tight_layout()
    return fig
