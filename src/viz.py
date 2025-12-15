from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .dynamics import CascadeResult
from .influence_game import Action, InfluenceGame

try:
    from pyvis.network import Network

    _HAS_PYVIS = True
except ImportError:  # pragma: no cover
    Network = object  # type: ignore
    _HAS_PYVIS = False


@dataclass
class LayoutCache:
    """Store node positions so multiple plots share a layout."""

    positions: Dict[Any, Tuple[float, float]]


def _compute_layout(
    game: InfluenceGame,
    layout_cache: Optional[LayoutCache] = None,
    layout: str = "spring",
    seed: int = 3210,
) -> LayoutCache:
    """Get or build node positions for plotting."""
    if layout_cache is not None:
        if set(layout_cache.positions.keys()) == set(game.nodes):
            return layout_cache

    graph = game.G

    if layout == "spring":
        pos = nx.spring_layout(graph, seed=seed)
    elif layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(graph)
        except Exception:
            # Kamada-Kawai needs SciPy; fall back to spring if it's missing or errors.
            pos = nx.spring_layout(graph, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "shell":
        pos = nx.shell_layout(graph)
    else:
        raise ValueError(f"Unknown layout '{layout}'")

    return LayoutCache(positions=pos)


def draw_profile_matplotlib(
    game: InfluenceGame,
    profile: Mapping[Any, Action],
    forcing_set: Optional[Iterable[Any]] = None,
    highlight_nodes: Optional[Iterable[Any]] = None,
    title: Optional[str] = None,
    layout_cache: Optional[LayoutCache] = None,
    layout: str = "spring",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, LayoutCache]:
    """
    Draw one action profile with matplotlib.
    """
    normalized = game.normalize_profile(profile)
    forcing_nodes: Set[Any] = set(forcing_set) if forcing_set is not None else set()
    highlight_nodes_set: Set[Any] = set(highlight_nodes) if highlight_nodes is not None else set()

    layout_cache = _compute_layout(game, layout_cache=layout_cache, layout=layout)
    positions = layout_cache.positions

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    node_colors: List[str] = []
    node_sizes: List[float] = []
    node_edge_colors: List[str] = []
    node_edge_widths: List[float] = []

    for node in game.nodes:
        action = normalized[node]
        color = "#d62728" if action == 1 else "#ffffff"
        size = 750 if node in highlight_nodes_set else 550
        edge_color = "#2c3e50"
        edge_width = 2.6 if node in forcing_nodes else 1.8

        node_colors.append(color)
        node_sizes.append(size)
        node_edge_colors.append(edge_color)
        node_edge_widths.append(edge_width)

    incoming_weight: Dict[Any, float] = {}
    for node in game.nodes:
        total = 0.0
        if game.directed:
            neighbors_iter = game.G.predecessors(node)
            for nbr in neighbors_iter:
                total += game.weight(nbr, node)
        else:
            for nbr in game.G.neighbors(node):
                total += game.weight(nbr, node)
        incoming_weight[node] = total

    edge_widths: List[float] = []
    for u, v in game.edges:
        weight = game.weight(u, v)
        edge_widths.append(0.5 + 1.5 * (weight / (1.0 + weight)))

    edge_kwargs = dict(
        ax=ax,
        width=edge_widths,
        alpha=0.7,
        arrows=game.directed,
    )
    if game.directed:
        edge_kwargs["connectionstyle"] = "arc3,rad=0.05"

    nx.draw_networkx_edges(game.G, positions, **edge_kwargs)

    nx.draw_networkx_nodes(
        game.G,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=node_edge_widths,
    )

    labels: Dict[Any, str] = {}
    for node in game.nodes:
        labels[node] = game.label(node) or str(node)

    nx.draw_networkx_labels(
        game.G,
        positions,
        labels=labels,
        font_size=10,
        ax=ax,
    )

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))

    for node, (x, y) in positions.items():
        theta = game.threshold(node)
        action = normalized[node]
        incoming = incoming_weight.get(node, 0.0)
        if incoming > 0 and theta != float("inf"):
            percent = 100.0 * theta / incoming
            percent_display = f"{percent:.1f}%"
        elif incoming == 0 and theta == 0:
            percent_display = "0% (no incoming)"
        elif incoming == 0 and theta == float("inf"):
            percent_display = "N/A (no incoming)"
        else:
            percent_display = "N/A"

        text = f"threshold={theta:g} ({percent_display}), action={action}"

        dx = x - center_x
        dy = y - center_y
        dist = np.hypot(dx, dy)
        if dist == 0.0:
            offset_x = 0.0
            offset_y = -0.12
        else:
            scale = 0.18
            offset_x = dx / dist * scale
            offset_y = dy / dist * scale

        ax.text(
            x + offset_x,
            y + offset_y,
            text,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="none",
                alpha=0.8,
            ),
        )

    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax, layout_cache


def draw_cascade_history_matplotlib(
    game: InfluenceGame,
    cascade: CascadeResult,
    forcing_set: Optional[Iterable[Any]] = None,
    layout_cache: Optional[LayoutCache] = None,
    layout: str = "spring",
    max_steps_to_plot: int = 4,
) -> Tuple[plt.Figure, LayoutCache]:
    """
    Draw a few frames from a cascade history side by side.
    """
    history = cascade.history
    total_steps = len(history) - 1

    if total_steps <= 0:
        fig, ax, layout_cache = draw_profile_matplotlib(
            game,
            history[0],
            forcing_set=forcing_set,
            title="t = 0",
            layout_cache=layout_cache,
            layout=layout,
        )
        return fig, layout_cache

    if total_steps + 1 <= max_steps_to_plot:
        steps_to_plot = list(range(total_steps + 1))
    else:
        indices = np.linspace(0, total_steps, num=max_steps_to_plot, dtype=int)
        steps_to_plot = sorted(set(int(i) for i in indices))

    num_frames = len(steps_to_plot)
    fig, axes = plt.subplots(1, num_frames, figsize=(4.0 * num_frames, 4.0))
    if num_frames == 1:
        axes = [axes]  # type: ignore

    layout_cache = _compute_layout(game, layout_cache=layout_cache, layout=layout)

    for axis, t in zip(axes, steps_to_plot):
        frame_title = f"t = {t}"
        if cascade.converged and t == steps_to_plot[-1]:
            frame_title += " (fixed point)"
        draw_profile_matplotlib(
            game,
            history[t],
            forcing_set=forcing_set,
            title=frame_title,
            layout_cache=layout_cache,
            layout=layout,
            ax=axis,
        )

    fig.tight_layout()
    return fig, layout_cache


def build_pyvis_network(
    game: InfluenceGame,
    profile: Optional[Mapping[Any, Action]] = None,
    forcing_set: Optional[Iterable[Any]] = None,
    highlight_nodes: Optional[Iterable[Any]] = None,
    directed: Optional[bool] = None,
) -> Network:
    """
    Build a pyvis Network to visualize an InfluenceGame.
    """
    if not _HAS_PYVIS:
        raise RuntimeError("pyvis is not installed. Install it to use this function.")

    if directed is None:
        directed = game.directed

    net = Network(directed=directed, notebook=False)
    net.barnes_hut()

    forcing_nodes: Set[Any] = set(forcing_set) if forcing_set is not None else set()
    highlight_nodes_set: Set[Any] = set(highlight_nodes) if highlight_nodes is not None else set()

    if profile is not None:
        profile_norm = game.normalize_profile(profile)
    else:
        profile_norm = {node: 0 for node in game.nodes}

    for node in game.nodes:
        action = profile_norm[node]
        label = game.label(node) or str(node)
        theta = game.threshold(node)

        color = "#d62728" if action == 1 else "#ffffff"
        size = 26 if node in highlight_nodes_set else 20
        border_width = 3 if node in forcing_nodes else 1

        title = f"Node: {label}<br>Threshold: {theta:g}<br>Action: {action}"

        net.add_node(
            n_id=str(node),
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
            borderWidthSelected=border_width + 1,
        )

    for u, v in game.edges:
        weight = game.weight(u, v)
        width = 1 + 3 * (weight / (1.0 + weight))
        title = f"Weight: {weight:g}"
        net.add_edge(str(u), str(v), value=weight, title=title, width=width)

    return net


if __name__ == "__main__":
    from .dynamics import CascadeSimulator

    game = InfluenceGame(directed=True)
    game.add_node("A", threshold=0.0, label="A")
    game.add_node("B", threshold=1.0, label="B")
    game.add_node("C", threshold=1.0, label="C")

    game.add_edge("A", "B", weight=1.0)
    game.add_edge("B", "C", weight=1.0)

    simulator = CascadeSimulator(game)
    initial = game.empty_profile(active_value=0)
    forcing_set = {"A"}
    fixed = {"A": 1}

    cascade_result = simulator.run_until_fixpoint(
        initial_profile=initial,
        fixed_actions=fixed,
        max_steps=10,
        detect_cycles=True,
    )

    print("Converged:", cascade_result.converged)
    print("Final profile:", cascade_result.final_profile)

    fig, layout_cache = draw_cascade_history_matplotlib(
        game,
        cascade=cascade_result,
        forcing_set=forcing_set,
        layout="spring",
        max_steps_to_plot=3,
    )
    fig.suptitle("Simple cascade A -> B -> C")
    plt.show()

    if _HAS_PYVIS:
        net = build_pyvis_network(
            game,
            profile=cascade_result.final_profile,
            forcing_set=forcing_set,
            highlight_nodes={"C"},
        )
        net.show("example_cascade.html")
