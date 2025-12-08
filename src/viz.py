# src/viz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Set

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
    """
    Cache for node positions in plots.

    This class stores a mapping from node to 2D coordinates so that
    multiple plots of the same InfluenceGame use consistent layouts.
    """

    positions: Dict[Any, Tuple[float, float]]


def _compute_layout(
    game: InfluenceGame,
    layout_cache: Optional[LayoutCache] = None,
    layout: str = "spring",
    seed: int = 3210,
) -> LayoutCache:
    """
    Compute or reuse a node layout for plotting.

    Parameters
    ----------
    game:
        InfluenceGame whose graph will be drawn.
    layout_cache:
        Optional existing LayoutCache to reuse.
    layout:
        Layout algorithm name. Supported: "spring", "kamada_kawai",
        "circular", "shell".
    seed:
        Random seed for layouts that use randomness.

    Returns
    -------
    LayoutCache
        Cache object with node positions.
    """
    if layout_cache is not None:
        if set(layout_cache.positions.keys()) == set(game.nodes):
            return layout_cache

    G = game.G

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
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
    Draw a single action profile using matplotlib.

    Visual encoding:
      - Node fill color:
          1 (active) -> strong red.
          0 (inactive) -> white.
      - Node border:
          thicker if the node is in the forcing set.
      - Node size:
          larger if the node is in highlight_nodes.
      - Node label inside circle:
          human readable label only (for example "A" or "v0").
      - Info tag outside node:
          text "θ=..., a=..." drawn just outside the node with a small
          white background for readability.

    Edge widths are proportional to influence weights.

    Parameters
    ----------
    game:
        InfluenceGame instance to draw.
    profile:
        Mapping from node to action in {0, 1}.
    forcing_set:
        Optional iterable of nodes that are part of a forcing set.
    highlight_nodes:
        Optional iterable of nodes to draw with larger size.
    title:
        Optional title for the plot.
    layout_cache:
        Optional LayoutCache to reuse positions across plots.
    layout:
        Name of layout algorithm if layout_cache is not provided.
    ax:
        Optional matplotlib Axes to draw on. If None, a new figure and
        axes are created.

    Returns
    -------
    fig, ax, layout_cache:
        Matplotlib Figure and Axes, plus the LayoutCache used.
    """
    profile_norm = game.normalize_profile(profile)
    forcing_set_nodes: Set[Any] = set(forcing_set) if forcing_set is not None else set()
    highlight_nodes_set: Set[Any] = set(highlight_nodes) if highlight_nodes is not None else set()

    layout_cache = _compute_layout(game, layout_cache=layout_cache, layout=layout)
    pos = layout_cache.positions

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    node_colors: List[str] = []
    node_sizes: List[float] = []
    node_edge_colors: List[str] = []
    node_edge_widths: List[float] = []

    for node in game.nodes:
        action = profile_norm[node]

        if action == 1:
            color = "#d62728"  # strong red
        else:
            color = "#ffffff"  # white

        size = 750 if node in highlight_nodes_set else 550
        edge_color = "#2c3e50"
        edge_width = 2.6 if node in forcing_set_nodes else 1.8

        node_colors.append(color)
        node_sizes.append(size)
        node_edge_colors.append(edge_color)
        node_edge_widths.append(edge_width)

    edge_widths: List[float] = []
    for u, v in game.edges:
        w = game.weight(u, v)
        edge_widths.append(0.5 + 1.5 * (w / (1.0 + w)))

    edge_kwargs = dict(
        ax=ax,
        width=edge_widths,
        alpha=0.7,
        arrows=game.directed,
    )
    if game.directed:
        edge_kwargs["connectionstyle"] = "arc3,rad=0.05"

    nx.draw_networkx_edges(
        game.G,
        pos,
        **edge_kwargs,
    )

    nx.draw_networkx_nodes(
        game.G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=node_edge_widths,
    )

    # Main labels: just the display name
    labels: Dict[Any, str] = {}
    for node in game.nodes:
        display = game.label(node) or str(node)
        labels[node] = display

    nx.draw_networkx_labels(
        game.G,
        pos,
        labels=labels,
        font_size=10,
        ax=ax,
    )

    # Info tags: threshold and current action outside each node
    # We push the tag slightly away from the node so it does not overlap the circle.
    # Compute a rough "center" of the layout to define outward directions.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))

    for node, (x, y) in pos.items():
        theta = game.threshold(node)
        action = profile_norm[node]
        text = f"θ={theta:g}, a={action}"

        # Vector from layout center to node
        dx = x - center_x
        dy = y - center_y
        norm = np.hypot(dx, dy)
        if norm == 0.0:
            # Graph with a single node, just offset down a bit
            offset_x = 0.0
            offset_y = -0.12
        else:
            # Unit vector, then scaled to push text outward
            scale = 0.18
            offset_x = dx / norm * scale
            offset_y = dy / norm * scale

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
    Draw several snapshots from a cascade trajectory.

    This function selects up to max_steps_to_plot frames from the
    CascadeResult history and renders them side by side.

    Parameters
    ----------
    game:
        InfluenceGame instance whose cascade is being shown.
    cascade:
        CascadeResult from CascadeSimulator.run_until_fixpoint.
    forcing_set:
        Optional forcing set to outline in every frame.
    layout_cache:
        Optional LayoutCache to reuse positions.
    layout:
        Layout algorithm name if a new layout is computed.
    max_steps_to_plot:
        Maximum number of time steps to show. If the cascade is longer
        than this, frames are sampled across the trajectory.

    Returns
    -------
    fig, layout_cache:
        Matplotlib Figure and the LayoutCache used.
    """
    history = cascade.history
    T = len(history) - 1

    if T <= 0:
        fig, ax, layout_cache = draw_profile_matplotlib(
            game,
            history[0],
            forcing_set=forcing_set,
            title="t = 0",
            layout_cache=layout_cache,
            layout=layout,
        )
        return fig, layout_cache

    if T + 1 <= max_steps_to_plot:
        steps_to_plot = list(range(T + 1))
    else:
        idxs = np.linspace(0, T, num=max_steps_to_plot, dtype=int)
        steps_to_plot = sorted(set(int(i) for i in idxs))

    num_frames = len(steps_to_plot)
    fig, axes = plt.subplots(1, num_frames, figsize=(4.0 * num_frames, 4.0))
    if num_frames == 1:
        axes = [axes]  # type: ignore

    layout_cache = _compute_layout(game, layout_cache=layout_cache, layout=layout)

    for ax, t in zip(axes, steps_to_plot):
        title = f"t = {t}"
        if cascade.converged and t == steps_to_plot[-1]:
            title += " (fixed point)"
        draw_profile_matplotlib(
            game,
            history[t],
            forcing_set=forcing_set,
            title=title,
            layout_cache=layout_cache,
            layout=layout,
            ax=ax,
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
    Build a pyvis Network from an InfluenceGame and an optional profile.

    Node color and size encode the action profile if provided. Edge
    width encodes influence weight.

    Parameters
    ----------
    game:
        InfluenceGame to export.
    profile:
        Optional mapping from node to action in {0, 1}. If omitted, all
        nodes are drawn in a neutral style.
    forcing_set:
        Optional forcing set, nodes outlined distinctly.
    highlight_nodes:
        Optional set of nodes drawn larger.
    directed:
        Override whether to treat the network as directed in the
        visualization. If None, this uses game.directed.

    Returns
    -------
    Network
        Pyvis Network instance. Use network.show("file.html") to render.

    Raises
    ------
    RuntimeError
        If pyvis is not installed.
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

        if action == 1:
            color = "#d62728"
        else:
            color = "#ffffff"

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
        w = game.weight(u, v)
        width = 1 + 3 * (w / (1.0 + w))
        title = f"Weight: {w:g}"
        net.add_edge(str(u), str(v), value=w, title=title, width=width)

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