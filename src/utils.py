# src/utils.py
from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from pathlib import Path

import networkx as nx

try:
    from .influence_game import Action, InfluenceGame
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.influence_game import Action, InfluenceGame


def profile_to_string(
    game: InfluenceGame,
    profile: Mapping[Any, Action],
    sort_nodes: bool = True,
) -> str:
    """
    Convert a profile to a compact, readable string.

    Example format:
        A:1 B:0 C:1

    Parameters
    ----------
    game:
        InfluenceGame whose node set defines the domain.
    profile:
        Mapping from node to action in {0, 1}. Missing nodes are treated as 0.
    sort_nodes:
        If True, nodes are printed in sorted order. Otherwise, they are printed
        in the internal game order.

    Returns
    -------
    str
        Human friendly representation of the profile.
    """
    norm = game.normalize_profile(profile)
    nodes = list(game.nodes)
    if sort_nodes:
        nodes.sort()

    parts: List[str] = []
    for node in nodes:
        parts.append(f"{node}:{norm[node]}")
    return " ".join(parts)


def build_complete_symmetric_game(
    n: int,
    threshold: float,
    weight: float = 1.0,
    directed: bool = False,
    label_prefix: str = "v",
) -> InfluenceGame:
    """
    Build a complete graph influence game with symmetric thresholds
    and edge weights.

    Parameters
    ----------
    n:
        Number of nodes.
    threshold:
        Threshold for each node.
    weight:
        Influence weight on each edge.
    directed:
        If True, builds a directed complete graph. Otherwise undirected.
    label_prefix:
        Prefix for node labels. Nodes are named 0..n-1 internally and
        labels are label_prefix + index.

    Returns
    -------
    InfluenceGame
        The constructed symmetric game.
    """
    if n <= 0:
        raise ValueError("Number of nodes must be positive")
    if threshold < 0:
        raise ValueError("Threshold must be nonnegative")
    if weight < 0:
        raise ValueError("Weight must be nonnegative")

    game = InfluenceGame(directed=directed)

    for i in range(n):
        node = i
        label = f"{label_prefix}{i}"
        game.add_node(node, threshold=threshold, label=label)

    if directed:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                game.add_edge(i, j, weight=weight)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                game.add_edge(i, j, weight=weight)

    return game


def build_line_graph_game(
    n: int,
    thresholds: Optional[Mapping[int, float]] = None,
    default_threshold: float = 1.0,
    weight: float = 1.0,
    directed: bool = True,
    label_prefix: str = "v",
) -> InfluenceGame:
    """
    Build a line graph influence game on n nodes.

    Nodes are 0, 1, ..., n-1 arranged in a path.

    For directed=True:
        edges are 0 -> 1 -> 2 -> ... -> n-1

    For directed=False:
        edges are undirected between consecutive nodes.

    Parameters
    ----------
    n:
        Number of nodes in the path.
    thresholds:
        Optional mapping from node index to threshold. Nodes not present
        in this mapping use default_threshold.
    default_threshold:
        Threshold used when thresholds does not specify a particular node.
    weight:
        Weight on each edge.
    directed:
        Whether to treat edges as directed in a forward chain.
    label_prefix:
        Prefix for node labels.

    Returns
    -------
    InfluenceGame
        The constructed line graph game.
    """
    if n <= 0:
        raise ValueError("Number of nodes must be positive")
    if default_threshold < 0:
        raise ValueError("Thresholds must be nonnegative")
    if weight < 0:
        raise ValueError("Weight must be nonnegative")

    thresholds = thresholds or {}
    game = InfluenceGame(directed=directed)

    for i in range(n):
        theta = thresholds.get(i, default_threshold)
        label = f"{label_prefix}{i}"
        game.add_node(i, threshold=theta, label=label)

    if directed:
        for i in range(n - 1):
            game.add_edge(i, i + 1, weight=weight)
    else:
        for i in range(n - 1):
            game.add_edge(i, i + 1, weight=weight)

    return game


def build_random_threshold_game(
    num_nodes: int,
    edge_prob: float,
    weight_range: Tuple[float, float] = (0.5, 1.5),
    threshold_range: Tuple[float, float] = (0.5, 2.0),
    directed: bool = True,
    seed: Optional[int] = None,
    label_prefix: str = "v",
) -> InfluenceGame:
    """
    Build a random influence game on an Erdos-Renyi graph.

    Parameters
    ----------
    num_nodes:
        Number of nodes.
    edge_prob:
        Probability that an edge exists between two nodes. If directed is
        True, this is for each ordered pair (i, j) with i != j.
    weight_range:
        Range [low, high] from which edge weights are drawn uniformly.
    threshold_range:
        Range [low, high] from which thresholds are drawn uniformly.
    directed:
        If True, use a directed Erdos-Renyi model. Otherwise undirected.
    seed:
        Optional random seed for reproducibility.
    label_prefix:
        Prefix for node labels. Defaults to "v".

    Returns
    -------
    InfluenceGame
        Random influence game instance.
    """
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be between 0 and 1")

    rng = random.Random(seed)
    game = InfluenceGame(directed=directed)

    theta_low, theta_high = threshold_range
    w_low, w_high = weight_range

    for i in range(num_nodes):
        theta = rng.uniform(theta_low, theta_high)
        game.add_node(i, threshold=theta, label=f"{label_prefix}{i}")

    if directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if rng.random() <= edge_prob:
                    w = rng.uniform(w_low, w_high)
                    game.add_edge(i, j, weight=w)
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if rng.random() <= edge_prob:
                    w = rng.uniform(w_low, w_high)
                    game.add_edge(i, j, weight=w)

    return game


def build_custom_game(
    num_nodes: int,
    thresholds: List[float],
    adjacency: List[List[float]],
    *,
    directed: bool = True,
    label_prefix: str = "",
) -> InfluenceGame:
    """
    Build an InfluenceGame from percentage thresholds and adjacency.

    Threshold inputs are interpreted as percentages of the total
    possible incoming influence for each node. They are converted to
    absolute thresholds using the provided adjacency matrix. Nodes
    with no incoming influence get θ = 0 if their percentage is 0,
    otherwise θ = ∞ so they cannot be activated by neighbors.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if len(thresholds) != num_nodes:
        raise ValueError("thresholds must have length num_nodes")
    if len(adjacency) != num_nodes:
        raise ValueError("adjacency must have num_nodes rows")
    for row in adjacency:
        if len(row) != num_nodes:
            raise ValueError("adjacency must be a num_nodes x num_nodes matrix")
    for p in thresholds:
        if p < 0 or p > 100:
            raise ValueError("threshold percentages must be between 0 and 100")

    node_ids = [f"{label_prefix}{i}" if label_prefix else str(i) for i in range(num_nodes)]
    game = InfluenceGame(directed=directed)

    # Prepare symmetric weights if the game is undirected.
    symmetric_weights: List[List[float]] = [
        [0.0 for _ in range(num_nodes)] for _ in range(num_nodes)
    ]
    if directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency[i][j] < 0:
                    raise ValueError("adjacency entries must be nonnegative")
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                w = max(adjacency[i][j], adjacency[j][i])
                if w < 0:
                    raise ValueError("adjacency entries must be nonnegative")
                symmetric_weights[i][j] = w
                symmetric_weights[j][i] = w

    # Convert percentage thresholds to absolute values based on incoming weight.
    absolute_thresholds: List[float] = []
    for i in range(num_nodes):
        if directed:
            incoming_total = sum(adjacency[j][i] for j in range(num_nodes) if j != i)
        else:
            incoming_total = sum(symmetric_weights[j][i] for j in range(num_nodes) if j != i)

        percent = thresholds[i]
        if incoming_total <= 0:
            if percent <= 0:
                theta = 0.0
            else:
                # No incoming influence but a positive requirement: make it unreachable.
                theta = float("inf")
        else:
            theta = (percent / 100.0) * incoming_total
        absolute_thresholds.append(theta)

    for i, node in enumerate(node_ids):
        game.add_node(node, threshold=absolute_thresholds[i], label=node)

    if directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                weight = adjacency[i][j]
                if weight > 0:
                    game.add_edge(node_ids[i], node_ids[j], weight=weight)
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = symmetric_weights[i][j]
                if weight > 0:
                    game.add_edge(node_ids[i], node_ids[j], weight=weight)

    return game


def kuran_style_star_example() -> InfluenceGame:
    """
    Build a small star network inspired by Kuran style preference
    falsification examples.

    Design:
        - Central node 0 represents a regime or elite figure with a high
          threshold (hard to flip).
        - Peripheral nodes 1..k each have a moderate threshold.
        - Influence flows from neighbors to each node. This example is
          undirected so that both center and leaves influence each other.

    This is only a toy example but it is useful for qualitative plots
    in the report.

    Returns
    -------
    InfluenceGame
        Star shaped influence game.
    """
    game = InfluenceGame(directed=False)

    center = 0
    leaves = [1, 2, 3, 4]

    # Center has higher threshold
    game.add_node(center, threshold=2.5, label="Center")

    # Leaves have lower thresholds
    for leaf in leaves:
        game.add_node(leaf, threshold=1.0, label=f"L{leaf}")

    # Equal weights on all star edges
    for leaf in leaves:
        game.add_edge(center, leaf, weight=1.0)

    return game


if __name__ == "__main__":
    # Tiny smoke checks for the helpers.

    # Complete symmetric game
    complete_game = build_complete_symmetric_game(
        n=3,
        threshold=1.5,
        weight=1.0,
        directed=False,
        label_prefix="c",
    )
    profile = complete_game.empty_profile(active_value=1)
    print("Complete game profile:", profile_to_string(complete_game, profile))

    # Line graph game
    line_game = build_line_graph_game(
        n=4,
        default_threshold=1.0,
        weight=1.0,
        directed=True,
        label_prefix="p",
    )
    initial_profile = line_game.empty_profile(active_value=0)
    print("Line game initial profile:", profile_to_string(line_game, initial_profile))

    # Random game
    random_game = build_random_threshold_game(
        num_nodes=5,
        edge_prob=0.4,
        directed=True,
        seed=3210,
    )
    random_profile = random_game.empty_profile(active_value=0)
    print("Random game profile:", profile_to_string(random_game, random_profile))

    # Kuran style star
    star_game = kuran_style_star_example()
    star_profile = star_game.empty_profile(active_value=0)
    print("Kuran style star profile:", profile_to_string(star_game, star_profile))

    # Custom game from adjacency using percentage thresholds
    custom_thresholds = [50.0, 50.0, 50.0]
    custom_adj = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 0.0],
    ]
    custom_game = build_custom_game(
        num_nodes=3,
        thresholds=custom_thresholds,
        adjacency=custom_adj,
        directed=True,
        label_prefix="n",
    )
    print("Custom game nodes:", custom_game.nodes)
    print("Custom game edges:", [(u, v, custom_game.weight(u, v)) for u, v in custom_game.edges])
