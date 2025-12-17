from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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
    Turn a profile into a compact string like 'A:1 B:0'.

    Sorts nodes unless sort_nodes=False. Good for debug prints.
    """
    normalized = game.normalize_profile(profile)
    nodes = list(game.nodes)
    if sort_nodes:
        nodes.sort()

    parts: List[str] = []
    for node in nodes:
        parts.append(f"{node}:{normalized[node]}")
    return " ".join(parts)


def build_complete_symmetric_game(
    n: int,
    threshold: float,
    weight: float = 1.0,
    directed: bool = False,
    label_prefix: str = "v",
) -> InfluenceGame:
    """Complete graph with the same threshold and weight everywhere."""
    if n <= 0:
        raise ValueError("Number of nodes must be positive")
    if threshold < 0:
        raise ValueError("Threshold must be nonnegative")
    if weight < 0:
        raise ValueError("Weight must be nonnegative")

    game = InfluenceGame(directed=directed)

    for i in range(n):
        game.add_node(i, threshold=threshold, label=f"{label_prefix}{i}")

    if directed:
        for i in range(n):
            for j in range(n):
                if i != j:
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
    """Path graph v0 -> v1 -> ... with simple thresholds."""
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
        game.add_node(i, threshold=theta, label=f"{label_prefix}{i}")

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
    """Random Erdos–Renyi style influence game."""
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
    Build a game from absolute thresholds and an adjacency matrix.

    Thresholds are constants in the same units as edge weights. We do not
    interpret them as percentages anywhere in the codebase.
    Edge weights come directly from the adjacency matrix (diagonal ignored).
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
    for theta in thresholds:
        if theta < 0:
            raise ValueError("thresholds must be nonnegative")

    node_ids = [f"{label_prefix}{i}" if label_prefix else str(i) for i in range(num_nodes)]
    game = InfluenceGame(directed=directed)

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
                weight = max(adjacency[i][j], adjacency[j][i])
                if weight < 0:
                    raise ValueError("adjacency entries must be nonnegative")
                symmetric_weights[i][j] = weight
                symmetric_weights[j][i] = weight

    for i, node in enumerate(node_ids):
        game.add_node(node, threshold=thresholds[i], label=node)

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
    """Toy star network used in report plots."""
    game = InfluenceGame(directed=False)

    center = 0
    leaves = [1, 2, 3, 4]

    game.add_node(center, threshold=2.5, label="Center")
    for leaf in leaves:
        game.add_node(leaf, threshold=1.0, label=f"L{leaf}")
        game.add_edge(center, leaf, weight=1.0)

    return game


if __name__ == "__main__":
    # Tiny smoke checks for the helpers.

    complete_game = build_complete_symmetric_game(
        n=3,
        threshold=1.5,
        weight=1.0,
        directed=False,
        label_prefix="c",
    )
    profile = complete_game.empty_profile(active_value=1)
    print("Complete game profile:", profile_to_string(complete_game, profile))

    line_game = build_line_graph_game(
        n=4,
        default_threshold=1.0,
        weight=1.0,
        directed=True,
        label_prefix="p",
    )
    initial_profile = line_game.empty_profile(active_value=0)
    print("Line game initial profile:", profile_to_string(line_game, initial_profile))

    random_game = build_random_threshold_game(
        num_nodes=5,
        edge_prob=0.4,
        directed=True,
        seed=3210,
    )
    random_profile = random_game.empty_profile(active_value=0)
    print("Random game profile:", profile_to_string(random_game, random_profile))

    star_game = kuran_style_star_example()
    star_profile = star_game.empty_profile(active_value=0)
    print("Kuran style star profile:", profile_to_string(star_game, star_profile))

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
