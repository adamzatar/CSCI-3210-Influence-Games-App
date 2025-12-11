from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set

from src import build_custom_game
from src.influence_game import InfluenceGame


@dataclass
class ExampleDefinition:
    key: str
    name: str
    description: str
    game: InfluenceGame
    default_forcing_set: Set[str]
    default_initial_profile: Dict[str, int]
    notes: str


def _kuran_star() -> ExampleDefinition:
    nodes = ["0", "1", "2", "3", "4"]
    thresholds = [75.0] + [25.0] * 4
    n = len(nodes)
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
    for leaf_idx in range(1, n):
        adjacency[0][leaf_idx] = 1.0
        adjacency[leaf_idx][0] = 1.0

    game = build_custom_game(
        num_nodes=n,
        thresholds=thresholds,
        adjacency=adjacency,
        directed=False,
        label_prefix="",
    )

    default_forcing_set: Set[str] = {"1", "2"}
    default_initial_profile = game.empty_profile(active_value=0)
    for node in default_forcing_set:
        default_initial_profile[node] = 1

    description = (
        "Star with resistant center (75%) and easy-to-tip leaves (25%). "
        "Captures a regime center surrounded by citizens with lower thresholds."
    )
    notes = (
        "Connects to Kuran’s preference falsification story: a few dissenting leaves "
        "can trigger wider cascades. The center resists until enough peripheral "
        "support forms."
    )

    return ExampleDefinition(
        key="kuran_star",
        name="Kuran star",
        description=description,
        game=game,
        default_forcing_set=default_forcing_set,
        default_initial_profile=default_initial_profile,
        notes=notes,
    )


def _mutual_pair() -> ExampleDefinition:
    game = build_custom_game(
        num_nodes=2,
        thresholds=[50.0, 50.0],
        adjacency=[
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        directed=False,
        label_prefix="",
    )

    default_forcing_set: Set[str] = {"0"}
    default_initial_profile = game.empty_profile(active_value=0)
    default_initial_profile["0"] = 1

    description = "Two people with mutual influence and 50% thresholds; PSNE are all 0 and all 1."
    notes = (
        "Clean PSNE/forcing example: for all-ones, minimal forcing size is 1 and either singleton works. "
        "Shows how the Most Influential Nodes definition works."
    )

    return ExampleDefinition(
        key="mutual_pair",
        name="Mutual pair",
        description=description,
        game=game,
        default_forcing_set=default_forcing_set,
        default_initial_profile=default_initial_profile,
        notes=notes,
    )


def _triangle() -> ExampleDefinition:
    game = build_custom_game(
        num_nodes=3,
        thresholds=[50.0, 50.0, 50.0],
        adjacency=[
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        directed=False,
        label_prefix="",
    )

    default_forcing_set: Set[str] = {"0"}
    default_initial_profile = game.empty_profile(active_value=0)
    default_initial_profile["0"] = 1

    description = "Undirected triangle, weight 1 edges, thresholds 50% (θ = 1 with two neighbors)."
    notes = (
        "Canonical Most Influential Nodes example: any single zealot forces all ones under the PSNE-based "
        "forcing definition, showing why we search subsets by size."
    )

    return ExampleDefinition(
        key="triangle",
        name="Symmetric triangle",
        description=description,
        game=game,
        default_forcing_set=default_forcing_set,
        default_initial_profile=default_initial_profile,
        notes=notes,
    )


def _two_communities_zealot() -> ExampleDefinition:
    nodes = ["Z", "A1", "A2", "A3", "B1", "B2", "B3"]
    idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]

    def connect(u: str, v: str) -> None:
        i, j = idx[u], idx[v]
        adjacency[i][j] = 1.0
        adjacency[j][i] = 1.0

    for node in nodes:
        if node != "Z":
            connect("Z", node)

    for u, v in combinations(["A1", "A2", "A3"], 2):
        connect(u, v)

    connect("B1", "B2")
    connect("B2", "B3")

    thresholds = [
        0.0,  # Z
        40.0,
        40.0,
        40.0,  # A nodes
        70.0,
        70.0,
        70.0,  # B nodes
    ]

    game = build_custom_game(
        num_nodes=n,
        thresholds=thresholds,
        adjacency=adjacency,
        directed=False,
        label_prefix="",
    )

    default_forcing_set: Set[str] = {"Z"}
    default_initial_profile = game.empty_profile(active_value=0)
    default_initial_profile["Z"] = 1

    description = "Zealot bridge between two communities (A dense, B sparse) with mixed thresholds."
    notes = (
        "Shows how a high-degree zealot (Z) may not uniquely force all ones. Forcing sets often need "
        "Z plus a well-placed B node (e.g., B2) to cover both communities, contrasting naive degree intuition."
    )

    return ExampleDefinition(
        key="two_communities_zealot",
        name="Two communities with zealot bridge",
        description=description,
        game=game,
        default_forcing_set=default_forcing_set,
        default_initial_profile=default_initial_profile,
        notes=notes,
    )


def get_all_examples() -> List[ExampleDefinition]:
    """Return canonical examples in a stable order."""
    return [
        _kuran_star(),
        _mutual_pair(),
        _triangle(),
        _two_communities_zealot(),
    ]


def get_example_by_key(key: str) -> ExampleDefinition:
    """Look up an example by key."""
    for ex in get_all_examples():
        if ex.key == key:
            return ex
    raise KeyError(f"Unknown example key: {key}")
