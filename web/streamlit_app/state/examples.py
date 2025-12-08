# web/streamlit_app/state/examples.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Set, Tuple

from src.influence_game import Action, InfluenceGame
from src.utils import (
    build_complete_symmetric_game,
    build_line_graph_game,
    build_random_threshold_game,
    build_custom_game,
)


@dataclass
class ExampleInstance:
    """
    Concrete instance of an example game for the frontend.

    Attributes
    ----------
    game:
        InfluenceGame object describing the graph, thresholds, and weights.
    default_initial_profile:
        Mapping from node to action in {0, 1} that the UI should use as
        the starting state for cascades. This is already normalized to
        contain all nodes.
    default_forcing_set:
        Set of nodes that form a natural default forcing set for this
        example, for example a zealot node or an initial activist set.
    description:
        Short human readable description of what this instance is meant
        to illustrate.
    """

    game: InfluenceGame
    default_initial_profile: Mapping[Any, Action]
    default_forcing_set: Set[Any]
    description: str


@dataclass
class ExampleSpec:
    """
    Specification of a named example for the frontend.

    An ExampleSpec knows how to build a fresh ExampleInstance when the
    user selects it from the UI.

    Attributes
    ----------
    key:
        Internal identifier, used in select boxes and routing.
    name:
        Short display name for the example.
    description:
        Human readable description for tooltips or side text.
    builder:
        Zero argument function that constructs a new ExampleInstance.
        The builder should not capture mutable global state so that
        each call starts from a clean game.
    """

    key: str
    name: str
    description: str
    builder: Callable[[], ExampleInstance]


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------


def _build_triangle_example() -> ExampleInstance:
    """
    Classic 3 node triangle where all neighbors influence each other.

    Structure
    ---------
    - Undirected complete graph on 3 nodes.
    - Edge weights all 1.0.
    - Thresholds 50 percent of incoming weight (θ = 1.0).

    Defaults
    --------
    - Initial profile: all zeros (nobody active).
    - Default forcing set: {first node} to make it easy to seed a
      cascade analysis in the UI.
    """
    game = build_complete_symmetric_game(
        n=3,
        threshold=1.0,
        weight=1.0,
        directed=False,
        label_prefix="v",
    )

    nodes: List[Any] = list(game.nodes)
    initial_profile: MutableMapping[Any, Action] = game.empty_profile(active_value=0)
    default_forcing_set: Set[Any] = {nodes[0]} if nodes else set()

    description = (
        "3 node triangle with symmetric thresholds. "
        "Thresholds set to 50 percent of total incoming weight. "
        "Illustrates multiple PSNE: all inactive and all active."
    )

    return ExampleInstance(
        game=game,
        default_initial_profile=initial_profile,
        default_forcing_set=default_forcing_set,
        description=description,
    )


def _build_mutual_pair_example() -> ExampleInstance:
    """
    Two nodes influencing each other with 50 percent thresholds.

    Structure
    ---------
    - Undirected edge between nodes 0 and 1 with weight 1.
    - Thresholds: 50 percent of incoming weight (θ = 0.5).

    Defaults
    --------
    - Initial profile: all zeros.
    - Default forcing set: empty, the user can pick either node as a zealot.
    """
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

    initial_profile = game.empty_profile(active_value=0)
    default_forcing_set: Set[Any] = set()

    description = (
        "Two nodes with mutual influence and 50 percent thresholds. "
        "PSNE are (0,0) and (1,1)."
    )

    return ExampleInstance(
        game=game,
        default_initial_profile=initial_profile,
        default_forcing_set=default_forcing_set,
        description=description,
    )


def _build_dense_zealot_example() -> ExampleInstance:
    """
    Dense 9 node graph with one zealot, matching the 80 percent board sketch.

    Structure
    ---------
    - Undirected complete graph on 9 nodes.
    - Eight regular nodes v0, ..., v7 with threshold 80.0.
    - One zealot node 'Z' with threshold 0.0.
    - All edges have weight 10.0.

    Defaults
    --------
    - Initial profile: all zeros (everyone initially inactive).
    - Default forcing set: {'Z'} to highlight the zealot as the
      influential player whose commitment may select the all-active
      equilibrium.
    """
    game = InfluenceGame(directed=False)

    regular_nodes = [f"v{i}" for i in range(8)]
    zealot_node = "Z"

    for node in regular_nodes:
        game.add_node(node, threshold=80.0, label=node)
    game.add_node(zealot_node, threshold=0.0, label=zealot_node)

    for u, v in combinations(regular_nodes + [zealot_node], 2):
        game.add_edge(u, v, weight=10.0)

    initial_profile = game.empty_profile(active_value=0)
    default_forcing_set: Set[Any] = {zealot_node}

    description = (
        "Dense 9 node graph with one zealot Z (threshold 0) and eight "
        "high threshold nodes (80). Useful for discussing most "
        "influential players and equilibrium selection."
    )

    return ExampleInstance(
        game=game,
        default_initial_profile=initial_profile,
        default_forcing_set=default_forcing_set,
        description=description,
    )


def _build_line_example() -> ExampleInstance:
    """
    Simple directed line graph to illustrate sequential cascades.

    Structure
    ---------
    - Directed path v0 -> v1 -> ... -> v4.
    - Thresholds default to 1.0.
    - Edge weights 1.0.

    Defaults
    --------
    - Initial profile: all zeros.
    - Default forcing set: {v0} so that the cascade can propagate from
      left to right when v0 is fixed to 1.

    This example is helpful for connecting Kuran style threshold
    cascades to the influence game dynamics.
    """
    n = 5
    game = build_line_graph_game(
        n=n,
        thresholds=None,
        default_threshold=1.0,
        weight=1.0,
        directed=True,
        label_prefix="v",
    )

    initial_profile = game.empty_profile(active_value=0)
    nodes: List[Any] = list(game.nodes)
    default_forcing_set: Set[Any] = {nodes[0]} if nodes else set()

    description = (
        "Directed line v0 -> ... -> v4 with threshold 1. "
        "Illustrates stepwise cascades from a single activist."
    )

    return ExampleInstance(
        game=game,
        default_initial_profile=initial_profile,
        default_forcing_set=default_forcing_set,
        description=description,
    )


def _build_random_network_example() -> ExampleInstance:
    """
    Random Erdos–Renyi style graph with random thresholds and weights.

    Structure
    ---------
    - Number of nodes: 10.
    - Edge probability: 0.25.
    - Directed edges.
    - Thresholds drawn uniformly from [0.5, 2.5].
    - Weights drawn uniformly from [0.5, 2.0].

    Defaults
    --------
    - Initial profile: all zeros.
    - Default forcing set: empty. The user can choose forcing sets in
      the UI.

    This example is mainly for exploratory experiments and stress
    testing the implementation.
    """
    game = build_random_threshold_game(
        num_nodes=10,
        edge_prob=0.25,
        directed=True,
        threshold_range=(0.5, 2.5),
        weight_range=(0.5, 2.0),
        seed=None,
        label_prefix="v",
    )

    initial_profile = game.empty_profile(active_value=0)
    default_forcing_set: Set[Any] = set()

    description = (
        "Random directed network with random thresholds and weights. "
        "Useful for exploratory cascades and robustness experiments."
    )

    return ExampleInstance(
        game=game,
        default_initial_profile=initial_profile,
        default_forcing_set=default_forcing_set,
        description=description,
    )


# ---------------------------------------------------------------------------
# Registry and helper functions
# ---------------------------------------------------------------------------

_EXAMPLE_SPECS: Dict[str, ExampleSpec] = {
    "mutual_pair": ExampleSpec(
        key="mutual_pair",
        name="Mutual pair (50%)",
        description="Two nodes with mutual influence and 50 percent thresholds.",
        builder=_build_mutual_pair_example,
    ),
    "triangle": ExampleSpec(
        key="triangle",
        name="Triangle (3 nodes)",
        description="3 node symmetric triangle with two PSNE: all 0 and all 1.",
        builder=_build_triangle_example,
    ),
    "dense_zealot": ExampleSpec(
        key="dense_zealot",
        name="Dense 80 percent + zealot",
        description=(
            "9 node dense graph with one zealot and eight high-threshold nodes. "
            "Matches the 80 percent board example."
        ),
        builder=_build_dense_zealot_example,
    ),
    "line": ExampleSpec(
        key="line",
        name="Directed line (5 nodes)",
        description="Directed path v0 -> ... -> v4 to show sequential cascades.",
        builder=_build_line_example,
    ),
    "random": ExampleSpec(
        key="random",
        name="Random network",
        description="Random directed graph with random thresholds and weights.",
        builder=_build_random_network_example,
    ),
}


def list_example_keys(keys: List[str] | None = None) -> List[str]:
    """
    Return the list of available example keys.

    If keys is provided, only return those keys that exist.
    """
    if keys is None:
        return list(_EXAMPLE_SPECS.keys())
    return [k for k in keys if k in _EXAMPLE_SPECS]


def list_example_specs(keys: List[str] | None = None) -> List[ExampleSpec]:
    """
    Return registered example specifications.

    If keys is provided, only specs for those keys are returned. This
    lets the UI present a smaller theory-focused menu while keeping the
    other examples available for later use.
    """
    if keys is None:
        return list(_EXAMPLE_SPECS.values())
    return [_EXAMPLE_SPECS[k] for k in keys if k in _EXAMPLE_SPECS]


def get_example_spec(key: str) -> ExampleSpec:
    """
    Retrieve an ExampleSpec by key.

    Parameters
    ----------
    key:
        Example identifier such as "triangle" or "dense_zealot".

    Returns
    -------
    ExampleSpec

    Raises
    ------
    KeyError
        If the key is not a known example.
    """
    return _EXAMPLE_SPECS[key]


def build_example_instance(key: str) -> ExampleInstance:
    """
    Build a fresh ExampleInstance for the given example key.

    This calls the corresponding builder function and returns a new
    game instance together with default initial profile and default
    forcing set.

    Parameters
    ----------
    key:
        Example identifier.

    Returns
    -------
    ExampleInstance
    """
    spec = get_example_spec(key)
    return spec.builder()
