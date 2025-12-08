# web/streamlit_app/components/controls.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Set

import pandas as pd

import streamlit as st

from src.influence_game import Action, InfluenceGame
from web.streamlit_app.state.examples import (
    ExampleInstance,
    ExampleSpec,
    build_example_instance,
    list_example_specs,
)


def _sorted_nodes(nodes: Iterable[Any]) -> List[Any]:
    """
    Return nodes as a sorted list, using string representation as a tie breaker.

    This keeps node order stable in Streamlit widgets.
    """
    return sorted(nodes, key=lambda x: str(x))


def example_selector(
    key: str = "example_selector",
    default_key: str = "mutual_pair",
    allowed_keys: List[str] | None = None,
) -> Tuple[ExampleSpec, ExampleInstance]:
    """
    Let the user choose which example game to work with.

    Parameters
    ----------
    key:
        Streamlit widget key prefix.
    default_key:
        Example key that should be selected by default if it exists.

    Returns
    -------
    ExampleSpec, ExampleInstance
        The chosen example specification and a fresh instance.
    """
    if allowed_keys is None:
        allowed_keys = ["mutual_pair", "triangle"]

    specs: List[ExampleSpec] = list_example_specs(keys=allowed_keys)
    if not specs:
        st.error("No example specifications are registered.")
        st.stop()

    default_index = 0
    for idx, spec in enumerate(specs):
        if spec.key == default_key:
            default_index = idx
            break

    selected_spec = st.selectbox(
        "Example game",
        options=specs,
        index=default_index,
        format_func=lambda s: s.name,
        key=f"{key}_selectbox",
    )

    if selected_spec.description:
        st.caption(selected_spec.description)

    instance = build_example_instance(selected_spec.key)

    st.write(
        f"Nodes: {len(list(instance.game.nodes))}, "
        f"Edges: {len(list(instance.game.edges))}"
    )

    return selected_spec, instance


def build_profile_from_active_nodes(
    game: InfluenceGame,
    active_nodes: Iterable[Any],
) -> Dict[Any, Action]:
    """
    Build a profile where exactly the chosen nodes are active.

    Parameters
    ----------
    game:
        InfluenceGame whose nodes define the profile domain.
    active_nodes:
        Iterable of nodes that should have action 1.

    Returns
    -------
    dict
        Mapping from node to action in {0, 1}.
    """
    profile: Dict[Any, Action] = game.empty_profile(active_value=0)
    for node in active_nodes:
        if node in profile:
            profile[node] = 1
    return profile


def forcing_set_selector(
    game: InfluenceGame,
    default_forcing_set: Set[Any] | None = None,
    key: str = "forcing_set_selector",
) -> Set[Any]:
    """
    Let the user choose which nodes belong to the forcing set.

    Parameters
    ----------
    game:
        InfluenceGame that provides the node set.
    default_forcing_set:
        Default forcing set suggestion, for example a zealot node.
    key:
        Streamlit widget key prefix.

    Returns
    -------
    set
        Selected forcing set as a set of node identifiers.
    """
    if default_forcing_set is None:
        default_forcing_set = set()

    node_list = _sorted_nodes(game.nodes)
    default_list = [n for n in node_list if n in default_forcing_set]

    selected_nodes = st.multiselect(
        "Forcing set (nodes fixed exogenously)",
        options=node_list,
        default=default_list,
        key=f"{key}_multiselect",
    )

    return set(selected_nodes)


def initial_profile_selector(
    game: InfluenceGame,
    default_initial_profile: Mapping[Any, Action],
    key: str = "initial_profile_selector",
) -> Dict[Any, Action]:
    """
    Let the user choose the initial profile for dynamics.

    The UI offers three modes:

      1. Use the example's default initial profile.
      2. All inactive.
      3. Custom, chosen by selecting which nodes start active.

    Parameters
    ----------
    game:
        InfluenceGame that provides the node set.
    default_initial_profile:
        Example specific default initial profile suggested by the model.
    key:
        Streamlit widget key prefix.

    Returns
    -------
    dict
        Mapping from node to action in {0, 1} used as initial state.
    """
    mode = st.radio(
        "Initial profile",
        options=[
            "Use example default",
            "All inactive",
            "Custom active nodes",
        ],
        key=f"{key}_mode",
    )

    if mode == "Use example default":
        # Normalize in case the example did something partial
        return dict(game.normalize_profile(default_initial_profile))

    if mode == "All inactive":
        return game.empty_profile(active_value=0)

    node_list = _sorted_nodes(game.nodes)
    default_active = [
        node
        for node in node_list
        if default_initial_profile.get(node, 0) == 1
    ]

    selected_active = st.multiselect(
        "Nodes initially active",
        options=node_list,
        default=default_active,
        key=f"{key}_active_multiselect",
    )

    return build_profile_from_active_nodes(game, selected_active)


def fixed_actions_from_forcing_set(
    forcing_set: Set[Any],
    target_profile: Mapping[Any, Action],
) -> Dict[Any, Action]:
    """
    Build a fixed_actions mapping from a forcing set and a target profile.

    Parameters
    ----------
    forcing_set:
        Set of nodes that will be fixed by external pressure.
    target_profile:
        Target profile that defines what action each forcing node
        should be fixed to.

    Returns
    -------
    dict
        Mapping from node to action for use as fixed_actions in the
        CascadeSimulator or PSNE solver.
    """
    fixed: Dict[Any, Action] = {}
    for node in forcing_set:
        if node in target_profile:
            fixed[node] = target_profile[node]
    return fixed


@dataclass
class CustomNetworkConfig:
    num_nodes: int
    thresholds: List[float]
    adjacency: List[List[float]]
    forcing_set: Set[str]
    directed: bool


def mode_selector(key: str = "mode_selector") -> str:
    """
    Choose between building a custom network or using a preset example.
    """
    return st.radio(
        "Mode",
        options=["Custom network", "Preset example"],
        index=0,
        key=f"{key}_radio",
    )


def render_custom_network_controls(
    key_prefix: str = "custom_network",
    default_num_nodes: int = 5,
) -> CustomNetworkConfig:
    """
    Render sidebar inputs for a custom influence network using percentage thresholds.
    """
    num_nodes = st.slider(
        "Number of nodes",
        min_value=2,
        max_value=10,
        value=default_num_nodes,
        key=f"{key_prefix}_num_nodes",
    )

    node_labels = [str(i) for i in range(num_nodes)]

    st.caption(
        "Thresholds are percentages of total incoming influence needed "
        "to switch to 1. Edge weights are relative influence strengths."
    )

    # Thresholds editor
    thresholds_df = pd.DataFrame(
        {"node": node_labels, "threshold": [50.0] * num_nodes}
    )
    edited_thresholds = st.data_editor(
        thresholds_df,
        hide_index=True,
        column_config={
            "node": st.column_config.TextColumn(
                "node", disabled=True
            ),
            "threshold": st.column_config.NumberColumn(
                "threshold", min_value=0.0, step=0.1
            ),
        },
        key=f"{key_prefix}_thresholds",
    )
    sorted_thresholds = (
        edited_thresholds.copy()
        .assign(node_index=lambda df: df["node"].astype(int))
        .sort_values("node_index")
    )
    thresholds: List[float] = [
        float(val) for val in sorted_thresholds["threshold"].tolist()
    ]

    # Adjacency editor
    adj_columns = node_labels
    adj_data = {"node": node_labels}
    for col in adj_columns:
        # Default to a simple line: i -> i+1
        col_values = [0.0] * num_nodes
        for i in range(num_nodes - 1):
            if int(col) == i + 1:
                col_values[i] = 1.0
        adj_data[col] = col_values

    adjacency_df = pd.DataFrame(adj_data)
    edited_adj = st.data_editor(
        adjacency_df,
        hide_index=True,
        column_config={
            "node": st.column_config.TextColumn("node", disabled=True)
        },
        key=f"{key_prefix}_adjacency",
    )
    sorted_adj = (
        edited_adj.copy()
        .assign(node_index=lambda df: df["node"].astype(int))
        .sort_values("node_index")
    )
    adjacency_matrix: List[List[float]] = []
    for idx, row in sorted_adj.iterrows():
        values = []
        for col in adj_columns:
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                val = 0.0
            values.append(val)
        adjacency_matrix.append(values)

    # Enforce zero diagonal
    for i in range(min(len(adjacency_matrix), num_nodes)):
        adjacency_matrix[i][i] = 0.0

    directed = st.checkbox(
        "Treat network as directed",
        value=True,
        key=f"{key_prefix}_directed",
    )

    forcing_set = st.multiselect(
        "Forced activists (always active)",
        options=node_labels,
        default=[],
        key=f"{key_prefix}_forcing",
    )

    return CustomNetworkConfig(
        num_nodes=num_nodes,
        thresholds=thresholds,
        adjacency=adjacency_matrix,
        forcing_set=set(forcing_set),
        directed=directed,
    )
