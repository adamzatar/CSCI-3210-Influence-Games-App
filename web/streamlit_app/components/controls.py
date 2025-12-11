from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

import pandas as pd
import streamlit as st

from src.influence_game import Action, InfluenceGame
from web.streamlit_app.state.examples import ExampleDefinition, get_all_examples


def _sorted_nodes(nodes: Iterable[Any]) -> List[Any]:
    """Return nodes as a sorted list for stable widget order."""
    return sorted(nodes, key=lambda x: str(x))


def example_selector(
    key: str = "example_selector",
    default_key: str = "kuran_star",
) -> ExampleDefinition:
    """Let the user choose one of the canonical examples."""
    examples = get_all_examples()
    default_index = 0
    for idx, ex in enumerate(examples):
        if ex.key == default_key:
            default_index = idx
            break

    selected = st.selectbox(
        "Example game",
        options=examples,
        index=default_index,
        format_func=lambda ex: ex.name,
        key=f"{key}_selectbox",
    )

    st.markdown(selected.description)
    with st.expander("How this connects to the project", expanded=False):
        st.caption(selected.notes)

    st.write(
        f"Nodes: {len(list(selected.game.nodes))}, "
        f"Edges: {len(list(selected.game.edges))}"
    )

    return selected


def build_profile_from_active_nodes(
    game: InfluenceGame,
    active_nodes: Iterable[Any],
) -> Dict[Any, Action]:
    """Build a profile where exactly the chosen nodes are active."""
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
    """Pick which nodes belong to the forcing set."""
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
    Choose the initial profile: default, all inactive, or custom active nodes.
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
    """Build fixed_actions from a forcing set and a target profile."""
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
    """Toggle between custom networks and preset examples."""
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
    Sidebar inputs for a custom network with percentage thresholds.
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
                "threshold", min_value=0.0, max_value=100.0, step=0.1
            ),
        },
        key=f"{key_prefix}_thresholds",
    )
    sorted_thresholds = (
        edited_thresholds.copy()
        .assign(node_index=lambda df: df["node"].astype(int))
        .sort_values("node_index")
    )
    raw_thresholds: List[float] = [
        float(val) for val in sorted_thresholds["threshold"].tolist()
    ]
    thresholds: List[float] = []
    clamped = False
    for val in raw_thresholds:
        if val < 0.0:
            thresholds.append(0.0)
            clamped = True
        elif val > 100.0:
            thresholds.append(100.0)
            clamped = True
        else:
            thresholds.append(val)
    if clamped:
        st.caption("Thresholds were clamped to the valid range [0, 100].")

    adj_columns = node_labels
    adj_data = {"node": node_labels}
    for col in adj_columns:
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
    for _, row in sorted_adj.iterrows():
        values = []
        for col in adj_columns:
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                val = 0.0
            values.append(val)
        adjacency_matrix.append(values)

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
