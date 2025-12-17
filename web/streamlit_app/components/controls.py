from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Set

import pandas as pd
import streamlit as st

from src.influence_game import Action, InfluenceGame


def _sorted_nodes(nodes: Iterable[Any]) -> List[Any]:
    """Return nodes as a sorted list for stable widget order."""
    return sorted(nodes, key=lambda x: str(x))


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


def mode_selector(key: str = "mode_selector") -> str:
    """Toggle between preset scenarios and a custom network."""
    return st.radio(
        "Mode",
        options=["Preset scenario", "Custom network"],
        index=0,
        key=f"{key}_radio",
    )


def render_custom_network_controls(
    key_prefix: str = "custom_network",
    default_num_nodes: int = 5,
) -> CustomNetworkConfig:
    """
    Sidebar inputs for a custom undirected network with absolute thresholds.
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
        "Thresholds θ_i are constants in the same units as edge weights. "
        "Edge weights encode how much one neighbor matters."
    )

    thresholds_df = pd.DataFrame(
        {"node": node_labels, "threshold": [1.0] * num_nodes}
    )
    edited_thresholds = st.data_editor(
        thresholds_df,
        hide_index=True,
        column_config={
            "node": st.column_config.TextColumn("node", disabled=True),
            "threshold": st.column_config.NumberColumn(
                "threshold", min_value=0.0, max_value=50.0, step=0.1
            ),
        },
        key=f"{key_prefix}_thresholds",
    )
    sorted_thresholds = (
        edited_thresholds.copy()
        .assign(node_index=lambda df: df["node"].astype(int))
        .sort_values("node_index")
    )
    thresholds: List[float] = [float(val) for val in sorted_thresholds["threshold"].tolist()]

    adj_columns = node_labels
    base_adjacency = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes - 1):
        base_adjacency[i][i + 1] = 1.0
        base_adjacency[i + 1][i] = 1.0

    adj_data = {"node": node_labels}
    for j, col in enumerate(adj_columns):
        adj_data[col] = [base_adjacency[i][j] for i in range(num_nodes)]

    adjacency_df = pd.DataFrame(adj_data)
    edited_adj = st.data_editor(
        adjacency_df,
        hide_index=True,
        column_config={"node": st.column_config.TextColumn("node", disabled=True)},
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

    min_weight = min((val for row in adjacency_matrix for val in row), default=0.0)
    if min_weight < 0:
        st.warning("Negative weights are not supported; they were clamped to 0.")
        adjacency_matrix = [[max(0.0, val) for val in row] for row in adjacency_matrix]

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
    )
