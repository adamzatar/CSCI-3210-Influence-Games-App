from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List
import sys

# Make sure src is on the path before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import build_custom_game
from src.dynamics import CascadeResult, CascadeSimulator
from src.forcing import ForcingSetFinder
from src.influence_game import Action, InfluenceGame
from src.psne import PSNESolver
from web.streamlit_app.components.controls import (
    CustomNetworkConfig,
    example_selector,
    fixed_actions_from_forcing_set,
    forcing_set_selector,
    mode_selector,
    render_custom_network_controls,
)
from web.streamlit_app.components.plots import show_profile_plot


def _incoming_weight(game: InfluenceGame, node: Any) -> float:
    """Total incoming influence weight for a node."""
    if game.directed:
        neighbors = game.G.predecessors(node)
        return sum(game.weight(neighbor, node) for neighbor in neighbors)
    neighbors = game.G.neighbors(node)
    return sum(game.weight(neighbor, node) for neighbor in neighbors)


def _threshold_percentages(game: InfluenceGame, baseline: str = "incoming") -> Dict[Any, float | None]:
    """Convert absolute thresholds to percentages for display."""
    percents: Dict[Any, float | None] = {}
    denom_population = max(len(game.nodes) - 1, 1)
    for node in game.nodes:
        incoming = _incoming_weight(game, node)
        theta = game.threshold(node)
        if baseline == "population":
            if theta == float("inf"):
                percents[node] = None
            else:
                percents[node] = 100.0 * theta / denom_population if denom_population > 0 else None
        else:
            if incoming > 0 and theta != float("inf"):
                percents[node] = 100.0 * theta / incoming
            elif incoming <= 0 and theta == float("inf"):
                percents[node] = None
            elif incoming <= 0:
                percents[node] = 0.0
            else:
                percents[node] = None
    return percents


def _profile_breakdown(profile: Dict[Any, Action]) -> Dict[str, List[Any]]:
    """Split a profile into sorted active/inactive node lists."""
    active = sorted([n for n, a in profile.items() if a == 1], key=str)
    inactive = sorted([n for n, a in profile.items() if a == 0], key=str)
    return {"active": active, "inactive": inactive}


def _apply_threshold_nudge(
    game: InfluenceGame,
    nudge_percent: float,
    baseline: str = "incoming",
) -> None:
    """
    Shift thresholds by a percentage offset while keeping percentages meaningful.

    For each node with incoming influence, recompute theta using:
    new_percent = clamp(original_percent + nudge_percent, 0, 100)
    new_theta = (new_percent / 100) * incoming_weight

    Nodes with zero incoming weight keep their theta.
    """
    baseline_denominator = max(len(game.nodes) - 1, 1)
    if nudge_percent == 0.0:
        return

    for node in game.nodes:
        incoming = _incoming_weight(game, node)
        theta = game.threshold(node)
        if baseline == "population":
            percent = (theta / baseline_denominator) * 100.0 if theta != float("inf") else 100.0
            total = baseline_denominator
        else:
            if incoming <= 0:
                continue
            percent = (theta / incoming) * 100.0 if theta != float("inf") else 100.0
            total = incoming
        new_percent = max(0.0, min(100.0, percent + nudge_percent))
        new_theta = (new_percent / 100.0) * total
        game.set_threshold(node, new_theta)


def main() -> None:
    """Streamlit dashboard for the influence games project."""
    st.set_page_config(
        page_title="Influence games dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("CSCI 3210 Influence Games Dashboard")
    st.markdown(
        "Explore Kuran-style threshold games, PSNE, and most influential nodes. "
        "Custom networks use thresholds in percent of incoming influence."
    )

    selected_example = None

    # Sidebar controls
    with st.sidebar:
        st.header("Game configuration")

        mode = mode_selector()
        description_text = ""
        notes_text = ""
        threshold_baseline = "incoming"

        if mode == "Preset example":
            selected_example = example_selector()
            game = selected_example.game
            threshold_baseline = "incoming"

            st.subheader("Dynamics configuration")

            forcing_set = forcing_set_selector(
                game,
                default_forcing_set=selected_example.default_forcing_set,
                key="sidebar_forcing",
            )

            description_text = selected_example.description
            notes_text = selected_example.notes
        else:
            st.subheader("Custom network")
            custom_config: CustomNetworkConfig = render_custom_network_controls()
            game = build_custom_game(
                num_nodes=custom_config.num_nodes,
                thresholds=custom_config.thresholds,
                adjacency=custom_config.adjacency,
                directed=custom_config.directed,
                label_prefix="",
                threshold_baseline=custom_config.threshold_baseline,
            )
            threshold_baseline = custom_config.threshold_baseline

            # For custom networks, the "Forced activists" picker is the source of truth.
            forcing_set = set(custom_config.forcing_set)

            st.caption(
                "Thresholds are percentages of total incoming influence. "
                "Edge weights encode relative influence strength."
            )
            description_text = (
                f"Custom network with {custom_config.num_nodes} nodes and "
                f"{len(list(game.edges))} edges."
            )

        allow_cascade = st.checkbox(
            "Let influence spread via best responses",
            value=True,
            help="Uncheck to pin only the forced nodes at 1 without contagion.",
            key="allow_cascade",
        )

        nudge = st.slider(
            "Threshold nudge (all nodes)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Uniform percent offset to explore latent bandwagons. 0 = original thresholds.",
            key="threshold_nudge",
        )

        target_profile = game.empty_profile(active_value=1)
        initial_profile = game.empty_profile(active_value=0)
        if selected_example is not None:
            initial_profile = dict(game.normalize_profile(selected_example.default_initial_profile))
        for node_id in forcing_set:
            initial_profile[node_id] = 1
        _apply_threshold_nudge(game, nudge_percent=nudge, baseline=threshold_baseline)

    # Main content
    if selected_example is not None:
        st.subheader(f"Selected example: {selected_example.name}")
        st.markdown(description_text)
        st.caption(notes_text)
    elif description_text:
        st.caption(description_text)

    nodes_list = list(game.nodes)
    solver = PSNESolver(game)
    forcing_finder = ForcingSetFinder(game)
    simulator = CascadeSimulator(game)

    fixed_actions = fixed_actions_from_forcing_set(
        forcing_set=forcing_set,
        target_profile=target_profile,
    )

    if allow_cascade:
        cascade_result = simulator.run_until_fixpoint(
            initial_profile=initial_profile,
            fixed_actions=fixed_actions if fixed_actions else None,
            max_steps=25,
            detect_cycles=True,
        )
    else:
        cascade_result = CascadeResult(
            history=[initial_profile],
            converged=True,
            steps=0,
        )

    final_profile = cascade_result.final_profile
    breakdown = _profile_breakdown(final_profile)

    if fixed_actions:
        final_is_psne = solver.is_psne_with_fixed(final_profile, fixed_actions)
    else:
        final_is_psne = solver.is_psne(final_profile)

    st.subheader("Game summary")
    st.write(
        f"Nodes: {len(nodes_list)}, Edges: {len(list(game.edges))}, "
        f"Directed: {game.directed}"
    )

    percents = _threshold_percentages(game, baseline=threshold_baseline)
    summary_rows = []
    zero_incoming: List[Any] = []
    for node in nodes_list:
        incoming = _incoming_weight(game, node)
        theta = game.threshold(node)
        percent = percents.get(node)
        percent_display = f"{percent:.1f}%" if percent is not None else "N/A"
        summary_rows.append(
            {
                "node": node,
                "incoming_weight": incoming,
                "threshold_theta": theta,
                "threshold_percent": percent_display,
            }
        )
        if incoming == 0 and theta > 0:
            zero_incoming.append(node)
    st.dataframe(summary_rows, hide_index=True)
    st.caption(
        "Thresholds shown as θ are absolute totals. Default: θ_i = (percent_i / 100) × incoming weight. "
        "If you pick the population baseline, θ_i = (percent_i / 100) × (n-1). "
        "Example: 50% with 3 units of incoming weight gives θ = 1.5. "
        "If incoming weight is 0 and percent > 0, θ is infinite and the node cannot flip unless forced."
    )
    if zero_incoming:
        st.warning(
            f"Nodes with no incoming influence and positive thresholds cannot flip without forcing: {sorted(zero_incoming)}"
        )

    with st.expander("How this maps to the report", expanded=False):
        st.markdown(
            "- Model: linear-threshold best responses on a directed/undirected graph. Action 1 means joining.\n"
            "- Thresholds: set as percents in the sidebar; we store absolute θ. Edge weights are nonnegative influence strength.\n"
            "- PSNE: stable action profiles under the threshold rule. Forcing sets fix nodes so the all-ones profile becomes the only PSNE.\n"
            "- Cascades: optional synchronous updates to illustrate how a forcing set can propagate."
        )

    st.subheader("Outcome snapshot")
    st.markdown(
        "Forced nodes stay at 1. If cascades are enabled, everyone best-responds until stable. "
        "This is the resulting profile."
    )

    cols = st.columns(3)
    cols[0].metric("Active in outcome", len(breakdown["active"]))
    cols[1].metric("Inactive", len(breakdown["inactive"]))
    cols[2].metric("Outcome is PSNE", "Yes" if final_is_psne else "No")

    st.write(
        "Active nodes:",
        ", ".join(breakdown["active"]) if breakdown["active"] else "None",
    )
    st.write(
        "Inactive nodes:",
        ", ".join(breakdown["inactive"]) if breakdown["inactive"] else "None",
    )

    if not cascade_result.converged:
        st.warning(
            "Best-response updates did not reach a strict fixed point "
            f"(stopped after {cascade_result.steps} steps). "
            "The snapshot shows the latest profile."
        )

    st.subheader("Network snapshot")
    has_scipy = importlib.util.find_spec("scipy") is not None
    layout_options = {
        "Organic (spring)": "spring",
        "Circle": "circular",
        "Shell": "shell",
    }
    if has_scipy:
        layout_options["Even spacing (Kamada-Kawai)"] = "kamada_kawai"
    layout_choice = st.selectbox(
        "Choose a layout for the plot",
        options=list(layout_options.keys()),
        index=0,
    )
    layout_name = layout_options[layout_choice]

    if not has_scipy:
        st.caption("Install 'scipy' to enable the Kamada-Kawai layout option.")

    show_profile_plot(
        game,
        profile=final_profile,
        forcing_set=forcing_set,
        title="Final stable profile",
        layout=layout_name,
        percent_baseline=threshold_baseline,
    )
    st.caption(
        "Red = active, white = inactive, thick border = forced nodes. "
        "This is the stable outcome for your current forcing set and thresholds."
    )

    psne_tab, forcing_tab, dynamics_tab = st.tabs(
        ["PSNE (no forcing)", "Forcing sets (all-ones target)", "Dynamics (optional)"]
    )

    with psne_tab:
        st.caption("PSNE are stable profiles without fixing any nodes by hand.")
        if len(nodes_list) > 12:
            st.info("Graph too large to enumerate PSNE (n > 12).")
            psne_profiles: list[Dict[Any, Action]] = []
        else:
            psne_result = solver.enumerate_psne_bruteforce()
            psne_profiles = [game.normalize_profile(p) for p in psne_result.profiles]
            st.write(f"Found {len(psne_profiles)} PSNE (complete={psne_result.complete}).")
            if len(psne_profiles) > 1:
                active_counts = sorted(len([n for n, a in prof.items() if a == 1]) for prof in psne_profiles)
                st.warning(
                    f"Multiple PSNE detected. Lowest activation = {active_counts[0]}, "
                    f"highest = {active_counts[-1]}. Adjust thresholds to see if lower PSNE disappear."
                )
            if psne_profiles:
                psne_rows = []
                for idx, prof in enumerate(psne_profiles, start=1):
                    active_nodes = sorted([n for n, a in prof.items() if a == 1], key=str)
                    psne_rows.append(
                        {
                            "PSNE #": idx,
                            "Active count": len(active_nodes),
                            "Active nodes": ", ".join(active_nodes) if active_nodes else "None",
                        }
                    )
                st.dataframe(psne_rows, hide_index=True, use_container_width=True)
                with st.expander("Show PSNE as raw profiles", expanded=False):
                    for idx, prof in enumerate(psne_profiles, start=1):
                        st.code(f"PSNE {idx}: {prof}")
            else:
                st.write("No PSNE found for this configuration.")

    with forcing_tab:
        st.caption("Smallest sets to fix at 1 so all-ones is the only PSNE (our 'most influential' idea).")
        forcing_result = forcing_finder.minimal_forcing_sets(
            target_profile=target_profile,
            max_size=len(nodes_list),
        )
        if forcing_result.size is None:
            st.info("No forcing set found within the search limits.")
        else:
            st.write(f"Minimal forcing size: {forcing_result.size}")
            if forcing_result.forcing_sets:
                for forcing in forcing_result.forcing_sets:
                    st.code(f"Forcing set: {sorted(forcing)}")
            else:
                st.write("No forcing sets returned.")
        with st.expander("Irfan's indicative nodes", expanded=False):
            st.info(
                "Irfan’s definition is the smallest set of nodes whose actions uniquely identify the target PSNE. "
                "Not implemented here; we focus on forcing sets to guarantee all-ones."
            )

    with dynamics_tab:
        st.caption(
            "Synchronous best responses starting from the initial profile with forced nodes pinned. "
            "Stops at a fixed point or a detected cycle."
        )
        st.write(f"Converged: {cascade_result.converged}")
        st.write(f"Steps: {cascade_result.steps}")
        st.write("Final profile:", final_profile)
        if not cascade_result.converged:
            st.warning(
                "Best-response updates did not reach a strict fixed point "
                f"(stopped after {cascade_result.steps} steps). "
                "The snapshot shows the latest profile."
            )


if __name__ == "__main__":
    main()
