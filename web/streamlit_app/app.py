# web/streamlit_app/app.py
from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import build_custom_game
from src.dynamics import CascadeSimulator
from src.influence_game import Action, InfluenceGame
from src.psne import PSNESolver
from src.forcing import ForcingSetFinder

from web.streamlit_app.components.controls import (
    mode_selector,
    example_selector,
    forcing_set_selector,
    fixed_actions_from_forcing_set,
    render_custom_network_controls,
    CustomNetworkConfig,
)
from web.streamlit_app.components.plots import show_profile_plot


def _incoming_weight(game: InfluenceGame, node: Any) -> float:
    """Total incoming influence weight for a node."""
    if game.directed:
        neighbors = game.G.predecessors(node)
        return sum(game.weight(neighbor, node) for neighbor in neighbors)
    neighbors = game.G.neighbors(node)
    return sum(game.weight(neighbor, node) for neighbor in neighbors)


def _threshold_percentages(game: InfluenceGame) -> Dict[Any, float | None]:
    """
    Convert absolute thresholds back to percentages of incoming weight for display.
    """
    percents: Dict[Any, float | None] = {}
    for node in game.nodes:
        incoming = _incoming_weight(game, node)
        theta = game.threshold(node)
        if incoming > 0 and theta != float("inf"):
            percents[node] = 100.0 * theta / incoming
        elif incoming <= 0 and theta == float("inf"):
            percents[node] = None
        elif incoming <= 0:
            percents[node] = 0.0
        else:
            percents[node] = None
    return percents


def main() -> None:
    """
    Streamlit dashboard for CSCI 3210 influence games project.

    Features in this first version:
      - Choose a preset influence game (triangle, dense 80 percent with
        zealot, line, random).
      - Choose forcing set and initial profile.
      - Run best response dynamics as a cascade with fixed players.
      - Visualize initial and final profiles and the cascade history.
      - Optionally enumerate PSNE for small games.
    """
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

    # Layout: sidebar for controls, main area for theory outputs.
    with st.sidebar:
        st.header("Game configuration")

        mode = mode_selector()
        description_text = ""

        if mode == "Preset example":
            spec, instance = example_selector()
            game = instance.game

            st.subheader("Dynamics configuration")

            forcing_set = forcing_set_selector(
                game,
                default_forcing_set=instance.default_forcing_set,
                key="sidebar_forcing",
            )

            description_text = (
                f"{spec.name}: {instance.description}"
                if instance.description
                else spec.name
            )
        else:
            st.subheader("Custom network")
            custom_config: CustomNetworkConfig = render_custom_network_controls()
            game = build_custom_game(
                num_nodes=custom_config.num_nodes,
                thresholds=custom_config.thresholds,
                adjacency=custom_config.adjacency,
                directed=custom_config.directed,
                label_prefix="",
            )

            forcing_set = forcing_set_selector(
                game,
                default_forcing_set=custom_config.forcing_set,
                key="sidebar_forcing_custom",
            )

            st.caption(
                "Thresholds are percentages of total incoming influence. "
                "Edge weights encode relative influence strength."
            )
            description_text = (
                f"Custom network with {custom_config.num_nodes} nodes and "
                f"{len(list(game.edges))} edges."
            )

        target_profile = game.empty_profile(active_value=1)
        initial_profile = game.empty_profile(active_value=0)
        for node_id in forcing_set:
            if node_id in initial_profile:
                initial_profile[node_id] = 1

        max_steps = st.slider(
            "Max cascade steps",
            min_value=1,
            max_value=50,
            value=10,
            key="max_steps",
        )

        run_cascade = st.button("Run cascade", type="primary")

    if description_text:
        st.caption(description_text)

    nodes_list = list(game.nodes)
    solver = PSNESolver(game)
    forcing_finder = ForcingSetFinder(game)

    st.subheader("Game summary")
    st.write(
        f"Nodes: {len(nodes_list)}, Edges: {len(list(game.edges))}, "
        f"Directed: {game.directed}"
    )

    percents = _threshold_percentages(game)
    summary_rows = []
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
    st.dataframe(summary_rows, hide_index=True)

    st.subheader("PSNE of the unrestricted game")
    if len(nodes_list) > 12:
        st.info("Graph too large to enumerate PSNE (n > 12).")
        psne_profiles: list[Dict[Any, Action]] = []
    else:
        psne_result = solver.enumerate_psne_bruteforce()
        psne_profiles = [game.normalize_profile(p) for p in psne_result.profiles]
        st.write(
            f"Found {len(psne_profiles)} PSNE "
            f"(complete={psne_result.complete})."
        )
        for idx, prof in enumerate(psne_profiles):
            st.code(f"PSNE {idx}: {prof}")

    st.subheader("Minimal forcing sets for all-ones profile")
    forcing_result = forcing_finder.minimal_forcing_sets(
        target_profile=target_profile,
        max_size=len(nodes_list),
    )
    if forcing_result.size is None:
        st.info("No forcing set found within the search limits.")
    else:
        st.write(f"Minimal forcing size: {forcing_result.size}")
        if forcing_result.forcing_sets:
            for S in forcing_result.forcing_sets:
                st.code(f"Forcing set: {sorted(S)}")
        else:
            st.write("No forcing sets returned.")

    st.subheader("Cascade from zealots")
    if run_cascade:
        simulator = CascadeSimulator(game)
        fixed_actions = fixed_actions_from_forcing_set(
            forcing_set=forcing_set,
            target_profile=target_profile,
        )
        cascade_result = simulator.run_until_fixpoint(
            initial_profile=initial_profile,
            fixed_actions=fixed_actions if fixed_actions else None,
            max_steps=max_steps,
            detect_cycles=True,
        )
        final_profile = cascade_result.final_profile
        if fixed_actions:
            final_is_psne = solver.is_psne_with_fixed(
                final_profile, fixed_actions
            )
        else:
            final_is_psne = solver.is_psne(final_profile)

        st.write(f"Converged: {cascade_result.converged}")
        st.write(f"Steps: {cascade_result.steps}")
        st.write(f"Final profile is PSNE: {final_is_psne}")
        st.write("Final profile:", final_profile)
    else:
        cascade_result = None
        final_profile = target_profile
        st.info("Click 'Run cascade' to simulate dynamics from zealots.")

    st.subheader("Network snapshot")
    show_profile_plot(
        game,
        profile=final_profile,
        forcing_set=forcing_set,
        title="Final profile" if run_cascade else "All-ones target profile",
    )


if __name__ == "__main__":
    main()
