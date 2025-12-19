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
from src.most_influential import MostInfluential
from src.influence_game import Action, InfluenceGame
from src.irfan_most_influential import IrfanMostInfluential
from src.psne import PSNESolver
from web.streamlit_app.components.controls import (
    CustomNetworkConfig,
    fixed_actions_from_forcing_set,
    forcing_set_selector,
    mode_selector,
    render_custom_network_controls,
)
from web.streamlit_app.components.plots import show_profile_plot
from web.streamlit_app.state.examples import (
    PresetDefinition,
    build_baseline_complete,
    build_latent_bandwagon,
    build_structure_extension,
    build_weighted_hub,
    get_presets,
)


def _incoming_weight(game: InfluenceGame, node: Any) -> float:
    """Total incoming influence weight for a node."""
    if game.directed:
        neighbors = game.G.predecessors(node)
        return sum(game.weight(neighbor, node) for neighbor in neighbors)
    neighbors = game.G.neighbors(node)
    return sum(game.weight(neighbor, node) for neighbor in neighbors)


def _profile_breakdown(profile: Dict[Any, Action]) -> Dict[str, List[Any]]:
    """Split a profile into sorted active/inactive node lists."""
    active = sorted([n for n, a in profile.items() if a == 1], key=str)
    inactive = sorted([n for n, a in profile.items() if a == 0], key=str)
    return {"active": active, "inactive": inactive}


def _apply_threshold_shift(game: InfluenceGame, epsilon: float) -> None:
    """Subtract epsilon from every threshold, clamped at zero."""
    if epsilon <= 0:
        return
    for node in game.nodes:
        theta = game.threshold(node)
        game.set_threshold(node, max(0.0, theta - epsilon))


def main() -> None:
    """Streamlit dashboard for the influence games project."""
    st.set_page_config(
        page_title="Influence games dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("CSCI 3210 Influence Games")
    st.markdown(
        "Pick a preset or enter a custom network. Thresholds θ are constants (same units as edge weights). "
        "Baselines mirror the report: complete graph (Kuran baseline), latent bandwagon with ε, sparse star, weighted hub."
    )

    preset_options = get_presets()
    preset: PresetDefinition | None = None

    # Sidebar controls
    with st.sidebar:
        st.header("Game configuration")

        mode = mode_selector()
        description_text = ""
        notes_text = ""
        threshold_shift = 0.0

        if mode == "Preset scenario":
            preset = st.selectbox(
                "Preset scenario",
                options=preset_options,
                format_func=lambda p: p.name,
                key="preset_selector",
            )
            notes_text = preset.notes

            if preset.key == "baseline_kuran":
                n = st.slider(
                    "Number of nodes (complete graph)",
                    min_value=3,
                    max_value=12,
                    value=6,
                    key="baseline_n",
                )
                theta_max = max(n - 1, 1)
                default_theta = min(max(theta_max / 2, 1), theta_max)
                theta_value = st.slider(
                    "Neighbors required (θ_i)",
                    min_value=0.0,
                    max_value=float(theta_max),
                    value=float(default_theta),
                    step=0.5,
                    help="With weight=1 on every edge, θ counts how many active neighbors a node needs.",
                    key="baseline_theta",
                )
                game = build_baseline_complete(n=n, theta=theta_value)
                description_text = preset.description

            elif preset.key == "latent_bandwagon":
                n_total = st.slider(
                    "Number of nodes (complete graph)",
                    min_value=5,
                    max_value=10,
                    value=6,
                    key="bandwagon_n",
                )
                n_low = st.slider(
                    "Low-threshold actors",
                    min_value=1,
                    max_value=n_total - 1,
                    value=min(2, n_total - 1),
                    key="bandwagon_n_low",
                    help="We keep at least one higher-threshold node to preserve the latent low PSNE.",
                )
                low_theta = st.slider(
                    "Low θ (core group)",
                    min_value=0.0,
                    max_value=float(max(n_total - 1, 1)),
                    value=1.0,
                    step=0.5,
                    key="bandwagon_low_theta",
                )
                high_theta = st.slider(
                    "High θ (rest of the population)",
                    min_value=float(low_theta),
                    max_value=float(max(n_total - 1, 1)),
                    value=4.0 if n_total >= 6 else float(max(n_total - 2, 2)),
                    step=0.5,
                    key="bandwagon_high_theta",
                    help="Keep this above the number of low-θ actors to retain the low-participation PSNE.",
                )
                threshold_shift = st.slider(
                    "Threshold tweak ε (subtract from every θ_i)",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.0,
                    step=0.1,
                    help="A small ε can erase the low PSNE and push the system to all-ones.",
                    key="bandwagon_shift",
                )
                game = build_latent_bandwagon(
                    n_total=n_total,
                    n_low=n_low,
                    low_theta=low_theta,
                    high_theta=high_theta,
                )
                description_text = preset.description

            elif preset.key == "structure_extension":
                leaves = st.slider(
                    "Number of leaves (star)",
                    min_value=3,
                    max_value=8,
                    value=4,
                    key="structure_leaves",
                )
                center_theta = st.slider(
                    "Center threshold θ_center",
                    min_value=0.0,
                    max_value=float(leaves),
                    value=min(3.0, float(leaves)),
                    step=0.5,
                    key="structure_center_theta",
                )
                leaf_theta = st.slider(
                    "Leaf threshold θ_leaf",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.5,
                    key="structure_leaf_theta",
                )
                game = build_structure_extension(
                    leaves=leaves,
                    center_theta=center_theta,
                    leaf_theta=leaf_theta,
                )
                description_text = preset.description

            else:
                # weighted_hub
                leaves = st.slider(
                    "Number of leaves",
                    min_value=3,
                    max_value=8,
                    value=4,
                    key="weighted_leaves",
                )
                hub_out = st.slider(
                    "Hub influence weight",
                    min_value=1.0,
                    max_value=4.0,
                    value=2.0,
                    step=0.1,
                    key="weighted_hub_out",
                )
                leaf_out = st.slider(
                    "Leaf influence weight (toward hub)",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key="weighted_leaf_out",
                )
                hub_theta = st.slider(
                    "Hub threshold θ_H",
                    min_value=0.0,
                    max_value=max(4.0, float(leaves)),
                    value=2.5,
                    step=0.5,
                    key="weighted_hub_theta",
                )
                leaf_theta = st.slider(
                    "Leaf threshold θ_leaf",
                    min_value=0.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.5,
                    key="weighted_leaf_theta",
                )
                game = build_weighted_hub(
                    leaves=leaves,
                    hub_out=hub_out,
                    leaf_out=leaf_out,
                    hub_theta=hub_theta,
                    leaf_theta=leaf_theta,
                )
                description_text = preset.description

            forcing_set = forcing_set_selector(
                game,
                default_forcing_set=set(),
                key="sidebar_forcing",
            )
        else:
            st.subheader("Custom network")
            custom_config: CustomNetworkConfig = render_custom_network_controls()
            try:
                game = build_custom_game(
                    num_nodes=custom_config.num_nodes,
                    thresholds=custom_config.thresholds,
                    adjacency=custom_config.adjacency,
                    directed=False,
                    label_prefix="",
                )
            except ValueError as exc:
                st.error(f"Invalid custom network: {exc}")
                return

            forcing_set = set(custom_config.forcing_set)

            st.info(
                "Custom networks are undirected; weights come straight from the matrix. "
                "Thresholds are absolute numbers."
            )
            description_text = (
                f"Custom network with {custom_config.num_nodes} nodes and "
                f"{len(list(game.edges))} edges."
            )
            notes_text = "Edit thresholds/weights directly. Zero weight means no edge."
            threshold_shift = st.slider(
                "Optional threshold tweak ε (subtract from every θ_i)",
                min_value=0.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                help="Uniform downward shift to explore sensitivity of equilibria.",
                key="custom_shift",
            )

        target_profile = game.empty_profile(active_value=1)
        initial_profile = game.empty_profile(active_value=0)
        for node_id in forcing_set:
            initial_profile[node_id] = 1
        _apply_threshold_shift(game, epsilon=threshold_shift)

    # Main content
    if preset is not None:
        st.subheader(preset.name)
        st.markdown(description_text)
        if notes_text:
            st.caption(notes_text)
    elif description_text:
        st.subheader("Custom setup")
        st.caption(description_text)
        if notes_text:
            st.caption(notes_text)

    nodes_list = list(game.nodes)
    solver = PSNESolver(game)
    most_influential = MostInfluential(game)
    simulator = CascadeSimulator(game)

    fixed_actions = fixed_actions_from_forcing_set(
        forcing_set=forcing_set,
        target_profile=target_profile,
    )

    cascade_result = simulator.run_until_fixpoint(
        initial_profile=initial_profile,
        fixed_actions=fixed_actions if fixed_actions else None,
        max_steps=25,
        detect_cycles=True,
    )

    final_profile = cascade_result.final_profile
    breakdown = _profile_breakdown(final_profile)

    if fixed_actions:
        final_is_psne = solver.is_psne_with_fixed(final_profile, fixed_actions)
    else:
        final_is_psne = solver.is_psne(final_profile)

    st.markdown("**Model summary**")
    cols_summary = st.columns(3)
    cols_summary[0].metric("Nodes", len(nodes_list))
    cols_summary[1].metric("Edges", len(list(game.edges)))
    cols_summary[2].metric("Directed", "Yes" if game.directed else "No")

    summary_rows = []
    zero_incoming: List[Any] = []
    for node in nodes_list:
        incoming = _incoming_weight(game, node)
        theta = game.threshold(node)
        ratio_display = (
            f"{theta / incoming:.2f}× incoming"
            if incoming > 0
            else "N/A"
        )
        summary_rows.append(
            {
                "node": node,
                "incoming": incoming,
                "θ": theta,
                "θ / incoming": ratio_display,
            }
        )
        if incoming == 0 and theta > 0:
            zero_incoming.append(node)
    st.dataframe(summary_rows, hide_index=True, use_container_width=True)
    st.caption(
        "θ is the incoming weight needed to join. In a complete weight-1 graph, θ counts active neighbors."
    )
    if threshold_shift > 0:
        st.caption(f"Applied ε = {threshold_shift:.2f} to every θ_i (clamped at 0).")
    if zero_incoming:
        st.warning(
            f"Nodes with no incoming influence and positive thresholds cannot flip without forcing: {sorted(zero_incoming)}"
        )

    with st.expander("How this maps to the report", expanded=False):
        st.markdown(
        "- Model: linear-threshold best responses; action 1 = join the revolution.\n"
        "- Thresholds: absolute θ_i in weight units. Baseline mapping of Kuran: complete graph, weight=1, so θ_i counts neighbors.\n"
        "- PSNE: stable participation profiles; we list lowest vs highest PSNE.\n"
        "- Most influential: smallest sets that, when forced to 1, trigger a best-response cascade to everyone active.\n"
        "- Irfan’s indicative nodes: smallest set that uniquely signals a PSNE; not yet implemented here.\n"
        "- Dynamics: optional best-response illustration; equilibrium is defined by PSNE, not by the dynamics."
    )

    st.markdown("**PSNE results (no forcing)**")
    if len(nodes_list) > 12:
        st.info("Graph too large to enumerate PSNE (n > 12).")
        psne_profiles: list[Dict[Any, Action]] = []
        psne_complete = False
    else:
        psne_result = solver.enumerate_psne_bruteforce()
        psne_profiles = [game.normalize_profile(p) for p in psne_result.profiles]
        psne_complete = psne_result.complete
        st.write(f"Found {len(psne_profiles)} PSNE (complete search={psne_complete}).")
        if psne_profiles:
            active_counts = [sum(1 for a in prof.values() if a == 1) for prof in psne_profiles]
            lowest = min(active_counts)
            highest = max(active_counts)
            st.caption(
                f"Lowest PSNE active count: {lowest}; highest: {highest}. "
                "Under our equilibrium story, participation stalls at the lowest PSNE."
            )
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
            if highest == len(nodes_list):
                st.caption("All-ones is a PSNE in this configuration.")
            with st.expander("PSNE as raw profiles", expanded=False):
                for idx, prof in enumerate(psne_profiles, start=1):
                    st.code(f"PSNE {idx}: {prof}")
        else:
            st.write("No PSNE found for this configuration.")

    st.markdown("**Most influential nodes (cascade to all-ones)**")
    mi_sets = most_influential.get_most_influential()
    if not mi_sets:
        st.info("No forcing sets found that drive everyone to 1 under best responses.")
    else:
        size = len(mi_sets[0]) if mi_sets else 0
        st.write(f"Minimal forcing size (cascade-based): {size}")
        for forcing in mi_sets:
            st.code(f"Set: {sorted(forcing)}")

    st.markdown("**Irfan’s indicative nodes**")
    if not psne_profiles:
        st.info("Compute PSNE first to show indicative nodes.")
    else:
        all_ones_profile = game.empty_profile(active_value=1)
        target_index = 0
        for idx, prof in enumerate(psne_profiles):
            if prof == all_ones_profile:
                target_index = idx
                break

        psne_options = [
            (
                idx,
                prof,
                len([n for n, a in prof.items() if a == 1]),
            )
            for idx, prof in enumerate(psne_profiles)
        ]
        selected_label = st.selectbox(
            "Target PSNE",
            options=psne_options,
            format_func=lambda tup: f"PSNE {tup[0]+1}: {tup[2]} active",
            index=target_index,
            key="irfan_target_psne",
        )
        _, target_psne_profile, _ = selected_label

        irfan_solver = IrfanMostInfluential(game)
        try:
            combos = irfan_solver.get_most_influential(target_psne_profile)
            if not combos:
                st.info("No distinguishing set found among the PSNE list.")
            else:
                size = len(combos[0]) if combos else 0
                st.write(f"Minimal indicative set size: {size}")
                for combo in combos:
                    formatted = ", ".join([f"{node}={action}" for node, action in sorted(combo, key=lambda x: str(x[0]))]) or "Empty set"
                    st.code(formatted)
                if not psne_complete:
                    st.caption("PSNE list may be incomplete for large graphs; indicative sets are based on the enumerated PSNE.")
        except ValueError as exc:
            st.warning(str(exc))

    st.markdown("**Outcome from forcing set (best responses to stability)**")
    st.caption("Forced nodes stay at 1; everyone else best-responds until the profile stops changing.")

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

    has_scipy = importlib.util.find_spec("scipy") is not None
    layout_options = {
        "Spring": "spring",
        "Circle": "circular",
    }
    if has_scipy:
        layout_options["Kamada-Kawai"] = "kamada_kawai"
    layout_choice = st.selectbox(
        "Layout for the plot",
        options=list(layout_options.keys()),
        index=0,
    )
    layout_name = layout_options[layout_choice]

    show_profile_plot(
        game,
        profile=final_profile,
        forcing_set=forcing_set,
        title="Outcome with current thresholds/forcing set",
        layout=layout_name,
    )
    st.caption(
        "Red = active, white = inactive, thick border = forced nodes. "
        "This is the outcome after applying the forcing set and optional dynamics."
    )


if __name__ == "__main__":
    main()
