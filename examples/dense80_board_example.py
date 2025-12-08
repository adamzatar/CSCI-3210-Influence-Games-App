# examples/dense80_board_example.py
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt

from src.influence_game import InfluenceGame
from src.dynamics import CascadeSimulator
from src.psne import PSNESolver
from src.viz import (
    draw_profile_matplotlib,
    draw_cascade_history_matplotlib,
)


def build_dense_80_example() -> Tuple[InfluenceGame, str]:
    """
    Construct the dense 9-node example with one zealot.

    Graph:
        - Undirected complete graph.
        - 8 regular nodes with threshold 80.0.
        - 1 zealot node "Z" with threshold 0.0.
        - All edges have weight 10.0.

    Returns
    -------
    tuple[InfluenceGame, str]
        The influence game and the zealot node identifier.
    """
    game = InfluenceGame(directed=False)

    regular_nodes = [f"v{i}" for i in range(8)]
    zealot_node = "Z"

    for node in regular_nodes:
        game.add_node(node, threshold=80.0, label=node)
    game.add_node(zealot_node, threshold=0.0, label=zealot_node)

    for u, v in combinations(regular_nodes + [zealot_node], 2):
        game.add_edge(u, v, weight=10.0)

    return game, zealot_node


def main() -> None:
    """
    Build the dense 80% + zealot example, run dynamics, and save figures.

    Outputs:
        - examples/out/dense80_board_example_static.png
          Static picture of the all-active profile, zealot highlighted.

        - examples/out/dense80_board_example_cascade.png
          Cascade history starting from all zeros with the zealot fixed to 1.
    """
    output_dir = Path(__file__).resolve().parent / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    static_path = output_dir / "dense80_board_example_static.png"
    cascade_path = output_dir / "dense80_board_example_cascade.png"

    game, zealot_node = build_dense_80_example()
    forcing_set = {zealot_node}

    # 1. Static: target PSNE profile "everyone active"
    target_profile = game.empty_profile(active_value=1)

    fig_static, _, _ = draw_profile_matplotlib(
        game,
        profile=target_profile,
        forcing_set=forcing_set,
        title="Dense 80% thresholds with one zealot (all active)",
    )
    fig_static.savefig(static_path, dpi=300, bbox_inches="tight")
    plt.close(fig_static)

    # 2. Dynamics: cascade from all zeros with zealot forced to 1
    simulator = CascadeSimulator(game)
    initial_profile = game.empty_profile(active_value=0)
    fixed_actions = {zealot_node: 1}

    cascade = simulator.run_until_fixpoint(
        initial_profile=initial_profile,
        fixed_actions=fixed_actions,
        max_steps=10,
        detect_cycles=True,
    )

    print("Cascade converged:", cascade.converged)
    print("Cascade steps:", cascade.steps)
    print("Final cascade profile:", cascade.final_profile)

    fig_cascade, _ = draw_cascade_history_matplotlib(
        game,
        cascade=cascade,
        forcing_set=forcing_set,
        max_steps_to_plot=4,
    )
    fig_cascade.savefig(cascade_path, dpi=300, bbox_inches="tight")
    plt.close(fig_cascade)

    # 3. PSNE of the restricted game (zealot fixed to 1)
    solver = PSNESolver(game)
    psne_result = solver.enumerate_psne_bruteforce(fixed_actions=fixed_actions)

    print("PSNE enumeration complete:", psne_result.complete)
    print("Number of PSNE with Z fixed to 1:", len(psne_result.profiles))
    for idx, prof in enumerate(psne_result.profiles):
        print(f"PSNE {idx}:", prof)

    print(f"Static figure written to  {static_path.resolve()}")
    print(f"Cascade figure written to {cascade_path.resolve()}")


if __name__ == "__main__":
    main()