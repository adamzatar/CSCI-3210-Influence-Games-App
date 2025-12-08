from __future__ import annotations

import matplotlib.pyplot as plt

from src import (
    build_complete_symmetric_game,
    CascadeSimulator,
    PSNESolver,
    ForcingSetFinder,
    draw_profile_matplotlib,
    draw_cascade_history_matplotlib,
)


def main() -> None:
    game = build_complete_symmetric_game(
        n=3,
        threshold=1.5,
        weight=1.0,
        directed=False,
        label_prefix="v",
    )

    print("Nodes:", list(game.nodes))
    print("Edges:", list(game.edges))

    solver = PSNESolver(game)
    psne_result = solver.enumerate_psne_bruteforce()
    print("PSNE complete:", psne_result.complete)
    for i, prof in enumerate(psne_result.profiles):
        print(f"PSNE {i}:", prof)

    finder = ForcingSetFinder(game)
    target = game.empty_profile(active_value=1)
    forcing_result = finder.minimal_forcing_sets(target_profile=target, max_size=3)
    print("Minimal forcing set size:", forcing_result.size)
    print("Forcing sets:", forcing_result.forcing_sets)

    forcing_set = next(iter(forcing_result.forcing_sets))

    simulator = CascadeSimulator(game)
    initial = game.empty_profile(active_value=0)
    fixed_actions = {node: target[node] for node in forcing_set}

    cascade = simulator.run_until_fixpoint(
        initial_profile=initial,
        fixed_actions=fixed_actions,
        max_steps=10,
        detect_cycles=True,
    )

    print("Cascade converged:", cascade.converged)
    print("Cascade steps:", cascade.steps)
    print("Final profile:", cascade.final_profile)

    fig, _ = draw_cascade_history_matplotlib(
        game,
        cascade=cascade,
        forcing_set=forcing_set,
        max_steps_to_plot=4,
    )
    fig.suptitle(f"Triangle cascade, forcing set = {forcing_set}")
    plt.show()

    fig2, _, _ = draw_profile_matplotlib(
        game,
        profile=cascade.final_profile,
        forcing_set=forcing_set,
        title="Final profile",
    )
    plt.show()


if __name__ == "__main__":
    main()
