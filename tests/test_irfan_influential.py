from __future__ import annotations

from src import IrfanMostInfluential, build_custom_game


def test_irfan_mutual_pair_distinguishes_all_ones():
    """
    Mutual pair with θ=1 has PSNE {00, 11}. Either single node at 1 identifies 11.
    """
    game = build_custom_game(
        num_nodes=2,
        thresholds=[1.0, 1.0],
        adjacency=[
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        directed=False,
        label_prefix="",
    )
    solver = IrfanMostInfluential(game)
    target = game.empty_profile(active_value=1)

    result = solver.minimal_distinguishing_sets(target_profile=target)
    assert result.size == 1

    expected = {
        frozenset({("0", 1)}),
        frozenset({("1", 1)}),
    }
    actual = {frozenset(s) for s in result.sets}
    assert actual == expected


def test_irfan_unique_psne_returns_empty_set():
    """
    Directed line with θ=1 everywhere has only the all-zero PSNE.
    The empty observation set already identifies it.
    """
    game = build_custom_game(
        num_nodes=3,
        thresholds=[1.0, 1.0, 1.0],
        adjacency=[
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        directed=True,
        label_prefix="",
    )
    solver = IrfanMostInfluential(game)
    target = game.empty_profile(active_value=0)

    result = solver.minimal_distinguishing_sets(target_profile=target)
    assert result.size == 0
    assert result.sets == [set()]
