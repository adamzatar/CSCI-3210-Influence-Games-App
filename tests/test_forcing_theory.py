from __future__ import annotations

from src import build_custom_game
from src.forcing import ForcingSetFinder


def _build_mutual_pair() -> tuple:
    """Two-node undirected 50% thresholds. Any single zealot should force all ones."""
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
    target = game.empty_profile(active_value=1)
    return game, target


def _build_triangle() -> tuple:
    """Undirected triangle, weight 1 edges, 50% thresholds (theta=1)."""
    game = build_custom_game(
        num_nodes=3,
        thresholds=[50.0, 50.0, 50.0],
        adjacency=[
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        directed=False,
        label_prefix="",
    )
    target = game.empty_profile(active_value=1)
    return game, target


def test_forcing_sets_mutual_pair_all_ones() -> None:
    """
    Claim: in the mutual pair, either singleton forcing set makes all-ones the only PSNE.
    """
    game, target = _build_mutual_pair()
    finder = ForcingSetFinder(game)

    result = finder.minimal_forcing_sets(target_profile=target, max_size=2)
    assert result.size == 1

    forcing_sets = {frozenset(S) for S in result.forcing_sets}
    assert forcing_sets == {frozenset({"0"}), frozenset({"1"})}


def test_forcing_sets_triangle_all_ones() -> None:
    """
    Claim: in the 50% triangle, any single node forced to 1 makes all-ones the unique PSNE.
    """
    game, target = _build_triangle()
    finder = ForcingSetFinder(game)

    result = finder.minimal_forcing_sets(target_profile=target, max_size=3)
    # With 50% thresholds and weight-1 edges, any single zealot forces all ones.
    assert result.size == 1

    expected_singletons = {
        frozenset({"0"}),
        frozenset({"1"}),
        frozenset({"2"}),
    }
    forcing_sets = {frozenset(S) for S in result.forcing_sets}

    assert forcing_sets == expected_singletons
