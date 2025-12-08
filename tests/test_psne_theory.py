from __future__ import annotations

from typing import Dict, List, Tuple

from src import build_custom_game
from src.psne import PSNESolver


def _profile_signatures(game, profiles: List[Dict]) -> set[Tuple[int, ...]]:
    nodes = sorted(game.nodes)
    signatures = set()
    for prof in profiles:
        normalized = game.normalize_profile(prof)
        signatures.add(tuple(normalized[n] for n in nodes))
    return signatures


def test_psne_mutual_pair_50_percent() -> None:
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
    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    signatures = _profile_signatures(game, result.profiles)
    assert signatures == {(0, 0), (1, 1)}


def test_psne_directed_line_50_percent() -> None:
    game = build_custom_game(
        num_nodes=3,
        thresholds=[50.0, 50.0, 50.0],
        adjacency=[
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        directed=True,
        label_prefix="",
    )
    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    signatures = _profile_signatures(game, result.profiles)
    # Node 0 has no incoming influence and prefers 0; others depend on it.
    assert signatures == {(0, 0, 0)}


def test_psne_triangle_50_percent() -> None:
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
    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    signatures = _profile_signatures(game, result.profiles)
    assert signatures == {(0, 0, 0), (1, 1, 1)}
