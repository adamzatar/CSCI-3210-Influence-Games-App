from __future__ import annotations

from typing import Dict, List, Tuple

from src import build_custom_game
from src.psne import PSNESolver


def _profile_signatures(game, profiles: List[Dict]) -> set[Tuple[int, ...]]:
    """
    Normalize and turn profiles into tuples for stable comparison.
    Keeps the canonical node order so set equality is easy to read.
    """
    nodes = sorted(game.nodes)
    signatures = set()
    for prof in profiles:
        normalized = game.normalize_profile(prof)
        signatures.add(tuple(normalized[n] for n in nodes))
    return signatures


def test_psne_mutual_pair_50_percent() -> None:
    """
    Two-node undirected mutual influence, 50% thresholds.
    Classic claim: only all-0 and all-1 are PSNE.
    """
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
    """
    Directed line 0 -> 1 -> 2 with 50% thresholds.
    Node 0 has no incoming influence, so it sticks at 0.
    Others never see enough active weight to flip; only all-0 is PSNE.
    """
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
    """
    Undirected triangle, weight 1 edges, 50% thresholds (theta=1 with two neighbors).
    Either everyone stays off or everyone turns on; mixed states are not best responses.
    """
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
