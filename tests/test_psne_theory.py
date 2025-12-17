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


def test_psne_mutual_pair_threshold_one() -> None:
    """
    Two-node undirected mutual influence, θ = 1 on each node.
    Only all-0 and all-1 should be PSNE.
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
    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    signatures = _profile_signatures(game, result.profiles)
    assert signatures == {(0, 0), (1, 1)}


def test_psne_directed_line_threshold_one() -> None:
    """
    Directed line 0 -> 1 -> 2 with θ = 1 on each node.
    Node 0 has no incoming influence, so only all-0 is PSNE.
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
    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    signatures = _profile_signatures(game, result.profiles)
    # Node 0 has no incoming influence and prefers 0; others depend on it.
    assert signatures == {(0, 0, 0)}


def test_psne_triangle_threshold_one() -> None:
    """
    Undirected triangle, weight 1 edges, θ = 1 with two neighbors.
    Either everyone stays off or everyone turns on; mixed states are not best responses.
    """
    game = build_custom_game(
        num_nodes=3,
        thresholds=[1.0, 1.0, 1.0],
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


def test_psne_latent_bandwagon_two_levels() -> None:
    """
    Complete graph with two low-threshold nodes (θ=1) and four higher ones (θ=4).
    There should be three PSNE: all-zeros, the low-threshold pair active alone, and all-ones.
    """
    game = build_custom_game(
        num_nodes=6,
        thresholds=[1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
        adjacency=[
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ],
        directed=False,
        label_prefix="",
    )
    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    signatures = _profile_signatures(game, result.profiles)
    assert signatures == {
        (0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 1),
    }
