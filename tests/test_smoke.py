# tests/test_smoke.py
from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib
import pytest

matplotlib.use("Agg")  # use non interactive backend for tests

from src.influence_game import InfluenceGame
from src.dynamics import CascadeSimulator
from src.psne import PSNESolver
from src.forcing import ForcingSetFinder
from src.viz import draw_profile_matplotlib


def build_triangle_game() -> InfluenceGame:
    """
    Complete graph on three nodes with symmetric thresholds and weights.

    Thresholds:
        θ = 1.5 for every node.
    Weights:
        w = 1 on every edge.

    As used in examples in psne.py and forcing.py comments.
    """
    game = InfluenceGame(directed=False)
    for node in ["A", "B", "C"]:
        game.add_node(node, threshold=1.5, label=node)

    edges: Iterable[Tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
    game.add_edges_from(edges, default_weight=1.0)
    return game


def test_cascade_two_node_line_converges():
    """
    Simple two node line A B with thresholds 1.0 and weights 1.0.

    Fix A = 1 and start from all zeros, B should become 1 in one step
    and the dynamics should converge to (1, 1).
    """
    game = InfluenceGame(directed=False)
    game.add_node("A", threshold=1.0, label="A")
    game.add_node("B", threshold=1.0, label="B")
    game.add_edge("A", "B", weight=1.0)

    simulator = CascadeSimulator(game)
    initial = game.empty_profile(active_value=0)
    fixed = {"A": 1}

    result = simulator.run_until_fixpoint(
        initial_profile=initial,
        fixed_actions=fixed,
        max_steps=10,
        detect_cycles=True,
    )

    assert result.converged is True
    final_profile = result.final_profile
    assert final_profile["A"] == 1
    assert final_profile["B"] == 1
    # Should take at least one step but not exceed max_steps
    assert 1 <= result.steps <= 10


def test_psne_triangle_all_zero_and_all_one():
    """
    The triangle game with θ = 1.5 and w = 1 on every edge has exactly
    two PSNE: all zeros and all ones.
    """
    game = build_triangle_game()
    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    assert result.complete is True
    profiles = result.profiles
    assert len(profiles) == 2

    normalized = [game.normalize_profile(p) for p in profiles]
    nodes = sorted(game.nodes)

    all_zero = {v: 0 for v in nodes}
    all_one = {v: 1 for v in nodes}

    assert all_zero in normalized
    assert all_one in normalized


def test_forcing_sets_triangle_all_active():
    """
    In the triangle game with θ = 1.5 and w = 1, minimal forcing sets
    for the all active profile should have size 2.
    """
    game = build_triangle_game()
    finder = ForcingSetFinder(game)

    target = game.empty_profile(active_value=1)
    result = finder.minimal_forcing_sets(target_profile=target, max_size=3)

    assert result.size == 2
    assert len(result.forcing_sets) >= 1
    for S in result.forcing_sets:
        assert len(S) == 2
        # Each forcing set consists of nodes from the game
        for node in S:
            assert node in game.nodes


def test_matplotlib_profile_render_does_not_crash(tmp_path):
    """
    Make sure draw_profile_matplotlib runs without raising and returns
    a valid figure object. Save it to disk to check pipeline end to end.
    """
    game = build_triangle_game()
    profile = game.empty_profile(active_value=1)

    fig, ax, layout_cache = draw_profile_matplotlib(
        game,
        profile=profile,
        title="triangle all active",
    )

    assert fig is not None
    assert ax is not None
    assert layout_cache is not None
    out_file = tmp_path / "triangle_profile.png"
    fig.savefig(out_file)
    assert out_file.exists()