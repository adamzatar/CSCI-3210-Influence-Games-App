# src/dynamics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .influence_game import Action, InfluenceGame


@dataclass
class CascadeResult:
    """
    Result of running a cascade simulation.

    Attributes
    ----------
    history:
        List of profiles, one per time step. history[0] is the initial
        profile, history[-1] is the final profile reached by the algorithm.
    converged:
        True if the dynamics reached a fixed point, False if max_steps
        was reached or a nontrivial cycle was detected.
    steps:
        Number of update steps performed. This is len(history) - 1.
    cycle_start:
        Index in history where the first occurrence of the final profile
        was seen, if a cycle was detected. None if no cycle was detected.
    cycle_length:
        Length of the detected cycle in steps. None if no cycle was detected.
    """

    history: List[Dict[Any, Action]]
    converged: bool
    steps: int
    cycle_start: Optional[int] = None
    cycle_length: Optional[int] = None

    @property
    def final_profile(self) -> Dict[Any, Action]:
        """Return the last profile in the trajectory."""
        return self.history[-1]


class CascadeSimulator:
    """
    Synchronous best response dynamics for an InfluenceGame.

    This class runs iterative best response updates for all nodes at once.
    It is designed to be a thin layer over InfluenceGame, so that PSNE
    and forcing set modules can call it without reimplementing update logic.
    """

    def __init__(self, game: InfluenceGame) -> None:
        """
        Create a cascade simulator for a given InfluenceGame.

        Parameters
        ----------
        game:
            InfluenceGame instance that defines graph, thresholds, and
            local best responses.
        """
        self.game = game
        self._nodes, self._index = self.game.canonical_order()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_profile(self, profile: Mapping[Any, Action]) -> Dict[Any, Action]:
        """
        Normalize a profile so every node has an explicit action.

        Uses the game's normalize_profile to enforce that actions are
        in {0, 1} and that all nodes are included.
        """
        return self.game.normalize_profile(profile)

    def _profile_key(self, profile: Mapping[Any, Action]) -> Tuple[Action, ...]:
        """
        Convert a profile to a hashable key using canonical node order.

        This is used to detect convergence and cycles.
        """
        full = self._normalize_profile(profile)
        return tuple(full[node] for node in self._nodes)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        profile: Mapping[Any, Action],
        fixed_actions: Optional[Mapping[Any, Action]] = None,
    ) -> Dict[Any, Action]:
        """
        Perform one synchronous best response update.

        Parameters
        ----------
        profile:
            Mapping from node to action in {0, 1}. Missing nodes are
            treated as 0.
        fixed_actions:
            Optional mapping from node to fixed action. Nodes in this
            mapping are held to that action regardless of their local
            best response. This is useful to represent committed activists
            or forcing sets.

        Returns
        -------
        Dict[Any, Action]
            New profile after the update.
        """
        current = self._normalize_profile(profile)
        next_profile: Dict[Any, Action] = {}

        for node in self._nodes:
            next_profile[node] = self.game.best_response(
                current,
                node,
                fixed_actions=fixed_actions,
            )

        return next_profile

    def run_until_fixpoint(
        self,
        initial_profile: Mapping[Any, Action],
        fixed_actions: Optional[Mapping[Any, Action]] = None,
        max_steps: int = 100,
        detect_cycles: bool = True,
    ) -> CascadeResult:
        """
        Run synchronous best response dynamics until a fixed point,
        a detected cycle, or a maximum number of steps.

        Parameters
        ----------
        initial_profile:
            Starting profile for the dynamics. Does not need to specify
            every node; missing nodes are treated as 0.
        fixed_actions:
            Optional mapping from node to fixed action. These nodes are
            treated as externally committed.
        max_steps:
            Maximum number of update steps. The trajectory length will
            be at most max_steps + 1.
        detect_cycles:
            If True, detect nontrivial cycles by tracking previously
            seen profiles. If False, only fixed points are considered.

        Returns
        -------
        CascadeResult
            Object containing the full history of profiles and metadata
            about convergence or cycles.
        """
        history: List[Dict[Any, Action]] = []
        seen: Dict[Tuple[Action, ...], int] = {}

        current = self._normalize_profile(initial_profile)
        history.append(current)
        key = self._profile_key(current)
        seen[key] = 0

        converged = False
        cycle_start: Optional[int] = None
        cycle_length: Optional[int] = None

        for step_index in range(1, max_steps + 1):
            next_profile = self.step(current, fixed_actions=fixed_actions)
            history.append(next_profile)

            if next_profile == current:
                converged = True
                break

            key = self._profile_key(next_profile)
            if detect_cycles:
                if key in seen:
                    cycle_start = seen[key]
                    cycle_length = step_index - cycle_start
                    converged = False
                    break
                seen[key] = step_index

            current = next_profile

        steps = len(history) - 1
        return CascadeResult(
            history=history,
            converged=converged,
            steps=steps,
            cycle_start=cycle_start,
            cycle_length=cycle_length,
        )


if __name__ == "__main__":
    # Simple sanity check.
    # Two node undirected graph: A - B
    # Thresholds: both 1.0, weight 1.0.
    # If we fix A to 1 and start from all zeros, B should become 1
    # in a single step and the system should converge.
    game = InfluenceGame(directed=False)
    game.add_node("A", threshold=1.0, label="A")
    game.add_node("B", threshold=1.0, label="B")
    game.add_edge("A", "B", weight=1.0)

    simulator = CascadeSimulator(game)

    initial = game.empty_profile(active_value=0)
    fixed = {"A": 1}

    result = simulator.run_until_fixpoint(initial_profile=initial, fixed_actions=fixed)

    print("History:")
    for t, profile in enumerate(result.history):
        print(f"t = {t}: {profile}")

    print("Converged:", result.converged)
    print("Steps:", result.steps)
    print("Final profile:", result.final_profile)