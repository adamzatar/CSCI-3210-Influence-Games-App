from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .influence_game import Action, InfluenceGame


@dataclass
class CascadeResult:
    """
    Records a full cascade run (synchronous best responses).

    history: list of profiles over time, starting at t=0.
    converged: True if we hit a fixed point (same profile twice).
    steps: how many update steps were taken.
    cycle_start/cycle_length: where a cycle was detected, else None.
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
    Synchronous best-response dynamics for an InfluenceGame.

    This is the "what happens if everyone updates at once" view.
    PSNE checking is exact; cascades are an optional dynamics lens.
    """

    def __init__(self, game: InfluenceGame) -> None:
        self.game = game
        self._nodes, self._index = self.game.canonical_order()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_profile(self, profile: Mapping[Any, Action]) -> Dict[Any, Action]:
        """Ensure every node has an explicit 0/1 action."""
        return self.game.normalize_profile(profile)

    def _profile_key(self, profile: Mapping[Any, Action]) -> Tuple[Action, ...]:
        """Turn a profile into a tuple in canonical node order."""
        normalized = self._normalize_profile(profile)
        return tuple(normalized[node] for node in self._nodes)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        profile: Mapping[Any, Action],
        fixed_actions: Optional[Mapping[Any, Action]] = None,
    ) -> Dict[Any, Action]:
        """
        One synchronous best-response update.

        Nodes in fixed_actions are pinned; others best-respond to the current profile.
        """
        current_profile = self._normalize_profile(profile)
        next_profile: Dict[Any, Action] = {}

        for node in self._nodes:
            next_profile[node] = self.game.best_response(
                current_profile,
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
        Run synchronous best responses until a fixed point, a cycle, or max_steps.

        This matches the Streamlit "cascade" behavior. If detect_cycles is True,
        we also stop when we see a repeated profile.
        """
        history: List[Dict[Any, Action]] = []
        seen_profiles: Dict[Tuple[Action, ...], int] = {}

        current_profile = self._normalize_profile(initial_profile)
        history.append(current_profile)
        seen_profiles[self._profile_key(current_profile)] = 0

        converged = False
        cycle_start: Optional[int] = None
        cycle_length: Optional[int] = None

        for step_index in range(1, max_steps + 1):
            next_profile = self.step(current_profile, fixed_actions=fixed_actions)
            history.append(next_profile)

            if next_profile == current_profile:
                converged = True
                break

            key = self._profile_key(next_profile)
            if detect_cycles:
                if key in seen_profiles:
                    cycle_start = seen_profiles[key]
                    cycle_length = step_index - cycle_start
                    converged = False
                    break
                seen_profiles[key] = step_index

            current_profile = next_profile

        steps_taken = len(history) - 1
        return CascadeResult(
            history=history,
            converged=converged,
            steps=steps_taken,
            cycle_start=cycle_start,
            cycle_length=cycle_length,
        )


if __name__ == "__main__":
    # Simple sanity check.
    # Two node undirected graph: A - B, thresholds 1.0, weight 1.0.
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
