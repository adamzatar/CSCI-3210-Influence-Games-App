from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .dynamics import CascadeResult, CascadeSimulator
from .influence_game import Action, InfluenceGame
from .psne import PSNESolver, PSNEResult


@dataclass
class ForcingSetResult:
    """
    Summary of a forcing set search.
    """

    forcing_sets: List[Set[Any]]
    size: Optional[int]
    complete: bool
    searched_subsets: int


class ForcingSetFinder:
    """
    Work with forcing sets and "most influential" nodes.
    """

    def __init__(self, game: InfluenceGame, psne_solver: Optional[PSNESolver] = None) -> None:
        self.game = game
        self.psne_solver = psne_solver or PSNESolver(game)
        self._nodes, self._index = self.game.canonical_order()

    # ------------------------------------------------------------------
    # Core definition: does a set force a target profile?
    # ------------------------------------------------------------------

    def forces_profile(
        self,
        forcing_set: Iterable[Any],
        target_profile: Mapping[Any, Action],
        max_psne: Optional[int] = None,
    ) -> bool:
        """
        Return True if forcing_set makes target_profile the unique PSNE of the restricted game.
        """
        fixed_nodes = set(forcing_set)
        target = self.game.normalize_profile(target_profile)
        fixed_actions: Dict[Any, Action] = {}

        for node in fixed_nodes:
            if node not in target:
                raise KeyError(f"Node {node!r} is not in the game")
            fixed_actions[node] = target[node]

        result: PSNEResult = self.psne_solver.enumerate_psne_bruteforce(
            fixed_actions=fixed_actions if fixed_actions else None,
            max_solutions=max_psne,
        )

        if not result.profiles:
            return False

        if max_psne is not None and not result.complete:
            return False

        if len(result.profiles) != 1:
            return False

        only_profile = self.game.normalize_profile(result.profiles[0])
        return only_profile == target

    # ------------------------------------------------------------------
    # Minimal forcing sets by brute force
    # ------------------------------------------------------------------

    def minimal_forcing_sets(
        self,
        target_profile: Mapping[Any, Action],
        max_size: Optional[int] = None,
        max_sets: Optional[int] = None,
        max_psne_per_check: Optional[int] = 2,
    ) -> ForcingSetResult:
        """
        Find smallest forcing sets for a target profile by checking subsets in order of size.
        """
        num_nodes = len(self._nodes)
        if max_size is None:
            max_size = num_nodes

        forcing_sets: List[Set[Any]] = []
        found_size: Optional[int] = None
        complete = True
        searched_subsets = 0

        target = self.game.normalize_profile(target_profile)

        for size in range(0, max_size + 1):
            if found_size is not None and size > found_size:
                break

            found_at_size = False
            for combo in combinations(self._nodes, size):
                searched_subsets += 1
                candidate = set(combo)

                if self.forces_profile(
                    forcing_set=candidate,
                    target_profile=target,
                    max_psne=max_psne_per_check,
                ):
                    forcing_sets.append(candidate)
                    found_at_size = True
                    if found_size is None:
                        found_size = size

                    if max_sets is not None and len(forcing_sets) >= max_sets:
                        complete = False
                        return ForcingSetResult(
                            forcing_sets=forcing_sets,
                            size=found_size,
                            complete=complete,
                            searched_subsets=searched_subsets,
                        )

            if found_at_size:
                break

        if found_size is None:
            return ForcingSetResult(
                forcing_sets=[],
                size=None,
                complete=complete,
                searched_subsets=searched_subsets,
            )

        return ForcingSetResult(
            forcing_sets=forcing_sets,
            size=found_size,
            complete=complete,
            searched_subsets=searched_subsets,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def minimal_forcing_sets_for_all_active(
        self,
        max_size: Optional[int] = None,
        max_sets: Optional[int] = None,
        max_psne_per_check: Optional[int] = 2,
    ) -> ForcingSetResult:
        """
        Shortcut for target profile where everyone is active.
        """
        target = self.game.empty_profile(active_value=1)
        return self.minimal_forcing_sets(
            target_profile=target,
            max_size=max_size,
            max_sets=max_sets,
            max_psne_per_check=max_psne_per_check,
        )

    def cascade_to_target_via_dynamics(
        self,
        forcing_set: Iterable[Any],
        target_profile: Mapping[Any, Action],
        initial_profile: Optional[Mapping[Any, Action]] = None,
        max_steps: int = 100,
    ) -> Tuple[Optional[Dict[Any, Action]], CascadeResult]:
        """
        Run a cascade with a forcing set and see if it converges to the target.
        """
        target = self.game.normalize_profile(target_profile)
        fixed = {node: target[node] for node in set(forcing_set)}

        if initial_profile is None:
            initial_profile = self.game.empty_profile(active_value=0)

        simulator = CascadeSimulator(self.game)
        result = simulator.run_until_fixpoint(
            initial_profile=initial_profile,
            fixed_actions=fixed,
            max_steps=max_steps,
            detect_cycles=True,
        )

        if not result.converged:
            return None, result

        final_profile = self.game.normalize_profile(result.final_profile)
        return final_profile, result


if __name__ == "__main__":
    # Small sanity example for forcing sets on a triangle.
    from .influence_game import InfluenceGame

    game = InfluenceGame(directed=False)
    for node in ["A", "B", "C"]:
        game.add_node(node, threshold=1.5, label=node)
    edges: Iterable[Tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
    game.add_edges_from(edges, default_weight=1.0)

    finder = ForcingSetFinder(game)
    target = game.empty_profile(active_value=1)
    result = finder.minimal_forcing_sets(target_profile=target, max_size=3)

    print("Minimal forcing sets for all active:")
    print("Size:", result.size)
    for S in result.forcing_sets:
        print("Forcing set:", S)

    final_profile, cascade = finder.cascade_to_target_via_dynamics(
        forcing_set={"A", "B"},
        target_profile=target,
        initial_profile=game.empty_profile(active_value=0),
    )
    print("Dynamics final profile for forcing set {A, B}:", final_profile)
