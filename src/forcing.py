# src/forcing.py
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
    Result of a minimal forcing set search.

    Attributes
    ----------
    forcing_sets:
        List of minimal forcing sets found by the algorithm. Each set is
        a set of node identifiers. All sets have the same cardinality.
    size:
        Cardinality of the forcing sets in forcing_sets. None if no
        forcing set was found within the search limits.
    complete:
        True if the search examined all subsets in the intended search
        space (for example all subsets up to max_size). False if the
        search stopped early due to a max_sets cap.
    searched_subsets:
        Total number of subsets of nodes that were tested.
    """

    forcing_sets: List[Set[Any]]
    size: Optional[int]
    complete: bool
    searched_subsets: int


class ForcingSetFinder:
    """
    Tools for working with forcing sets and most influential nodes.

    This class implements an Irfan and Ortiz style notion of forcing sets:

      - Fix a subset S of nodes to the actions they take in a target
        profile x_star.
      - Consider the restricted game where nodes in S are committed
        and only nodes in V \\ S best respond.
      - Compute all PSNE of this restricted game that are consistent
        with the fixed actions of S.
      - If the only such PSNE is exactly x_star, then S is a forcing
        set for x_star.

    For small graphs, minimal forcing sets can be found by brute force
    enumeration over subsets of nodes. For larger graphs this becomes
    expensive, but still works for classroom sized examples.
    """

    def __init__(self, game: InfluenceGame, psne_solver: Optional[PSNESolver] = None) -> None:
        """
        Create a ForcingSetFinder for a given InfluenceGame.

        Parameters
        ----------
        game:
            InfluenceGame instance that defines the graph and thresholds.
        psne_solver:
            Optional PSNESolver to reuse. If not provided, a new one is
            created internally.
        """
        self.game = game
        self.psne_solver = psne_solver or PSNESolver(game)
        self._nodes, self._index = self.game.canonical_order()

    # ------------------------------------------------------------------
    # Core definition: does a given set force a target profile
    # ------------------------------------------------------------------

    def forces_profile(
        self,
        forcing_set: Iterable[Any],
        target_profile: Mapping[Any, Action],
        max_psne: Optional[int] = None,
    ) -> bool:
        """
        Check whether a given subset of nodes forces a target profile.

        Definition:
          - Let x_star be the target profile, normalized to all nodes.
          - Let S be the forcing_set.
          - Create fixed_actions by taking x_star restricted to S.
          - Enumerate all PSNE of the restricted game consistent with
            fixed_actions.
          - If there is exactly one PSNE and it equals x_star, then S
            is a forcing set for x_star.

        Parameters
        ----------
        forcing_set:
            Iterable of node identifiers to treat as committed.
        target_profile:
            Desired target profile x_star. It does not have to be checked
            in advance as a PSNE of the original game, but in typical
            use it will be.
        max_psne:
            Optional cap on the number of PSNE to collect during the
            restricted game enumeration. If provided, enumeration may
            stop early once this many PSNE are found. If this cap is
            hit, the result tends to be a conservative "False".

        Returns
        -------
        bool
            True if forcing_set is a forcing set for target_profile in
            the sense above, False otherwise.
        """
        S: Set[Any] = set(forcing_set)
        if not S:
            # Empty set can only force if the original game already has
            # target_profile as unique PSNE.
            # We handle it through the same machinery below.
            pass

        x_star = self.game.normalize_profile(target_profile)
        fixed_actions: Dict[Any, Action] = {}

        for node in S:
            if node not in x_star:
                raise KeyError(f"Node {node!r} is not in the game")
            fixed_actions[node] = x_star[node]

        # If we know there is more than one PSNE, we are done.
        # Ask the PSNE solver to enumerate PSNE for the restricted game.
        result: PSNEResult = self.psne_solver.enumerate_psne_bruteforce(
            fixed_actions=fixed_actions if fixed_actions else None,
            max_solutions=max_psne,
        )

        if not result.profiles:
            return False

        # If we hit a cap, be conservative and say this does not qualify
        # as a forcing set. The search will be more accurate with a
        # higher max_psne or None.
        if max_psne is not None and not result.complete:
            return False

        if len(result.profiles) != 1:
            return False

        only_profile = self.game.normalize_profile(result.profiles[0])
        return only_profile == x_star

    # ------------------------------------------------------------------
    # Minimal forcing sets by brute force search
    # ------------------------------------------------------------------

    def minimal_forcing_sets(
        self,
        target_profile: Mapping[Any, Action],
        max_size: Optional[int] = None,
        max_sets: Optional[int] = None,
        max_psne_per_check: Optional[int] = 2,
    ) -> ForcingSetResult:
        """
        Search for minimal forcing sets for a target profile by brute force.

        The search proceeds in increasing set size k. For each size k it
        tests all subsets S of that size. Once any forcing set is found
        at size k, the algorithm does not consider larger sizes. All
        forcing sets found at that minimal size are returned.

        Parameters
        ----------
        target_profile:
            Desired target profile x_star. In the political cascades
            example this might be "everyone plays 1".
        max_size:
            Maximum subset size to consider. If None, this defaults to
            the number of nodes in the game.
        max_sets:
            Optional cap on the number of minimal forcing sets to collect.
            If not None and this many forcing sets are found, the search
            stops early and the result is marked incomplete.
        max_psne_per_check:
            Maximum number of PSNE to collect during each restricted
            game enumeration in forces_profile. This can prune searches
            that clearly produce multiple PSNE.

        Returns
        -------
        ForcingSetResult
            Object containing the minimal forcing sets, their size, and
            metadata about the search.
        """
        n = len(self._nodes)
        if max_size is None:
            max_size = n

        forcing_sets: List[Set[Any]] = []
        best_size: Optional[int] = None
        complete = True
        searched_subsets = 0

        # Normalize the target once for consistent equality checks.
        x_star = self.game.normalize_profile(target_profile)

        for size in range(0, max_size + 1):
            # If we already have a best size, do not search larger sizes.
            if best_size is not None and size > best_size:
                break

            # Generate all subsets of given size.
            found_at_this_size = False
            for combo in combinations(self._nodes, size):
                searched_subsets += 1

                S = set(combo)
                if self.forces_profile(
                    forcing_set=S,
                    target_profile=x_star,
                    max_psne=max_psne_per_check,
                ):
                    forcing_sets.append(S)
                    found_at_this_size = True
                    if best_size is None:
                        best_size = size

                    if max_sets is not None and len(forcing_sets) >= max_sets:
                        complete = False
                        return ForcingSetResult(
                            forcing_sets=forcing_sets,
                            size=best_size,
                            complete=complete,
                            searched_subsets=searched_subsets,
                        )

            if found_at_this_size:
                # We have exhausted all subsets of this minimum size.
                # There is no need to search larger sizes.
                break

        if best_size is None:
            # No forcing set found within the constraints.
            return ForcingSetResult(
                forcing_sets=[],
                size=None,
                complete=complete,
                searched_subsets=searched_subsets,
            )

        return ForcingSetResult(
            forcing_sets=forcing_sets,
            size=best_size,
            complete=complete,
            searched_subsets=searched_subsets,
        )

    # ------------------------------------------------------------------
    # Simple convenience wrappers
    # ------------------------------------------------------------------

    def minimal_forcing_sets_for_all_active(
        self,
        max_size: Optional[int] = None,
        max_sets: Optional[int] = None,
        max_psne_per_check: Optional[int] = 2,
    ) -> ForcingSetResult:
        """
        Convenience wrapper for the special case where the target
        profile is "everyone is active" (all ones).

        Parameters
        ----------
        max_size:
            Maximum subset size to consider.
        max_sets:
            Optional cap on the number of minimal forcing sets to collect.
        max_psne_per_check:
            Maximum number of PSNE to collect per restricted game check.

        Returns
        -------
        ForcingSetResult
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
        Run best response dynamics with a forcing set and check whether
        the cascade converges to the target profile.

        This is a Kuran style "cascade" notion of influence, which is
        different from the equilibrium selection notion used in the
        forces_profile definition. It is useful to compare the two.

        Parameters
        ----------
        forcing_set:
            Nodes to treat as committed to their actions in target_profile.
        target_profile:
            Desired target profile.
        initial_profile:
            Starting configuration for dynamics. If None, start from all
            zeros.
        max_steps:
            Maximum number of update steps.

        Returns
        -------
        final_profile:
            The final profile if dynamics converged. None if a cycle was
            detected or max_steps was reached without convergence.
        cascade_result:
            The full CascadeResult from the simulation.
        """
        x_star = self.game.normalize_profile(target_profile)
        S = set(forcing_set)
        fixed_actions: Dict[Any, Action] = {node: x_star[node] for node in S}

        if initial_profile is None:
            initial_profile = self.game.empty_profile(active_value=0)

        simulator = CascadeSimulator(self.game)
        result = simulator.run_until_fixpoint(
            initial_profile=initial_profile,
            fixed_actions=fixed_actions,
            max_steps=max_steps,
            detect_cycles=True,
        )

        if not result.converged:
            return None, result

        final_profile = self.game.normalize_profile(result.final_profile)
        return final_profile, result


if __name__ == "__main__":
    # Small sanity example for forcing sets.

    # Complete graph on 3 nodes A, B, C.
    # Thresholds: 1.5 for each node, weights 1 on each edge.
    # As in psne.py, the PSNE are:
    #   - all zeros
    #   - all ones
    #
    # In this tiny game, a single node forcing itself to 1 is not enough
    # to make all ones the unique PSNE. Any two nodes forcing themselves
    # to 1 is enough.

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
    print("Complete search:", result.complete)
    print("Searched subsets:", result.searched_subsets)
    for S in result.forcing_sets:
        print("Forcing set:", S)

    # Dynamics based check for one particular forcing set, for example {"A", "B"}.
    final_profile, cascade = finder.cascade_to_target_via_dynamics(
        forcing_set={"A", "B"},
        target_profile=target,
        initial_profile=game.empty_profile(active_value=0),
    )
    print("Dynamics final profile for forcing set {A, B}:", final_profile)
    print("Cascade steps:", cascade.steps)