# src/psne.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .dynamics import CascadeResult, CascadeSimulator
from .influence_game import Action, InfluenceGame


@dataclass
class PSNEResult:
    """
    Result object for PSNE computations.

    Attributes
    ----------
    profiles:
        List of PSNE profiles found by the algorithm. Each profile is a
        mapping from node to action in {0, 1}.
    complete:
        True if the search procedure explored the entire state space it
        intended to search (for example all 2^n profiles in brute force).
        False if it stopped early due to a max_solutions cap.
    """

    profiles: List[Dict[Any, Action]]
    complete: bool


class PSNESolver:
    """
    Utilities for working with pure strategy Nash equilibria in an
    InfluenceGame.

    This class provides:
      - A PSNE checker for a given profile.
      - A brute force PSNE enumerator for small graphs.
      - A wrapper that uses dynamics to find a PSNE from an initial profile.

    It supports both the unrestricted game and the restricted game where
    some nodes are held fixed (for forcing sets and committed players).
    """

    def __init__(self, game: InfluenceGame) -> None:
        """
        Create a PSNESolver for a given InfluenceGame.

        Parameters
        ----------
        game:
            InfluenceGame instance that defines graph and best responses.
        """
        self.game = game
        self._nodes, self._index = self.game.canonical_order()

    # ------------------------------------------------------------------
    # PSNE checking
    # ------------------------------------------------------------------

    def is_psne(self, profile: Mapping[Any, Action]) -> bool:
        """
        Check whether a profile is a PSNE of the unrestricted game.

        Every node is required to play a best response to the actions of
        all other nodes under the influence threshold rule.

        Parameters
        ----------
        profile:
            Mapping from node to action in {0, 1}. Missing nodes are
            treated as 0.

        Returns
        -------
        bool
            True if the profile is a PSNE, False otherwise.
        """
        current = self.game.normalize_profile(profile)
        for node in self._nodes:
            br = self.game.best_response(current, node, fixed_actions=None)
            if br != current[node]:
                return False
        return True

    def is_psne_with_fixed(
        self,
        profile: Mapping[Any, Action],
        fixed_actions: Mapping[Any, Action],
    ) -> bool:
        """
        Check whether a profile is a PSNE of the restricted game with
        some nodes fixed.

        In this restricted game definition:
          - Nodes in fixed_actions are externally committed and do not
            need to be best responding.
          - Nodes not in fixed_actions must be best responding given the
            committed actions.

        Parameters
        ----------
        profile:
            Candidate joint action profile.
        fixed_actions:
            Mapping from node to fixed action in {0, 1}. The profile must
            be consistent with this mapping for those nodes.

        Returns
        -------
        bool
            True if the profile is a PSNE of the restricted game,
            False otherwise.
        """
        current = self.game.normalize_profile(profile)

        # Check consistency with fixed actions
        for node, value in fixed_actions.items():
            if value not in (0, 1):
                raise ValueError("Fixed actions must be 0 or 1")
            if node not in current:
                raise KeyError(f"Fixed node {node!r} does not exist in the game")
            if current[node] != value:
                return False

        # For non fixed nodes, check best response under influence of fixed
        for node in self._nodes:
            if node in fixed_actions:
                continue
            br = self.game.best_response(current, node, fixed_actions=fixed_actions)
            if br != current[node]:
                return False

        return True

    # ------------------------------------------------------------------
    # Brute force enumeration
    # ------------------------------------------------------------------

    def enumerate_psne_bruteforce(
        self,
        fixed_actions: Optional[Mapping[Any, Action]] = None,
        max_solutions: Optional[int] = None,
    ) -> PSNEResult:
        """
        Enumerate PSNE by brute force for small games.

        This method explores all 2^n profiles (or all profiles consistent
        with fixed_actions) and tests PSNE conditions.

        Parameters
        ----------
        fixed_actions:
            Optional mapping from node to fixed action. If provided, only
            profiles consistent with this mapping are considered. For the
            PSNE test, the restricted game definition is used.
        max_solutions:
            Optional cap on the number of PSNE profiles to collect. If
            not None, the search stops once this many PSNE have been found
            and the result is marked as incomplete.

        Returns
        -------
        PSNEResult
            Object containing the list of PSNE profiles and whether the
            search was complete.

        Notes
        -----
        This is exponential in the number of nodes and is intended only
        for small graphs, for example n up to 16 or so.
        """
        n = len(self._nodes)
        num_states = 1 << n  # 2^n

        psne_profiles: List[Dict[Any, Action]] = []
        complete = True

        # Pre-validate fixed actions, and also precompute which indices
        # they correspond to in the canonical order.
        fixed_bits: Dict[int, Action] = {}
        if fixed_actions is not None:
            for node, value in fixed_actions.items():
                if value not in (0, 1):
                    raise ValueError("Fixed actions must be 0 or 1")
                if node not in self._index:
                    raise KeyError(f"Fixed node {node!r} does not exist in the game")
                fixed_bits[self._index[node]] = value

        for bits in range(num_states):
            # Enforce fixed actions at the bit level to prune early
            if fixed_bits:
                consistent = True
                for idx, value in fixed_bits.items():
                    bit_value = (bits >> idx) & 1
                    if bit_value != value:
                        consistent = False
                        break
                if not consistent:
                    continue

            profile = self.game.profile_from_bits(bits)

            if fixed_actions is None:
                ok = self.is_psne(profile)
            else:
                ok = self.is_psne_with_fixed(profile, fixed_actions=fixed_actions)

            if ok:
                psne_profiles.append(profile)
                if max_solutions is not None and len(psne_profiles) >= max_solutions:
                    complete = False
                    break

        return PSNEResult(profiles=psne_profiles, complete=complete)

    # ------------------------------------------------------------------
    # Dynamics assisted PSNE search
    # ------------------------------------------------------------------

    def find_psne_via_dynamics(
        self,
        initial_profile: Mapping[Any, Action],
        fixed_actions: Optional[Mapping[Any, Action]] = None,
        max_steps: int = 100,
    ) -> Tuple[Optional[Dict[Any, Action]], CascadeResult]:
        """
        Use best response dynamics to search for a PSNE.

        This method wraps the CascadeSimulator and then verifies that the
        final profile in the returned trajectory is indeed a PSNE of the
        appropriate game (unrestricted or restricted).

        Parameters
        ----------
        initial_profile:
            Starting configuration for dynamics.
        fixed_actions:
            Optional mapping from node to fixed action. If provided,
            dynamics are run in the restricted game.
        max_steps:
            Maximum number of update steps in the cascade.

        Returns
        -------
        psne:
            The final profile if it is a PSNE, or None if the dynamics
            did not converge to a PSNE (for example due to a cycle).
        cascade_result:
            The full CascadeResult object from the simulation, which
            includes the trajectory and convergence metadata.
        """
        simulator = CascadeSimulator(self.game)
        result = simulator.run_until_fixpoint(
            initial_profile=initial_profile,
            fixed_actions=fixed_actions,
            max_steps=max_steps,
            detect_cycles=True,
        )
        final_profile = result.final_profile

        if fixed_actions is None:
            ok = self.is_psne(final_profile)
        else:
            ok = self.is_psne_with_fixed(final_profile, fixed_actions=fixed_actions)

        if not ok:
            return None, result
        return final_profile, result


if __name__ == "__main__":
    # Small example to sanity check brute force PSNE enumeration.

    # Complete graph on three nodes with symmetric thresholds.
    # Node thresholds: all 1.5, weights: 1 on every edge.
    #
    # Intuition:
    #   - If exactly one node is active, influence on others is 1 (<1.5),
    #     so they want 0, and the active node wants 0 (influence from others is 0).
    #   - If exactly two nodes are active, they each see influence 1 (<1.5),
    #     so they prefer 0, and the inactive node sees influence 2 (>=1.5),
    #     so it prefers 1.
    #   - If all three are active, each sees influence 2 (>=1.5), so 1 is a
    #     best response.
    #   - If all three are inactive, each sees influence 0 (<1.5), so 0 is a
    #     best response.
    #
    # So the PSNE should be: all zeros and all ones.

    from .influence_game import InfluenceGame

    game = InfluenceGame(directed=False)
    for node in ["A", "B", "C"]:
        game.add_node(node, threshold=1.5, label=node)

    edges: Iterable[Tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
    game.add_edges_from(edges, default_weight=1.0)

    solver = PSNESolver(game)
    result = solver.enumerate_psne_bruteforce()

    print("Found PSNE (complete search:", result.complete, ")")
    for profile in result.profiles:
        print(profile)

    # Example dynamics assisted search from all zeros
    initial = game.empty_profile(active_value=0)
    psne_profile, cascade = solver.find_psne_via_dynamics(initial_profile=initial)

    print("Dynamics based PSNE:", psne_profile)
    print("Cascade steps:", cascade.steps)