from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .dynamics import CascadeResult, CascadeSimulator
from .influence_game import Action, InfluenceGame


@dataclass
class PSNEResult:
    """
    Tiny container for pure Nash results.

    profiles: every PSNE we found (normalized profiles).
    complete: True when we actually checked all 2^n profiles
      that respect any fixed actions. False only if a max limit
      cut the search early.
    """

    profiles: List[Dict[Any, Action]]
    complete: bool


class PSNESolver:
    """
    Check and list pure strategy Nash equilibria (PSNE) for an InfluenceGame.

    Uses the standard linear-threshold best response:
    a node plays 1 when incoming active weight >= its theta.
    """

    def __init__(self, game: InfluenceGame) -> None:
        self.game = game
        self._nodes, self._index = self.game.canonical_order()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def sum_influence(self, node: Any, profile: Mapping[Any, Action]) -> float:
        """
        Total incoming active weight on a node for a given profile.

        Delegates to InfluenceGame.total_influence to keep semantics consistent.
        """
        normalized = self.game.normalize_profile(profile)
        return self.game.total_influence(normalized, node)

    def best_response_value(
        self,
        node: Any,
        profile: Mapping[Any, Action],
        fixed_actions: Optional[Mapping[Any, Action]] = None,
    ) -> Action:
        """
        Best response for a node under the current profile and optional fixed actions.

        Uses InfluenceGame.best_response so we do not drift from the project's core semantics
        (ties go to 1, thresholds are absolute).
        """
        normalized = self.game.normalize_profile(profile)
        return self.game.best_response(normalized, node, fixed_actions=fixed_actions)

    # ------------------------------------------------------------------
    # PSNE checking
    # ------------------------------------------------------------------

    def is_psne(self, profile: Mapping[Any, Action]) -> bool:
        """
        Test if a profile is a PSNE with no fixed nodes.

        Every node must already be playing its best response given
        the current actions. This is the "revolution stops here"
        notion we use throughout the project.
        """
        current_profile = self.game.normalize_profile(profile)
        for node in self._nodes:
            best = self.best_response_value(node, current_profile, fixed_actions=None)
            if best != current_profile[node]:
                return False
        return True

    def is_psne_with_fixed(
        self,
        profile: Mapping[Any, Action],
        fixed_actions: Mapping[Any, Action],
    ) -> bool:
        """
        Test PSNE when some nodes are fixed to specific actions.

        We use this when a forcing set pins certain players to 1.
        Only non-fixed nodes have to best respond; fixed nodes must
        already match the provided profile.
        """
        current_profile = self.game.normalize_profile(profile)

        # Fixed nodes must already match the profile
        for node, value in fixed_actions.items():
            if value not in (0, 1):
                raise ValueError("Fixed actions must be 0 or 1")
            if node not in current_profile:
                raise KeyError(f"Fixed node {node!r} does not exist in the game")
            if current_profile[node] != value:
                return False

        # Remaining nodes must best respond given the fixed actions
        for node in self._nodes:
            if node in fixed_actions:
                continue
            best = self.best_response_value(node, current_profile, fixed_actions=fixed_actions)
            if best != current_profile[node]:
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
        Enumerate PSNE by checking every profile.

        If fixed_actions is given, we only consider profiles that already
        match those fixed nodes. We bail early if max_solutions is hit,
        setting complete=False so callers know we truncated.
        """
        num_nodes = len(self._nodes)
        num_profiles = 1 << num_nodes  # 2^n

        psne_profiles: List[Dict[Any, Action]] = []
        complete = True

        # Precompute fixed bits for pruning
        fixed_bits: Dict[int, Action] = {}
        if fixed_actions is not None:
            for node, value in fixed_actions.items():
                if value not in (0, 1):
                    raise ValueError("Fixed actions must be 0 or 1")
                if node not in self._index:
                    raise KeyError(f"Fixed node {node!r} does not exist in the game")
                fixed_bits[self._index[node]] = value

        for bits in range(num_profiles):
            if fixed_bits:
                matches_fixed = True
                for bit_index, required_value in fixed_bits.items():
                    bit_value = (bits >> bit_index) & 1
                    if bit_value != required_value:
                        matches_fixed = False
                        break
                if not matches_fixed:
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
        Run synchronous best responses and see if the fixed point is a PSNE.

        Handy when enumeration is too big: we still need to check whether the
        dynamics landed on a true PSNE under the current fixed actions.
        """
        simulator = CascadeSimulator(self.game)
        cascade_result = simulator.run_until_fixpoint(
            initial_profile=initial_profile,
            fixed_actions=fixed_actions,
            max_steps=max_steps,
            detect_cycles=True,
        )
        final_profile = cascade_result.final_profile

        if fixed_actions is None:
            ok = self.is_psne(final_profile)
        else:
            ok = self.is_psne_with_fixed(final_profile, fixed_actions=fixed_actions)

        if not ok:
            return None, cascade_result
        return final_profile, cascade_result


if __name__ == "__main__":
    # Sanity check brute force PSNE enumeration on a triangle.
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

    initial = game.empty_profile(active_value=0)
    psne_profile, cascade = solver.find_psne_via_dynamics(initial_profile=initial)

    print("Dynamics based PSNE:", psne_profile)
    print("Cascade steps:", cascade.steps)
