from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .influence_game import Action, InfluenceGame
from .psne import PSNESolver


@dataclass
class IrfanResult:
    """
    Minimal subsets of node-action pairs that uniquely identify a PSNE.

    sets: list of minimal distinguishing subsets (as sets of (node, action)).
    size: size of those subsets, or None if no distinguishing set exists.
    complete: True if we searched all subsets up to the first feasible size.
    """

    sets: List[Set[Tuple[Any, Action]]]
    size: int | None
    complete: bool


class IrfanMostInfluential:
    """
    Find "most indicative" nodes under Professor Irfan's definition.

    Definition: given the list of PSNE and a target PSNE, find the smallest set
    of (node, action) assignments from the target whose values appear only in
    that PSNE and no other PSNE. Observing those nodes' actions uniquely
    identifies the target equilibrium.
    """

    def __init__(self, game: InfluenceGame) -> None:
        self.game = game
        self._nodes, self._index = self.game.canonical_order()
        self._psne_profiles = self._enumerate_psne()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enumerate_psne(self) -> List[Dict[Any, Action]]:
        """Enumerate all PSNE for the current game."""
        solver = PSNESolver(self.game)
        result = solver.enumerate_psne_bruteforce()
        return [self.game.normalize_profile(p) for p in result.profiles]

    def _profile_pairs(self, profile: Mapping[Any, Action]) -> List[Tuple[Any, Action]]:
        """Return profile as a list of (node, action) pairs in canonical order."""
        normalized = self.game.normalize_profile(profile)
        return [(node, normalized[node]) for node in self._nodes]

    def _distinguishes(
        self,
        candidate: Sequence[Tuple[Any, Action]],
        other_profile: Mapping[Any, Action],
    ) -> bool:
        """True if other_profile differs on at least one pair in candidate."""
        normalized_other = self.game.normalize_profile(other_profile)
        for node, action in candidate:
            if normalized_other[node] != action:
                return True
        return False

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def minimal_distinguishing_sets(
        self,
        target_profile: Mapping[Any, Action],
    ) -> IrfanResult:
        """
        Compute minimal subsets of node-action pairs that uniquely identify target_profile.
        """
        target_norm = self.game.normalize_profile(target_profile)
        if target_norm not in self._psne_profiles:
            raise ValueError("Target profile is not a PSNE of this game.")

        other_profiles = [p for p in self._psne_profiles if p != target_norm]
        pairs = self._profile_pairs(target_norm)

        # If target is the only PSNE, the empty set is sufficient.
        if not other_profiles:
            return IrfanResult(sets=[set()], size=0, complete=True)

        minimal_sets: List[Set[Tuple[Any, Action]]] = []
        complete = True

        for size in range(1, len(pairs) + 1):
            found_at_size = False
            for combo in combinations(pairs, size):
                # Skip combos that fail to exclude at least one other PSNE
                if not all(self._distinguishes(combo, other) for other in other_profiles):
                    continue

                found_at_size = True
                combo_set: Set[Tuple[Any, Action]] = set(combo)

                # Keep all combos of this minimal size
                minimal_sets.append(combo_set)

            if found_at_size:
                return IrfanResult(sets=minimal_sets, size=size, complete=complete)

        return IrfanResult(sets=[], size=None, complete=complete)


if __name__ == "__main__":
    # Small sanity run on a triangle: PSNE = {all-0, all-1}.
    game = InfluenceGame(directed=False)
    for node in ["A", "B", "C"]:
        game.add_node(node, threshold=1.0, label=node)
    edges: Iterable[Tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]
    game.add_edges_from(edges, default_weight=1.0)

    solver = IrfanMostInfluential(game)
    psne = solver._psne_profiles
    target = psne[1] if len(psne) > 1 else psne[0]
    result = solver.minimal_distinguishing_sets(target_profile=target)
    print("Minimal distinguishing sets:", result)
