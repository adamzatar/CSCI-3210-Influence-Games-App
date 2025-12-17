from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src import build_custom_game
from src.influence_game import InfluenceGame


@dataclass
class PresetDefinition:
    """Metadata for the Streamlit preset selector."""

    key: str
    name: str
    description: str
    notes: str


def get_presets() -> List[PresetDefinition]:
    """Canonical presets that align with the report narrative."""
    return [
        PresetDefinition(
            key="baseline_kuran",
            name="Baseline Kuran mapping",
            description="Fully connected, weight=1 edges. Threshold θ is the number of active neighbors required.",
            notes="Matches Kuran’s baseline: uniform influence, everyone connected. We highlight the lowest vs highest PSNE.",
        ),
        PresetDefinition(
            key="latent_bandwagon",
            name="Latent bandwagon",
            description="Complete network with two threshold tiers so a low-participation PSNE coexists with all-ones.",
            notes="Use the ε control to shave every θ_i slightly; once high thresholds drop enough, the low PSNE disappears.",
        ),
        PresetDefinition(
            key="structure_extension",
            name="Network structure extension",
            description="Star network to show how local neighborhoods change equilibrium selection.",
            notes="Not fully connected: the center needs several leaves before joining; leaves mostly look to the center.",
        ),
        PresetDefinition(
            key="weighted_hub",
            name="Non-uniform influence extension",
            description="Hub-and-spoke where the hub influences leaves more than leaves influence the hub.",
            notes="Weights go beyond Kuran’s uniform baseline; helps illustrate a celebrity-style node.",
        ),
    ]


def get_preset_by_key(key: str) -> PresetDefinition:
    """Look up a preset definition by key."""
    for preset in get_presets():
        if preset.key == key:
            return preset
    raise KeyError(f"Unknown preset key: {key}")


# ---------------------------------------------------------------------------
# Builders for each scenario
# ---------------------------------------------------------------------------


def build_baseline_complete(n: int, theta: float) -> InfluenceGame:
    """Complete graph with uniform thresholds and weight=1 edges."""
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adjacency[i][j] = 1.0
    thresholds = [theta] * n
    return build_custom_game(
        num_nodes=n,
        thresholds=thresholds,
        adjacency=adjacency,
        directed=False,
        label_prefix="",
    )


def build_latent_bandwagon(n_total: int, n_low: int, low_theta: float, high_theta: float) -> InfluenceGame:
    """
    Complete graph with a small core of low-threshold actors and a larger group with higher θ.

    Defaults are chosen so two PSNE exist: the low-core active alone, and all-ones.
    A small downward shift to θ can erase the low PSNE.
    """
    n_low = max(1, min(n_low, n_total - 1))
    adjacency = [[0.0 for _ in range(n_total)] for _ in range(n_total)]
    for i in range(n_total):
        for j in range(n_total):
            if i != j:
                adjacency[i][j] = 1.0
    thresholds = [low_theta] * n_low + [high_theta] * (n_total - n_low)
    return build_custom_game(
        num_nodes=n_total,
        thresholds=thresholds,
        adjacency=adjacency,
        directed=False,
        label_prefix="",
    )


def build_structure_extension(leaves: int, center_theta: float, leaf_theta: float) -> InfluenceGame:
    """Undirected star graph to highlight neighborhood effects."""
    n = leaves + 1
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
    for leaf in range(1, n):
        adjacency[0][leaf] = 1.0
        adjacency[leaf][0] = 1.0
    thresholds = [center_theta] + [leaf_theta] * leaves
    return build_custom_game(
        num_nodes=n,
        thresholds=thresholds,
        adjacency=adjacency,
        directed=False,
        label_prefix="",
    )


def build_weighted_hub(leaves: int, hub_out: float, leaf_out: float, hub_theta: float, leaf_theta: float) -> InfluenceGame:
    """
    Directed hub-and-spoke with heavier hub influence.

    Hub node is labeled 'H'; leaves are labeled L1, L2, ...
    """
    if leaves <= 0:
        raise ValueError("leaves must be positive")

    game = InfluenceGame(directed=True)
    hub = "H"
    game.add_node(hub, threshold=hub_theta, label=hub)

    leaf_nodes = [f"L{i}" for i in range(1, leaves + 1)]
    for leaf in leaf_nodes:
        game.add_node(leaf, threshold=leaf_theta, label=leaf)
        game.add_edge(hub, leaf, weight=hub_out)
        game.add_edge(leaf, hub, weight=leaf_out)

    return game
