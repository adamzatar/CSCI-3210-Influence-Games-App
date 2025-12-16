from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx


Action = int  # 0 or 1


@dataclass
class NodeAttributes:
    """Threshold and optional label for a node."""

    threshold: float
    label: Optional[str] = None


class InfluenceGame:
    """
    Linear-threshold influence game on a graph.

    - Actions are 0 (inactive) or 1 (active/dissent).
    - Edge weights measure how strongly a neighbor matters.
    - A node prefers 1 when incoming active weight >= its threshold.
    Directed graphs use predecessors as "influencers." Undirected graphs
    treat edges as mutual influence.
    """

    def __init__(self, directed: bool = True) -> None:
        """Create an empty game with either a directed or undirected graph."""
        self._directed: bool = directed
        self.G: nx.DiGraph | nx.Graph
        self.G = nx.DiGraph() if directed else nx.Graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: Any, threshold: float, label: Optional[str] = None) -> None:
        """
        Add a node with an absolute threshold theta and an optional label.

        theta is the raw number we compare against summed active weight.
        In the UI we often derive theta from a percent; here it is already absolute.
        """
        if node in self.G:
            raise ValueError(f"Node {node!r} already exists in the graph")
        if threshold < 0:
            raise ValueError("Thresholds must be nonnegative")

        attrs = NodeAttributes(threshold=threshold, label=label)
        self.G.add_node(node, attrs=attrs)

    def add_nodes_from(
        self,
        nodes: Iterable[Any],
        threshold: float,
        labels: Optional[Mapping[Any, str]] = None,
    ) -> None:
        """Add several nodes at once with the same absolute threshold."""
        for node in nodes:
            label = labels[node] if labels and node in labels else None
            self.add_node(node, threshold=threshold, label=label)

    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        """
        Add an edge with a nonnegative weight.

        In directed games, u influences v. In undirected games, edges are mutual.
        """
        if weight < 0:
            raise ValueError("Weights must be nonnegative in this model")
        if u not in self.G or v not in self.G:
            raise ValueError("Both nodes must exist before adding an edge")

        self.G.add_edge(u, v, weight=weight)

    def add_edges_from(
        self,
        edges: Iterable[Tuple[Any, Any]],
        default_weight: float = 1.0,
    ) -> None:
        """Add several edges that share the same weight."""
        for u, v in edges:
            self.add_edge(u, v, weight=default_weight)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def directed(self) -> bool:
        """True if the game uses a directed graph."""
        return self._directed

    @property
    def nodes(self) -> List[Any]:
        """List of nodes in the graph."""
        return list(self.G.nodes())

    @property
    def edges(self) -> List[Tuple[Any, Any]]:
        """List of edges as (u, v) pairs."""
        return list(self.G.edges())

    def node_attrs(self, node: Any) -> NodeAttributes:
        """Return stored attributes for a node."""
        if node not in self.G:
            raise KeyError(f"Node {node!r} is not in the graph")
        return self.G.nodes[node]["attrs"]

    def threshold(self, node: Any) -> float:
        """Get a node's threshold."""
        return self.node_attrs(node).threshold

    def set_threshold(self, node: Any, threshold: float) -> None:
        """Update a node's threshold."""
        if threshold < 0:
            raise ValueError("Thresholds must be nonnegative")
        attrs = self.node_attrs(node)
        attrs.threshold = threshold
        self.G.nodes[node]["attrs"] = attrs

    def label(self, node: Any) -> Optional[str]:
        """Get a node's label, if it has one."""
        return self.node_attrs(node).label

    def set_label(self, node: Any, label: Optional[str]) -> None:
        """Set or clear a node label."""
        attrs = self.node_attrs(node)
        attrs.label = label
        self.G.nodes[node]["attrs"] = attrs

    def weight(self, u: Any, v: Any) -> float:
        """Return the weight on an edge (u, v)."""
        return self.G[u][v]["weight"]

    def set_weight(self, u: Any, v: Any, weight: float) -> None:
        """Update the weight on an edge."""
        if weight < 0:
            raise ValueError("Weights must be nonnegative in this model")
        if not self.G.has_edge(u, v):
            raise KeyError(f"Edge ({u!r}, {v!r}) does not exist")
        self.G[u][v]["weight"] = weight

    # ------------------------------------------------------------------
    # Profiles and influence
    # ------------------------------------------------------------------

    def empty_profile(self, active_value: Action = 0) -> Dict[Any, Action]:
        """
        Create a profile where every node plays the same action.
        Handy for "everyone off" (0) or "everyone on" (1) baselines.
        """
        if active_value not in (0, 1):
            raise ValueError("Action must be 0 or 1")
        return {node: active_value for node in self.nodes}

    def normalize_profile(self, profile: Mapping[Any, Action]) -> Dict[Any, Action]:
        """
        Ensure every node has an explicit 0/1 action in the profile.
        Missing nodes default to 0. Raises if any value is not 0 or 1.
        """
        normalized: Dict[Any, Action] = {}
        for node in self.nodes:
            value = profile.get(node, 0)
            if value not in (0, 1):
                raise ValueError(f"Invalid action {value!r} for node {node!r}")
            normalized[node] = int(value)
        return normalized

    def total_influence(self, profile: Mapping[Any, Action], target: Any) -> float:
        """
        Sum incoming weight from active neighbors of a target node.

        Directed: use predecessors. Undirected: use neighbors.
        If a neighbor plays 1, its edge weight adds to the total.
        """
        if target not in self.G:
            raise KeyError(f"Node {target!r} is not in the graph")

        if isinstance(profile, dict):
            current_profile = profile
        else:
            current_profile = self.normalize_profile(profile)

        if self._directed:
            neighbors: Iterable[Any] = self.G.predecessors(target)
        else:
            neighbors = self.G.neighbors(target)

        total = 0.0
        for neighbor in neighbors:
            if current_profile.get(neighbor, 0) == 1:
                total += self.G[neighbor][target]["weight"]
        return total

    def best_response(
        self,
        profile: Mapping[Any, Action],
        node: Any,
        fixed_actions: Optional[Mapping[Any, Action]] = None,
    ) -> Action:
        """
        Compute the threshold best response for one node.

        If the node is in fixed_actions, return that value.
        Otherwise return 1 when incoming active weight >= theta, else 0.
        Ties go to 1 (consistent with our PSNE and forcing definitions).
        """
        if fixed_actions and node in fixed_actions:
            value = fixed_actions[node]
            if value not in (0, 1):
                raise ValueError("Fixed actions must be 0 or 1")
            return value

        influence = self.total_influence(profile, node)
        theta = self.threshold(node)
        return 1 if influence >= theta else 0

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def canonical_order(self) -> Tuple[List[Any], Dict[Any, int]]:
        """
        Stable node order and a mapping from node to index.
        """
        nodes = sorted(self.nodes)
        node_to_index = {node: i for i, node in enumerate(nodes)}
        return nodes, node_to_index

    def profile_from_bits(self, bits: int) -> Dict[Any, Action]:
        """
        Convert an integer bitmask into a profile using canonical order.
        """
        nodes, node_to_index = self.canonical_order()
        profile: Dict[Any, Action] = {}
        for node in nodes:
            bit_index = node_to_index[node]
            profile[node] = 1 if (bits >> bit_index) & 1 else 0
        return profile

    def copy(self) -> "InfluenceGame":
        """Shallow copy of this game (networkx copies attributes)."""
        new = InfluenceGame(directed=self._directed)
        new.G = self.G.copy()
        return new


if __name__ == "__main__":
    game = InfluenceGame(directed=False)
    game.add_node("A", threshold=1.0, label="A")
    game.add_node("B", threshold=1.0, label="B")
    game.add_edge("A", "B", weight=1.0)

    profile = {"A": 1, "B": 0}
    print("Total influence on B:", game.total_influence(profile, "B"))
    print("Best response for B:", game.best_response(profile, "B"))
