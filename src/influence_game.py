# src/influence_game.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx


Action = int  # 0 or 1


@dataclass
class NodeAttributes:
    """
    Attributes associated with a node in an influence game.

    Parameters
    ----------
    threshold:
        Activation threshold for the node. A node becomes active (action 1)
        when the weighted sum of incoming influence from active neighbors
        is at least this value.
    label:
        Optional human readable label for visualization or reporting.
    """

    threshold: float
    label: Optional[str] = None


class InfluenceGame:
    """
    Linear threshold influence game on a graph.

    Nodes have binary actions {0, 1}. Edges have nonnegative weights.
    A node becomes active (1) when the sum of weights from active
    incoming neighbors is at least its threshold.

    This class does not know anything about equilibria or cascades.
    It only encodes the game data and local best response behavior.
    Higher level modules (dynamics, psne, forcing) can build on it.
    """

    def __init__(self, directed: bool = True) -> None:
        """
        Create an empty influence game.

        Parameters
        ----------
        directed:
            If True, use a directed graph and treat influence as flowing
            along edge direction (u, v) meaning u influences v.
            If False, use an undirected graph and treat edges as symmetric.
        """
        self._directed: bool = directed
        self.G: nx.DiGraph | nx.Graph
        self.G = nx.DiGraph() if directed else nx.Graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: Any, threshold: float, label: Optional[str] = None) -> None:
        """
        Add a node with a threshold to the influence game.

        Parameters
        ----------
        node:
            Hashable node identifier.
        threshold:
            Activation threshold for this node.
        label:
            Optional human readable label.
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
        """
        Add many nodes at once with a shared threshold.

        Parameters
        ----------
        nodes:
            Iterable of node identifiers.
        threshold:
            Threshold to assign to all nodes in this call.
        labels:
            Optional mapping from node to label. If provided and a node
            is present in labels, that label will be used.
        """
        for node in nodes:
            label = labels[node] if labels and node in labels else None
            self.add_node(node, threshold=threshold, label=label)

    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        """
        Add an edge with a given influence weight.

        Parameters
        ----------
        u, v:
            Endpoints of the edge. For directed games, edge direction is u -> v.
        weight:
            Influence weight. Must be nonnegative.

        Raises
        ------
        ValueError
            If one of the endpoints is missing or the weight is negative.
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
        """
        Add many edges with a shared default weight.

        Parameters
        ----------
        edges:
            Iterable of (u, v) pairs.
        default_weight:
            Weight to assign to each edge.
        """
        for u, v in edges:
            self.add_edge(u, v, weight=default_weight)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def directed(self) -> bool:
        """Return True if the underlying graph is directed."""
        return self._directed

    @property
    def nodes(self) -> List[Any]:
        """Return the list of nodes in a stable order."""
        return list(self.G.nodes())

    @property
    def edges(self) -> List[Tuple[Any, Any]]:
        """Return the list of edges as (u, v) pairs."""
        return list(self.G.edges())

    def node_attrs(self, node: Any) -> NodeAttributes:
        """Return the NodeAttributes object for a given node."""
        if node not in self.G:
            raise KeyError(f"Node {node!r} is not in the graph")
        return self.G.nodes[node]["attrs"]

    def threshold(self, node: Any) -> float:
        """Return the threshold of the given node."""
        return self.node_attrs(node).threshold

    def set_threshold(self, node: Any, threshold: float) -> None:
        """
        Set the threshold of a node.

        Parameters
        ----------
        node:
            Node identifier.
        threshold:
            New threshold value, must be nonnegative.
        """
        if threshold < 0:
            raise ValueError("Thresholds must be nonnegative")
        attrs = self.node_attrs(node)
        attrs.threshold = threshold
        self.G.nodes[node]["attrs"] = attrs

    def label(self, node: Any) -> Optional[str]:
        """Return the label of the given node, if any."""
        return self.node_attrs(node).label

    def set_label(self, node: Any, label: Optional[str]) -> None:
        """
        Set or clear the label of a node.

        Parameters
        ----------
        node:
            Node identifier.
        label:
            New label or None to clear.
        """
        attrs = self.node_attrs(node)
        attrs.label = label
        self.G.nodes[node]["attrs"] = attrs

    def weight(self, u: Any, v: Any) -> float:
        """
        Return the influence weight on edge (u, v).

        For undirected graphs, (u, v) and (v, u) are treated identically.
        """
        return self.G[u][v]["weight"]

    def set_weight(self, u: Any, v: Any, weight: float) -> None:
        """
        Set the influence weight on edge (u, v).

        Parameters
        ----------
        u, v:
            Endpoints of the edge.
        weight:
            New nonnegative weight.
        """
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
        Create a uniform action profile for all nodes.

        Parameters
        ----------
        active_value:
            Value to assign to every node, usually 0 (all inactive)
            or 1 (all active).

        Returns
        -------
        Dict[Any, Action]
            Mapping from node to action.
        """
        if active_value not in (0, 1):
            raise ValueError("Action must be 0 or 1")
        return {node: active_value for node in self.nodes}

    def normalize_profile(self, profile: Mapping[Any, Action]) -> Dict[Any, Action]:
        """
        Normalize a profile mapping so every node has an explicit action.

        Missing nodes default to action 0.

        Parameters
        ----------
        profile:
            Mapping from node to action. Does not need to mention all nodes.

        Returns
        -------
        Dict[Any, Action]
            Mapping from all nodes in the game to actions in {0, 1}.
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
        Compute total incoming influence on a target node.

        Parameters
        ----------
        profile:
            Mapping from node to action in {0, 1}. Nodes not present
            are treated as playing 0.
        target:
            Node whose incoming influence is being measured.

        Returns
        -------
        float
            Sum of weights on edges from active neighbors into target.
        """
        if target not in self.G:
            raise KeyError(f"Node {target!r} is not in the graph")

        normalized = profile if isinstance(profile, dict) else self.normalize_profile(profile)
        if self._directed:
            neighbors: Iterable[Any] = self.G.predecessors(target)
        else:
            neighbors = self.G.neighbors(target)

        total = 0.0
        for neighbor in neighbors:
            if normalized.get(neighbor, 0) == 1:
                total += self.G[neighbor][target]["weight"]
        return total

    def best_response(
        self,
        profile: Mapping[Any, Action],
        node: Any,
        fixed_actions: Optional[Mapping[Any, Action]] = None,
    ) -> Action:
        """
        Compute the best response action for a node.

        Nodes in fixed_actions are treated as externally committed
        and always return their fixed action. This is useful when
        modeling forcing sets or committed activists.

        Parameters
        ----------
        profile:
            Mapping from node to action in {0, 1}.
        node:
            Node whose best response is computed.
        fixed_actions:
            Optional mapping of nodes to fixed actions.

        Returns
        -------
        Action
            1 if the node should be active under threshold rule,
            0 otherwise.
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
        Return a stable node ordering and a mapping to indices.

        This is helpful when representing profiles as bitmasks or
        numpy arrays in PSNE enumeration and dynamics.

        Returns
        -------
        nodes:
            List of nodes in canonical order.
        index:
            Mapping from node to position in that list.
        """
        nodes = sorted(self.nodes)
        index = {node: i for i, node in enumerate(nodes)}
        return nodes, index

    def profile_from_bits(self, bits: int) -> Dict[Any, Action]:
        """
        Convert a bitmask to a node profile using canonical order.

        Parameters
        ----------
        bits:
            Integer whose binary representation encodes a profile.
            Least significant bit corresponds to nodes[0].

        Returns
        -------
        Dict[Any, Action]
            Mapping from node to action in {0, 1}.
        """
        nodes, index = self.canonical_order()
        profile: Dict[Any, Action] = {}
        for node in nodes:
            i = index[node]
            profile[node] = 1 if (bits >> i) & 1 else 0
        return profile

    def copy(self) -> "InfluenceGame":
        """
        Return a shallow copy of this InfluenceGame.

        Node and edge attributes are copied by networkx.
        """
        new = InfluenceGame(directed=self._directed)
        new.G = self.G.copy()
        return new


if __name__ == "__main__":
    # Minimal sanity check example.
    game = InfluenceGame(directed=False)
    game.add_node("A", threshold=1.0, label="A")
    game.add_node("B", threshold=1.0, label="B")
    game.add_edge("A", "B", weight=1.0)

    # Profile where only A is active.
    profile = {"A": 1, "B": 0}

    infl_B = game.total_influence(profile, "B")
    br_B = game.best_response(profile, "B")

    print("Total influence on B:", infl_B)          # expects 1.0
    print("Best response for B:", br_B)             # expects 1 (threshold 1.0)