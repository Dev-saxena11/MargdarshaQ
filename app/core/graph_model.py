"""
graph_model.py
----------------
Represents the transportation network as a weighted graph.

Each edge carries:
    - distance   (km)
    - base_time  (minutes, free-flow travel time)
    - congestion_factor (>=1.0 multiplier applied to base_time to get real travel time)

Real-time / simulated traffic updates congestion_factor per edge, which is what
lets the routing engine react to dynamic conditions instead of just static shortest path.
"""

from __future__ import annotations
import random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional



@dataclass
class TrafficIncident:
    u: int
    v: int
    factor: float
    start_time: float = 0.0
    duration_min: Optional[float] = None

    def is_active(self, current_time: float) -> bool:
        if current_time < self.start_time:
            return False
        if self.duration_min is not None and current_time > (self.start_time + self.duration_min):
            return False
        return True


@dataclass
class TrafficNetwork:
    """Wraps a networkx.Graph with transportation-specific edge attributes and dynamic traffic features."""

    graph: nx.Graph = field(default_factory=nx.Graph)
    incidents: List[TrafficIncident] = field(default_factory=list)

    # ---------- construction helpers ----------

    def add_node(self, node_id: int, x: float, y: float, **attrs):
        """x, y are coordinates (used for visualization + heuristic distance)."""
        self.graph.add_node(node_id, x=x, y=y, **attrs)

    def add_edge(
        self,
        u: int,
        v: int,
        distance: float,
        base_speed_kmph: float = 40.0,
        congestion_factor: float = 1.0,
    ):
        """
        distance: km
        base_speed_kmph: free-flow speed on this road segment
        congestion_factor: >=1.0, multiplies travel time (1.0 = no congestion)
        """
        base_time = (distance / base_speed_kmph) * 60.0  # minutes
        self.graph.add_edge(
            u, v,
            distance=distance,
            base_time=base_time,
            congestion_factor=congestion_factor,
        )

    # ---------- dynamic traffic ----------

    def get_edge_congestion(self, u: int, v: int, current_time: Optional[float] = None) -> float:
        """Calculate effective congestion factor considering base factor, active incidents, and time-of-day surge."""
        if not self.graph.has_edge(u, v):
            return 1.0
        edge = self.graph[u][v]
        cong = edge.get("congestion_factor", 1.0)

        # Check for active incidents on this edge
        for inc in self.incidents:
            if (inc.u == u and inc.v == v) or (inc.u == v and inc.v == u):
                if current_time is None or inc.is_active(current_time):
                    cong = max(cong, inc.factor)

        # Time-varying rush-hour modulation (if current_time is provided)
        if current_time is not None:
            # Gaussian rush hour peak at t=120 min with std=30 min
            rush_surge = 0.5 * np.exp(-((current_time - 120.0) / 30.0) ** 2)
            cong += rush_surge

        return max(1.0, float(cong))

    def travel_time(self, u: int, v: int, current_time: Optional[float] = None) -> float:
        """Current effective travel time (minutes) for edge u-v, congestion-adjusted."""
        edge = self.graph[u][v]
        cong = self.get_edge_congestion(u, v, current_time)
        return edge["base_time"] * cong

    def update_congestion(self, u: int, v: int, factor: float):
        """Set a new congestion factor for an edge (simulating real-time traffic)."""
        if self.graph.has_edge(u, v):
            self.graph[u][v]["congestion_factor"] = max(1.0, factor)

    def apply_incident(
        self, u: int, v: int, factor: float, start_time: float = 0.0, duration_min: Optional[float] = None
    ) -> TrafficIncident:
        """Register a traffic incident/bottleneck on edge (u, v)."""
        inc = TrafficIncident(u=u, v=v, factor=factor, start_time=start_time, duration_min=duration_min)
        self.incidents.append(inc)
        self.update_congestion(u, v, factor)
        return inc

    def clear_incidents(self):
        """Clear all active incidents and reset edge factors."""
        self.incidents.clear()

    def randomize_congestion(self, seed: Optional[int] = None,
                               low: float = 1.0, high: float = 3.0):
        """Simulate real-time traffic by randomizing congestion on every edge."""
        rng = random.Random(seed)
        for u, v in self.graph.edges():
            self.graph[u][v]["congestion_factor"] = rng.uniform(low, high)

    # ---------- route evaluation ----------

    def route_cost(self, route: List[int]) -> Dict[str, float]:
        """
        Given a route as a list of node ids, return total distance, total travel time,
        and average congestion encountered. Returns inf cost if route is disconnected.
        """
        total_distance = 0.0
        total_time = 0.0
        congestion_sum = 0.0
        n_edges = 0

        for u, v in zip(route[:-1], route[1:]):
            if not self.graph.has_edge(u, v):
                return {"distance": float("inf"), "time": float("inf"),
                        "avg_congestion": float("inf"), "feasible": False}
            total_distance += self.graph[u][v]["distance"]
            total_time += self.travel_time(u, v)
            congestion_sum += self.graph[u][v]["congestion_factor"]
            n_edges += 1

        avg_congestion = congestion_sum / n_edges if n_edges else 0.0
        return {
            "distance": total_distance,
            "time": total_time,
            "avg_congestion": avg_congestion,
            "feasible": True,
        }

    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def neighbors(self, node: int) -> List[int]:
        return list(self.graph.neighbors(node))


# ---------------------------------------------------------------------------
# Synthetic network generator
# ---------------------------------------------------------------------------

def generate_synthetic_city_graph(
    n_nodes: int = 30,
    connectivity: float = 0.15,
    seed: int = 42,
    grid_size: float = 100.0,
) -> TrafficNetwork:
    """
    Generates a random, connected, city-like road network.

    Strategy:
        1. Scatter nodes randomly in a 2D plane (simulating intersections).
        2. Connect each node to its k-nearest neighbors (roads follow proximity,
           like a real street grid) -- NOT pure random edges, so the graph
           looks like a plausible city rather than a random graph.
        3. Ensure the graph is fully connected (add bridge edges if needed).
        4. Assign distance from euclidean coordinates + randomized base speed.
    """
    rng = random.Random(seed)
    net = TrafficNetwork()

    # 1. Place nodes randomly
    coords = {}
    for i in range(n_nodes):
        x, y = rng.uniform(0, grid_size), rng.uniform(0, grid_size)
        coords[i] = (x, y)
        net.add_node(i, x=x, y=y)

    def euclidean(a, b):
        (x1, y1), (x2, y2) = coords[a], coords[b]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    # 2. k-nearest neighbor connections (k derived from connectivity param)
    k = max(2, int(n_nodes * connectivity))
    for i in range(n_nodes):
        dists = sorted(
            [(j, euclidean(i, j)) for j in range(n_nodes) if j != i],
            key=lambda t: t[1],
        )
        for j, dist_km_raw in dists[:k]:
            if not net.graph.has_edge(i, j):
                distance_km = round(dist_km_raw / 5.0, 2)  # scale plane units -> km
                distance_km = max(distance_km, 0.3)
                base_speed = rng.choice([30, 40, 50, 60])  # kmph, road-type variety
                net.add_edge(i, j, distance=distance_km, base_speed_kmph=base_speed)

    # 3. Ensure connectivity: link any isolated components with a bridge edge
    if not nx.is_connected(net.graph):
        components = list(nx.connected_components(net.graph))
        for a, b in zip(components[:-1], components[1:]):
            u, v = next(iter(a)), next(iter(b))
            distance_km = max(round(euclidean(u, v) / 5.0, 2), 0.3)
            net.add_edge(u, v, distance=distance_km, base_speed_kmph=40)

    # 4. Randomize initial congestion (simulated "current traffic")
    net.randomize_congestion(seed=seed, low=1.0, high=2.5)

    return net


if __name__ == "__main__":
    # quick smoke test
    net = generate_synthetic_city_graph(n_nodes=15, seed=1)
    print(f"Nodes: {net.num_nodes()}, Edges: {net.graph.number_of_edges()}")
    print("Connected:", nx.is_connected(net.graph))
    sample_route = list(nx.shortest_path(net.graph, 0, 5))
    print("Sample shortest path 0->5:", sample_route)
    print("Cost:", net.route_cost(sample_route))
