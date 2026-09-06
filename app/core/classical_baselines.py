"""
classical_baselines.py
------------------------
Classical / exact methods used to benchmark QPSO against, per the problem
statement's requirement: "benchmarked against conventional metaheuristics
and exact methods."

Included:
    Exact methods:
        - Dijkstra (shortest path by travel time)
        - A* (shortest path with euclidean heuristic)
    Classical metaheuristics:
        - Genetic Algorithm (GA)
        - Simulated Annealing (SA)
        - Standard (non-quantum) Particle Swarm Optimization (PSO)

All metaheuristics use the SAME priority-vector encoding + decoder as QPSO
(see qpso.py) so comparisons are apples-to-apples: the only thing that
differs between QPSO / GA / SA / standard-PSO is the search strategy used
to explore the same continuous priority-vector space.
"""

from __future__ import annotations
import math
import time
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.core.graph_model import TrafficNetwork


@dataclass
class BenchmarkResult:
    algorithm: str
    best_route: List[int]
    best_cost: float
    runtime_sec: float
    n_evaluations: int
    convergence_curve: List[float]
    feasible: bool


# ---------------------------------------------------------------------------
# Exact methods
# ---------------------------------------------------------------------------

def run_dijkstra(net: TrafficNetwork, source: int, destination: int) -> BenchmarkResult:
    """Exact shortest path minimizing congestion-adjusted travel time."""
    t0 = time.perf_counter()
    G = net.graph

    def weight_fn(u, v, d):
        return net.travel_time(u, v)

    try:
        route = nx.dijkstra_path(G, source, destination, weight=weight_fn)
        cost = net.route_cost(route)
        feasible = True
    except nx.NetworkXNoPath:
        route, cost, feasible = [], {"time": float("inf")}, False

    runtime = time.perf_counter() - t0
    return BenchmarkResult(
        algorithm="Dijkstra (exact)",
        best_route=route,
        best_cost=cost.get("time", float("inf")),
        runtime_sec=runtime,
        n_evaluations=G.number_of_edges(),  # rough proxy
        convergence_curve=[cost.get("time", float("inf"))],
        feasible=feasible,
    )


def run_astar(net: TrafficNetwork, source: int, destination: int) -> BenchmarkResult:
    """Exact shortest path using A* with euclidean-distance heuristic."""
    t0 = time.perf_counter()
    G = net.graph

    def heuristic(u, v):
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        return math.hypot(x1 - x2, y1 - y2) / 5.0  # rough distance-based heuristic

    def weight_fn(u, v, d):
        return net.travel_time(u, v)

    try:
        route = nx.astar_path(G, source, destination, heuristic=heuristic, weight=weight_fn)
        cost = net.route_cost(route)
        feasible = True
    except nx.NetworkXNoPath:
        route, cost, feasible = [], {"time": float("inf")}, False

    runtime = time.perf_counter() - t0
    return BenchmarkResult(
        algorithm="A* (exact)",
        best_route=route,
        best_cost=cost.get("time", float("inf")),
        runtime_sec=runtime,
        n_evaluations=G.number_of_edges(),
        convergence_curve=[cost.get("time", float("inf"))],
        feasible=feasible,
    )


# ---------------------------------------------------------------------------
# Shared route decode + fitness (mirrors qpso.py so comparisons are fair)
# ---------------------------------------------------------------------------

def _decode_route(net: TrafficNetwork, source: int, destination: int,
                   priorities: np.ndarray, max_hops: int) -> Optional[List[int]]:
    route = [source]
    visited = {source}
    current = source
    for _ in range(max_hops):
        if current == destination:
            return route
        neighbors = [v for v in net.neighbors(current) if v not in visited]
        if not neighbors:
            return None
        best_v = max(neighbors, key=lambda v: priorities[v])
        route.append(best_v)
        visited.add(best_v)
        current = best_v
    return route if current == destination else None


def _fitness(net: TrafficNetwork, route: Optional[List[int]],
             w_time: float, w_distance: float, w_congestion: float) -> float:
    if route is None:
        return 1e6
    cost = net.route_cost(route)
    if not cost["feasible"]:
        return 1e6
    return (
        w_time * cost["time"]
        + w_distance * cost["distance"] * 5.0
        + w_congestion * cost["avg_congestion"] * 20.0
    )


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

def run_ga(
    net: TrafficNetwork, source: int, destination: int,
    pop_size: int = 40, max_iter: int = 150,
    mutation_rate: float = 0.1, crossover_rate: float = 0.8,
    w_time: float = 0.5, w_distance: float = 0.3, w_congestion: float = 0.2,
    seed: Optional[int] = None,
) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    n = net.num_nodes()
    max_hops = n * 2
    t0 = time.perf_counter()
    n_eval = 0

    pop = rng.uniform(0, 1, size=(pop_size, n))
    convergence_curve = []
    best_route, best_fit = None, np.inf

    def eval_ind(ind):
        nonlocal n_eval
        n_eval += 1
        route = _decode_route(net, source, destination, ind, max_hops)
        return _fitness(net, route, w_time, w_distance, w_congestion), route

    for it in range(max_iter):
        fits = np.zeros(pop_size)
        routes = [None] * pop_size
        for i in range(pop_size):
            fits[i], routes[i] = eval_ind(pop[i])
            if fits[i] < best_fit:
                best_fit, best_route = fits[i], routes[i]

        convergence_curve.append(best_fit)

        # Tournament selection
        new_pop = []
        for _ in range(pop_size):
            i1, i2 = rng.integers(0, pop_size, size=2)
            winner = pop[i1] if fits[i1] < fits[i2] else pop[i2]
            new_pop.append(winner.copy())
        new_pop = np.array(new_pop)

        # Crossover (uniform)
        for i in range(0, pop_size - 1, 2):
            if rng.uniform() < crossover_rate:
                mask = rng.uniform(0, 1, size=n) < 0.5
                a, b = new_pop[i].copy(), new_pop[i + 1].copy()
                new_pop[i][mask] = b[mask]
                new_pop[i + 1][mask] = a[mask]

        # Mutation
        mutation_mask = rng.uniform(0, 1, size=(pop_size, n)) < mutation_rate
        new_pop[mutation_mask] = rng.uniform(0, 1, size=np.sum(mutation_mask))

        pop = new_pop

    runtime = time.perf_counter() - t0
    return BenchmarkResult(
        algorithm="Genetic Algorithm",
        best_route=best_route or [],
        best_cost=best_fit,
        runtime_sec=runtime,
        n_evaluations=n_eval,
        convergence_curve=convergence_curve,
        feasible=best_route is not None,
    )


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

def run_sa(
    net: TrafficNetwork, source: int, destination: int,
    max_iter: int = 3000, T_start: float = 10.0, T_end: float = 0.01,
    w_time: float = 0.5, w_distance: float = 0.3, w_congestion: float = 0.2,
    seed: Optional[int] = None,
) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    n = net.num_nodes()
    max_hops = n * 2
    t0 = time.perf_counter()
    n_eval = 0

    def eval_pos(pos):
        nonlocal n_eval
        n_eval += 1
        route = _decode_route(net, source, destination, pos, max_hops)
        return _fitness(net, route, w_time, w_distance, w_congestion), route

    current = rng.uniform(0, 1, size=n)
    current_fit, current_route = eval_pos(current)
    best, best_fit, best_route = current.copy(), current_fit, current_route

    convergence_curve = []
    for it in range(max_iter):
        T = T_start * ((T_end / T_start) ** (it / max_iter))  # exponential cooling

        candidate = current + rng.normal(0, 0.15, size=n)
        candidate = np.clip(candidate, 0, 1)
        cand_fit, cand_route = eval_pos(candidate)

        delta = cand_fit - current_fit
        if delta < 0 or rng.uniform() < math.exp(-delta / max(T, 1e-9)):
            current, current_fit, current_route = candidate, cand_fit, cand_route

        if current_fit < best_fit:
            best, best_fit, best_route = current.copy(), current_fit, current_route

        convergence_curve.append(best_fit)

    runtime = time.perf_counter() - t0
    return BenchmarkResult(
        algorithm="Simulated Annealing",
        best_route=best_route or [],
        best_cost=best_fit,
        runtime_sec=runtime,
        n_evaluations=n_eval,
        convergence_curve=convergence_curve,
        feasible=best_route is not None,
    )


# ---------------------------------------------------------------------------
# Standard (classical, non-quantum) PSO -- for QPSO comparison
# ---------------------------------------------------------------------------

def run_standard_pso(
    net: TrafficNetwork, source: int, destination: int,
    n_particles: int = 40, max_iter: int = 150,
    w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
    w_time: float = 0.5, w_distance: float = 0.3, w_congestion: float = 0.2,
    seed: Optional[int] = None,
) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    n = net.num_nodes()
    max_hops = n * 2
    t0 = time.perf_counter()
    n_eval = 0

    def eval_pos(pos):
        nonlocal n_eval
        n_eval += 1
        route = _decode_route(net, source, destination, pos, max_hops)
        return _fitness(net, route, w_time, w_distance, w_congestion), route

    positions = rng.uniform(0, 1, size=(n_particles, n))
    velocities = rng.uniform(-0.1, 0.1, size=(n_particles, n))
    pbest = positions.copy()
    pbest_fit = np.full(n_particles, np.inf)
    pbest_routes = [None] * n_particles
    gbest, gbest_fit, gbest_route = None, np.inf, None

    convergence_curve = []
    for it in range(max_iter):
        for i in range(n_particles):
            fit, route = eval_pos(positions[i])
            if fit < pbest_fit[i]:
                pbest_fit[i], pbest[i], pbest_routes[i] = fit, positions[i].copy(), route
            if fit < gbest_fit:
                gbest_fit, gbest, gbest_route = fit, positions[i].copy(), route

        r1 = rng.uniform(0, 1, size=(n_particles, n))
        r2 = rng.uniform(0, 1, size=(n_particles, n))
        velocities = (
            w * velocities
            + c1 * r1 * (pbest - positions)
            + c2 * r2 * (gbest - positions)
        )
        positions = np.clip(positions + velocities, 0, 1)

        convergence_curve.append(gbest_fit)

    runtime = time.perf_counter() - t0
    return BenchmarkResult(
        algorithm="Standard PSO",
        best_route=gbest_route or [],
        best_cost=gbest_fit,
        runtime_sec=runtime,
        n_evaluations=n_eval,
        convergence_curve=convergence_curve,
        feasible=gbest_route is not None,
    )


if __name__ == "__main__":
    from app.core.graph_model import generate_synthetic_city_graph

    net = generate_synthetic_city_graph(n_nodes=20, seed=7)
    for fn in (run_dijkstra, run_astar):
        r = fn(net, 0, 10)
        print(f"{r.algorithm:20s} | route={r.best_route} | cost={r.best_cost:.3f} | t={r.runtime_sec*1000:.2f}ms")

    for fn in (run_ga, run_sa, run_standard_pso):
        r = fn(net, 0, 10, seed=1) if fn is not run_sa else fn(net, 0, 10, seed=1)
        print(f"{r.algorithm:20s} | route={r.best_route} | cost={r.best_cost:.3f} | t={r.runtime_sec*1000:.2f}ms | evals={r.n_evaluations}")
