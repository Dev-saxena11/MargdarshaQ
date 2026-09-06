"""
qpso.py
--------
Quantum-inspired Particle Swarm Optimization (QPSO) for route optimization.

Why QPSO instead of classical PSO?
    Classical PSO uses velocity + position updates (Newtonian motion model) and can
    get trapped in local optima / suffers from premature convergence in large search
    spaces. QPSO replaces the velocity-based update with a quantum-mechanical model:
    each particle's position is sampled from a probability distribution (delta
    potential well) centered on a stochastic combination of its personal best (pbest)
    and the global best (gbest), attracted toward a "mean best" (mbest) position of
    the whole swarm. This gives:
        - global search guarantee (particles can, in principle, reach any point
          in the search space every iteration - no bounded velocity)
        - fewer control parameters (no inertia weight, no c1/c2 tuning)
        - typically faster convergence & better exploration-exploitation balance

Encoding routing as continuous QPSO variables:
    Each particle is a vector of "priority" values, one per node in the graph.
    We decode a particle into a route via a priority-based greedy/random-key decoder:
    starting at the source node, at each step move to the unvisited neighbor with
    the highest priority value (until destination reached or stuck).
    This "random-key" style encoding is a standard technique for applying
    continuous metaheuristics (PSO/GA) to discrete routing/VRP problems.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple

from app.core.graph_model import TrafficNetwork


@dataclass
class QPSOResult:
    best_route: List[int]
    best_cost: float
    convergence_curve: List[float]        # best cost per iteration
    avg_cost_curve: List[float]            # swarm average cost per iteration
    n_evaluations: int
    feasible: bool


class QPSORouteOptimizer:
    """
    QPSO tuned for single-vehicle shortest / least-cost path optimization on a
    TrafficNetwork. Objective blends travel time, distance, and congestion
    (weights configurable) -- matching the stated goal of minimizing total
    travel time, distance, and traffic congestion simultaneously.
    """

    def __init__(
        self,
        network: TrafficNetwork,
        source: int,
        destination: int,
        n_particles: int = 40,
        max_iter: int = 150,
        alpha_start: float = 1.0,     # contraction-expansion coefficient (start)
        alpha_end: float = 0.3,       # CE coefficient (end) -- controls convergence
        w_time: float = 0.5,
        w_distance: float = 0.3,
        w_congestion: float = 0.2,
        max_hops: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.net = network
        self.source = source
        self.destination = destination
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.w_time = w_time
        self.w_distance = w_distance
        self.w_congestion = w_congestion
        self.n_nodes = network.num_nodes()
        self.max_hops = max_hops or self.n_nodes * 2
        self.rng = np.random.default_rng(seed)
        self._n_evaluations = 0

    # ------------------------------------------------------------------
    # Decoding: continuous particle -> discrete route
    # ------------------------------------------------------------------

    def _decode_route(self, priorities: np.ndarray) -> Optional[List[int]]:
        """
        Greedy priority-based decoder:
        From current node, look at unvisited neighbors, pick the one with the
        highest priority value. Repeat until destination reached, stuck
        (no unvisited neighbors), or max_hops exceeded.
        """
        route = [self.source]
        visited = {self.source}
        current = self.source

        for _ in range(self.max_hops):
            if current == self.destination:
                return route

            neighbors = [v for v in self.net.neighbors(current) if v not in visited]
            if not neighbors:
                return None  # dead end -- infeasible route

            # pick neighbor with highest priority score
            best_v = max(neighbors, key=lambda v: priorities[v])
            route.append(best_v)
            visited.add(best_v)
            current = best_v

        return route if current == self.destination else None

    # ------------------------------------------------------------------
    # Objective / fitness function
    # ------------------------------------------------------------------

    def _evaluate(self, priorities: np.ndarray) -> Tuple[float, Optional[List[int]]]:
        self._n_evaluations += 1
        route = self._decode_route(priorities)
        if route is None:
            return 1e6, None  # heavy penalty for infeasible routes

        cost = self.net.route_cost(route)
        if not cost["feasible"]:
            return 1e6, None

        # normalize components roughly so weights are meaningful
        fitness = (
            self.w_time * cost["time"]
            + self.w_distance * cost["distance"] * 5.0   # scale distance to comparable range
            + self.w_congestion * cost["avg_congestion"] * 20.0
        )
        return fitness, route

    # ------------------------------------------------------------------
    # Main QPSO loop
    # ------------------------------------------------------------------

    def optimize(self, verbose: bool = False) -> QPSOResult:
        n, d = self.n_particles, self.n_nodes

        # Initialize swarm: random priority vectors in [0, 1]
        positions = self.rng.uniform(0, 1, size=(n, d))

        pbest = positions.copy()
        pbest_fitness = np.full(n, np.inf)
        pbest_routes: List[Optional[List[int]]] = [None] * n

        gbest = None
        gbest_fitness = np.inf
        gbest_route: Optional[List[int]] = None

        convergence_curve = []
        avg_cost_curve = []

        for it in range(self.max_iter):
            fitnesses = np.zeros(n)
            for i in range(n):
                fitness, route = self._evaluate(positions[i])
                fitnesses[i] = fitness

                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest[i] = positions[i].copy()
                    pbest_routes[i] = route

                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest = positions[i].copy()
                    gbest_route = route

            # mean best position (mbest) -- centroid of all personal bests
            # this is the key "quantum field center" all particles are drawn toward
            mbest = pbest.mean(axis=0)

            # contraction-expansion coefficient: linearly anneal from explore -> exploit
            alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (it / max(1, self.max_iter - 1))

            # QPSO position update (delta-potential-well model):
            #   u, phi ~ Uniform(0,1)
            #   p = phi * pbest + (1 - phi) * gbest      (stochastic attractor point)
            #   position = p +/- alpha * |mbest - position| * ln(1/u)
            phi = self.rng.uniform(0, 1, size=(n, d))
            p_attractor = phi * pbest + (1 - phi) * gbest

            u = self.rng.uniform(1e-6, 1.0, size=(n, d))  # avoid log(0)
            sign = np.where(self.rng.uniform(0, 1, size=(n, d)) > 0.5, 1.0, -1.0)

            positions = p_attractor + sign * alpha * np.abs(mbest - positions) * np.log(1.0 / u)

            # keep priorities bounded for numerical stability
            positions = np.clip(positions, 0.0, 1.0)

            convergence_curve.append(gbest_fitness)
            avg_cost_curve.append(float(np.mean(fitnesses[np.isfinite(fitnesses)])) if np.any(np.isfinite(fitnesses)) else float("nan"))

            if verbose and (it % 20 == 0 or it == self.max_iter - 1):
                print(f"Iter {it:4d} | gbest_fitness={gbest_fitness:.4f} | route={gbest_route}")

        feasible = gbest_route is not None
        return QPSOResult(
            best_route=gbest_route or [],
            best_cost=gbest_fitness,
            convergence_curve=convergence_curve,
            avg_cost_curve=avg_cost_curve,
            n_evaluations=self._n_evaluations,
            feasible=feasible,
        )


if __name__ == "__main__":
    from app.core.graph_model import generate_synthetic_city_graph

    net = generate_synthetic_city_graph(n_nodes=20, seed=7)
    optimizer = QPSORouteOptimizer(
        network=net, source=0, destination=10,
        n_particles=30, max_iter=100, seed=1,
    )
    result = optimizer.optimize(verbose=True)
    print("\n--- Final Result ---")
    print("Feasible:", result.feasible)
    print("Best route:", result.best_route)
    print("Best cost:", result.best_cost)
    print("Evaluations:", result.n_evaluations)
