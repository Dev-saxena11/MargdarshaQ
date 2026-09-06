"""
classical_baselines_vrp.py
----------------------------
GA, SA, and standard PSO adapted to the same VRP random-key encoding as
qpso_vrp.py, so the comparison against QPSO is apples-to-apples (identical
encoding/decoding/fitness -- only the search strategy differs).

Also includes a Greedy Nearest-Neighbor construction heuristic as a simple
non-metaheuristic reference baseline (fast, deterministic, commonly used as
a naive baseline in VRP literature).
"""

from __future__ import annotations
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from app.core.vrp_problem import VRPProblem, evaluate_chromosome, VRPSolution


@dataclass
class VRPBenchmarkResult:
    algorithm: str
    best_solution: Optional[VRPSolution]
    best_fitness: float
    runtime_sec: float
    n_evaluations: int
    convergence_curve: List[float]


# ---------------------------------------------------------------------------
# Genetic Algorithm (VRP)
# ---------------------------------------------------------------------------

def run_ga_vrp(
    problem: VRPProblem, pop_size: int = 50, max_iter: int = 200,
    mutation_rate: float = 0.1, crossover_rate: float = 0.8,
    seed: Optional[int] = None,
) -> VRPBenchmarkResult:
    rng = np.random.default_rng(seed)
    d = len(problem.customers)
    upper = problem.n_vehicles
    t0 = time.perf_counter()
    n_eval = 0

    pop = rng.uniform(0, upper, size=(pop_size, d))
    convergence_curve = []
    best_sol, best_fit = None, np.inf

    def eval_ind(ind):
        nonlocal n_eval
        n_eval += 1
        return evaluate_chromosome(problem, ind)

    for it in range(max_iter):
        fits = np.zeros(pop_size)
        sols = [None] * pop_size
        for i in range(pop_size):
            sol = eval_ind(pop[i])
            fits[i] = sol.fitness
            sols[i] = sol
            if sol.fitness < best_fit:
                best_fit, best_sol = sol.fitness, sol

        convergence_curve.append(best_fit)

        # Tournament selection
        new_pop = []
        for _ in range(pop_size):
            i1, i2 = rng.integers(0, pop_size, size=2)
            winner = pop[i1] if fits[i1] < fits[i2] else pop[i2]
            new_pop.append(winner.copy())
        new_pop = np.array(new_pop)

        # Uniform crossover
        for i in range(0, pop_size - 1, 2):
            if rng.uniform() < crossover_rate:
                mask = rng.uniform(0, 1, size=d) < 0.5
                a, b = new_pop[i].copy(), new_pop[i + 1].copy()
                new_pop[i][mask] = b[mask]
                new_pop[i + 1][mask] = a[mask]

        # Mutation
        mutation_mask = rng.uniform(0, 1, size=(pop_size, d)) < mutation_rate
        new_pop[mutation_mask] = rng.uniform(0, upper, size=np.sum(mutation_mask))

        pop = new_pop

    runtime = time.perf_counter() - t0
    return VRPBenchmarkResult(
        algorithm="Genetic Algorithm", best_solution=best_sol, best_fitness=best_fit,
        runtime_sec=runtime, n_evaluations=n_eval, convergence_curve=convergence_curve,
    )


# ---------------------------------------------------------------------------
# Simulated Annealing (VRP)
# ---------------------------------------------------------------------------

def run_sa_vrp(
    problem: VRPProblem, max_iter: int = 4000,
    T_start: float = 500.0, T_end: float = 0.5,
    seed: Optional[int] = None,
) -> VRPBenchmarkResult:
    rng = np.random.default_rng(seed)
    d = len(problem.customers)
    upper = problem.n_vehicles
    t0 = time.perf_counter()
    n_eval = 0

    def eval_pos(pos):
        nonlocal n_eval
        n_eval += 1
        return evaluate_chromosome(problem, pos)

    current = rng.uniform(0, upper, size=d)
    current_sol = eval_pos(current)
    best, best_sol, best_fit = current.copy(), current_sol, current_sol.fitness

    convergence_curve = []
    step_scale = upper * 0.15

    for it in range(max_iter):
        T = T_start * ((T_end / T_start) ** (it / max_iter))

        candidate = current + rng.normal(0, step_scale, size=d)
        candidate = np.clip(candidate, 0, upper - 1e-9)
        cand_sol = eval_pos(candidate)

        delta = cand_sol.fitness - current_sol.fitness
        if delta < 0 or rng.uniform() < math.exp(-delta / max(T, 1e-9)):
            current, current_sol = candidate, cand_sol

        if current_sol.fitness < best_fit:
            best, best_sol, best_fit = current.copy(), current_sol, current_sol.fitness

        convergence_curve.append(best_fit)

    runtime = time.perf_counter() - t0
    return VRPBenchmarkResult(
        algorithm="Simulated Annealing", best_solution=best_sol, best_fitness=best_fit,
        runtime_sec=runtime, n_evaluations=n_eval, convergence_curve=convergence_curve,
    )


# ---------------------------------------------------------------------------
# Standard (classical) PSO (VRP)
# ---------------------------------------------------------------------------

def run_standard_pso_vrp(
    problem: VRPProblem, n_particles: int = 50, max_iter: int = 200,
    w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
    seed: Optional[int] = None,
) -> VRPBenchmarkResult:
    rng = np.random.default_rng(seed)
    d = len(problem.customers)
    upper = problem.n_vehicles
    t0 = time.perf_counter()
    n_eval = 0

    def eval_pos(pos):
        nonlocal n_eval
        n_eval += 1
        return evaluate_chromosome(problem, pos)

    positions = rng.uniform(0, upper, size=(n_particles, d))
    velocities = rng.uniform(-upper * 0.1, upper * 0.1, size=(n_particles, d))
    pbest = positions.copy()
    pbest_fit = np.full(n_particles, np.inf)
    gbest, gbest_fit, gbest_sol = None, np.inf, None

    convergence_curve = []
    for it in range(max_iter):
        for i in range(n_particles):
            sol = eval_pos(positions[i])
            if sol.fitness < pbest_fit[i]:
                pbest_fit[i], pbest[i] = sol.fitness, positions[i].copy()
            if sol.fitness < gbest_fit:
                gbest_fit, gbest, gbest_sol = sol.fitness, positions[i].copy(), sol

        r1 = rng.uniform(0, 1, size=(n_particles, d))
        r2 = rng.uniform(0, 1, size=(n_particles, d))
        velocities = (
            w * velocities
            + c1 * r1 * (pbest - positions)
            + c2 * r2 * (gbest - positions)
        )
        positions = np.clip(positions + velocities, 0, upper - 1e-9)

        convergence_curve.append(gbest_fit)

    runtime = time.perf_counter() - t0
    return VRPBenchmarkResult(
        algorithm="Standard PSO", best_solution=gbest_sol, best_fitness=gbest_fit,
        runtime_sec=runtime, n_evaluations=n_eval, convergence_curve=convergence_curve,
    )


# ---------------------------------------------------------------------------
# Greedy Nearest-Neighbor construction heuristic (naive baseline, no search)
# ---------------------------------------------------------------------------

def run_greedy_nn_vrp(problem: VRPProblem) -> VRPBenchmarkResult:
    """
    Simple greedy construction: repeatedly assign the nearest unvisited customer
    (by travel time) to the current vehicle's route, respecting capacity; open
    a new vehicle when capacity or time window would be violated.
    No search/optimization -- just a fast deterministic baseline.
    """
    t0 = time.perf_counter()
    unvisited = {c.node_id: c for c in problem.customers}
    routes: List[List[int]] = []

    while unvisited:
        route = []
        current = problem.depot
        current_time = 0.0
        load = 0.0

        while True:
            # find nearest feasible unvisited customer
            best_node, best_time = None, float("inf")
            for node_id, cust in unvisited.items():
                if load + cust.demand > problem.vehicle_capacity:
                    continue
                t = problem.travel_time(current, node_id)
                if not np.isfinite(t):
                    continue
                arrival = current_time + t
                start = max(arrival, cust.ready_time)
                if start > cust.due_time:
                    continue  # would violate time window
                if t < best_time:
                    best_time, best_node = t, node_id

            if best_node is None:
                break  # no feasible next customer -- close this route

            cust = unvisited.pop(best_node)
            arrival = current_time + best_time
            start = max(arrival, cust.ready_time)
            current_time = start + cust.service_time
            load += cust.demand
            current = best_node
            route.append(best_node)

        if route:
            routes.append(route)
        else:
            # no feasible customer could be placed at all -- avoid infinite loop
            # dump one remaining customer into its own route (last resort)
            node_id, cust = next(iter(unvisited.items()))
            unvisited.pop(node_id)
            routes.append([node_id])

    # pad/trim to n_vehicles if needed (extra routes beyond n_vehicles get merged/flagged)
    from app.core.vrp_problem import evaluate_solution
    sol = evaluate_solution(problem, routes)

    runtime = time.perf_counter() - t0
    return VRPBenchmarkResult(
        algorithm="Greedy Nearest-Neighbor", best_solution=sol, best_fitness=sol.fitness,
        runtime_sec=runtime, n_evaluations=len(problem.customers),
        convergence_curve=[sol.fitness],
    )


if __name__ == "__main__":
    from app.core.graph_model import generate_synthetic_city_graph
    from app.core.vrp_problem import generate_synthetic_vrp

    net = generate_synthetic_city_graph(n_nodes=40, seed=7)
    vrp = generate_synthetic_vrp(net, n_customers=15, depot=0, vehicle_capacity=80, seed=3)

    for fn, kwargs in [
        (run_greedy_nn_vrp, {}),
        (run_ga_vrp, {"max_iter": 150, "seed": 1}),
        (run_sa_vrp, {"max_iter": 6000, "seed": 1}),
        (run_standard_pso_vrp, {"max_iter": 150, "seed": 1}),
    ]:
        r = fn(vrp, **kwargs)
        feas = r.best_solution.feasible if r.best_solution else False
        print(f"{r.algorithm:24s} | fitness={r.best_fitness:9.2f} | feasible={feas} | "
              f"t={r.runtime_sec*1000:8.2f}ms | evals={r.n_evaluations}")
