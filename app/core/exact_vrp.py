"""
exact_vrp.py
-------------
An exact CVRPTW solver for small instances, so QPSO can be measured against a
proven optimum rather than only against other heuristics.

Why this exists: every other baseline here (GA, SA, standard PSO, greedy) is
itself approximate. Beating them says QPSO is the best of a field of guesses;
it does not say how close to optimal any of them are. On instances small enough
to solve exactly, "QPSO found the optimum" and "QPSO was 3 percent off" are
different claims, and neither can be made honestly without this.

Method. The objective in evaluate_solution is separable: total time and
distance are sums over routes, capacity and time-window penalties are computed
per route, and the idle-vehicle penalty depends only on how many routes are
empty. So the optimum can be assembled in two stages:

  1. For every subset of customers, the cheapest route serving exactly that
     subset, minimised over all orderings. One depth-first walk out of the
     depot enumerates every ordering of every subset, extending a route by one
     stop at a time and carrying the clock forward -- O(1) per step, rather
     than re-pricing a whole sequence each time.

  2. A DP over subsets assembles those routes into a fleet assignment:
     best[v][mask] is the cheapest way to serve exactly `mask` using vehicles
     0..v. That is O(n_vehicles * 3^n), negligible beside stage 1.

Stage 1 is the cost. It visits the sum over subsets of |S| factorial
sequences, which is about e * n factorial. Measured on the synthetic benchmark
instances with the default weights (scripts/measure_exact_vrp.py reproduces
this):

    n = 5      0.00s          n = 8      0.55s
    n = 6      0.01s          n = 9      5.85s
    n = 7      0.07s          n = 10    62.61s

Ten customers is the practical ceiling, and the default guard sits there. The
growth is exponential and meant to be: this is a yardstick for small instances,
not a solver to route a city with.

The objective is evaluate_solution's, term for term and weight for weight. An
exact optimum under a different objective than the one the metaheuristics
minimise would not be a baseline, it would be a second opinion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.classical_baselines_vrp import VRPBenchmarkResult
from app.core.vrp_problem import VRPProblem, evaluate_solution

# Stage 1 grows like e * n factorial, so each extra customer costs roughly ten
# times the last. The guard is a keyword argument rather than a hard limit, so
# a deliberate long run stays possible but nobody trips into one by accident.
EXACT_MAX_CUSTOMERS = 10


class ExactSolverTooLarge(ValueError):
    """Raised when an instance is past the size this solver can finish."""


@dataclass
class _RouteCost:
    """The order-dependent part of one route's cost, plus its total demand."""

    base: float          # weighted time + distance + time-window penalty
    demand: float        # capacity penalty depends on the vehicle, added later
    order: Tuple[int, ...]


def _best_route_per_subset(
    problem: VRPProblem,
    w_time: float,
    w_distance: float,
    time_window_penalty_weight: float,
) -> Dict[int, _RouteCost]:
    """
    The cheapest route for every subset of customers, over all visiting orders.

    One depth-first walk out of the depot. Arriving at a node means "this
    subset, in this order, ending here", so closing the route back to the depot
    yields a candidate for that subset -- which is why a single traversal
    answers every subset at once.
    """
    customers = problem.customers
    n = len(customers)
    depot = problem.depot

    node_id = [c.node_id for c in customers]
    ready = [c.ready_time for c in customers]
    due = [c.due_time for c in customers]
    service = [c.service_time for c in customers]
    demand = [c.demand for c in customers]

    travel_time = problem.travel_time
    travel_distance = problem.travel_distance

    best: Dict[int, _RouteCost] = {0: _RouteCost(0.0, 0.0, ())}

    def walk(mask, last, clock, dist, elapsed, tw, order, load):
        # Close the route here: drive back to the depot and price what we have.
        back_t = travel_time(last, depot, depart_at=clock)
        back_d = travel_distance(last, depot)
        reachable = np.isfinite(back_t)
        total_d = dist + (back_d if reachable else 0.0)
        total_t = elapsed + (back_t if reachable else 0.0)
        base = w_time * total_t + w_distance * total_d + time_window_penalty_weight * tw

        current = best.get(mask)
        if current is None or base < current.base - 1e-12:
            best[mask] = _RouteCost(base, load, order)

        for i in range(n):
            bit = 1 << i
            if mask & bit:
                continue
            t = travel_time(last, node_id[i], depart_at=clock)
            d = travel_distance(last, node_id[i])
            if not np.isfinite(t):
                # Mirrors evaluate_solution: an unreachable leg is a heavy
                # penalty, and the clock does not advance across it.
                walk(mask | bit, node_id[i], clock, dist, elapsed,
                     tw + 1000.0, order + (i,), load + demand[i])
                continue
            arrival = clock + t
            start = ready[i] if arrival < ready[i] else arrival
            wait = start - arrival
            late = start - due[i] if start > due[i] else 0.0
            walk(mask | bit, node_id[i], start + service[i], dist + d,
                 elapsed + t + wait, tw + late, order + (i,), load + demand[i])

    for i in range(n):
        t = travel_time(depot, node_id[i], depart_at=0.0)
        d = travel_distance(depot, node_id[i])
        if not np.isfinite(t):
            walk(1 << i, node_id[i], 0.0, 0.0, 0.0, 1000.0, (i,), demand[i])
            continue
        arrival = t
        start = ready[i] if arrival < ready[i] else arrival
        wait = start - arrival
        late = start - due[i] if start > due[i] else 0.0
        walk(1 << i, node_id[i], start + service[i], d,
             t + wait, late, (i,), demand[i])

    return best


def _submasks(mask: int):
    """Every submask of `mask`, including 0 and `mask` itself."""
    sub = mask
    while True:
        yield sub
        if sub == 0:
            break
        sub = (sub - 1) & mask


def solve_vrp_exact(
    problem: VRPProblem,
    capacity_penalty_weight: float = 50.0,
    time_window_penalty_weight: float = 10.0,
    w_time: float = 0.6,
    w_distance: float = 0.4,
    idle_vehicle_penalty_weight: float = 200.0,
    max_customers: int = EXACT_MAX_CUSTOMERS,
) -> VRPBenchmarkResult:
    """
    The provably optimal routing for a small instance, under exactly the
    objective evaluate_solution scores.

    Raises ExactSolverTooLarge past `max_customers` rather than appearing to
    hang.
    """
    n = len(problem.customers)
    if n > max_customers:
        raise ExactSolverTooLarge(
            f"{n} customers is past this solver's limit of {max_customers}. "
            "The search grows like e * n factorial, so each extra customer "
            "costs roughly ten times the last. Raise max_customers "
            "deliberately if you mean to wait."
        )

    t0 = time.perf_counter()
    n_vehicles = problem.n_vehicles

    if n == 0:
        empty = evaluate_solution(problem, [[] for _ in range(n_vehicles)])
        return VRPBenchmarkResult(
            algorithm="Exact (optimal)", best_solution=empty,
            best_fitness=empty.fitness, runtime_sec=time.perf_counter() - t0,
            n_evaluations=0, convergence_curve=[empty.fitness],
        )

    best_route = _best_route_per_subset(
        problem, w_time, w_distance, time_window_penalty_weight
    )

    full = (1 << n) - 1
    requires_all = getattr(problem, "require_all_vehicles", False)

    def route_cost(mask: int, vehicle: int) -> Optional[float]:
        if mask == 0:
            return idle_vehicle_penalty_weight if requires_all else 0.0
        entry = best_route.get(mask)
        if entry is None:
            return None
        over = entry.demand - problem.capacity_for(vehicle)
        penalty = capacity_penalty_weight * over if over > 0 else 0.0
        return entry.base + penalty

    INF = float("inf")
    prev = [INF] * (full + 1)
    first_pick = [0] * (full + 1)
    for mask in range(full + 1):
        cost = route_cost(mask, 0)
        if cost is not None:
            prev[mask] = cost
            first_pick[mask] = mask

    picks: List[List[int]] = [first_pick]
    for v in range(1, n_vehicles):
        cur = [INF] * (full + 1)
        cur_pick = [0] * (full + 1)
        for mask in range(full + 1):
            best_val, best_sub = INF, 0
            for sub in _submasks(mask):
                head = prev[mask ^ sub]
                if head == INF:
                    continue
                cost = route_cost(sub, v)
                if cost is None:
                    continue
                total = head + cost
                if total < best_val:
                    best_val, best_sub = total, sub
            cur[mask] = best_val
            cur_pick[mask] = best_sub
        prev, picks = cur, picks + [cur_pick]

    if prev[full] == INF:
        raise ExactSolverTooLarge(
            "no assignment covers every customer; this is a bug in the solver"
        )

    # Walk the picks back out into one route per vehicle.
    routes: List[List[int]] = [[] for _ in range(n_vehicles)]
    remaining = full
    for v in range(n_vehicles - 1, -1, -1):
        chosen = picks[v][remaining]
        entry = best_route.get(chosen)
        routes[v] = [problem.customers[i].node_id for i in entry.order] if entry else []
        remaining ^= chosen

    solution = evaluate_solution(
        problem, routes,
        capacity_penalty_weight=capacity_penalty_weight,
        time_window_penalty_weight=time_window_penalty_weight,
        w_time=w_time, w_distance=w_distance,
        idle_vehicle_penalty_weight=idle_vehicle_penalty_weight,
    )
    return VRPBenchmarkResult(
        algorithm="Exact (optimal)",
        best_solution=solution,
        best_fitness=solution.fitness,
        runtime_sec=time.perf_counter() - t0,
        n_evaluations=len(best_route),
        convergence_curve=[solution.fitness],
    )


def optimality_gap(heuristic_fitness: float, optimal_fitness: float) -> float:
    """
    How far above the optimum a heuristic landed, as a percentage.

    Returns 0.0 when the optimum is 0.0, which happens only on a degenerate
    instance with nothing to serve.
    """
    if abs(optimal_fitness) < 1e-12:
        return 0.0
    return 100.0 * (heuristic_fitness - optimal_fitness) / optimal_fitness
