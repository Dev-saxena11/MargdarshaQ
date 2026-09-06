"""
local_search.py
-----------------
2-opt and or-opt local search operators for CVRPTW routes. Used to build a
"memetic" (hybrid metaheuristic + local search) version of QPSO: the global
search finds promising regions of the solution space, and local search polishes
solutions within those regions -- standard practice for closing the gap between
population-based metaheuristics and true optimality on larger VRP instances.

    2-opt (intra-route):  reverse a segment of a single vehicle's route to
                           remove distance-increasing "crossings" in the path.
    or-opt (inter-route):  relocate a single customer to a different position
                           (possibly a different vehicle's route) if it reduces
                           total cost -- lets the search fix bad vehicle
                           assignments that 2-opt alone can't touch.
"""

from __future__ import annotations
import copy
from typing import List

from app.core.vrp_problem import VRPProblem, evaluate_solution, VRPSolution


def two_opt_pass(problem: VRPProblem, routes: List[List[int]]) -> List[List[int]]:
    """One first-improvement 2-opt pass over every route independently."""
    routes = copy.deepcopy(routes)
    base_fit = evaluate_solution(problem, routes).fitness

    for r_idx, route in enumerate(routes):
        n = len(route)
        if n < 3:
            continue
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    trial_routes = routes[:r_idx] + [candidate] + routes[r_idx + 1:]
                    fit = evaluate_solution(problem, trial_routes).fitness
                    if fit < base_fit - 1e-9:
                        route = candidate
                        base_fit = fit
                        improved = True
            routes[r_idx] = route

    return routes


def or_opt_pass(problem: VRPProblem, routes: List[List[int]]) -> List[List[int]]:
    """One first-improvement or-opt pass: try relocating each customer to every
    other position (same or different route) and keep the move if it helps.

    Important: we look up each customer's CURRENT position fresh on every
    iteration (rather than iterating a stale (route_index, position) snapshot
    taken before any moves) -- routes mutate as we go, so stale indices would
    silently duplicate or drop customers once an earlier move shifts list
    positions out from under a later one.
    """
    routes = copy.deepcopy(routes)
    base_fit = evaluate_solution(problem, routes).fitness
    all_customer_ids = [c.node_id for c in problem.customers]

    for customer in all_customer_ids:
        src_idx, pos = None, None
        for r_idx, route in enumerate(routes):
            if customer in route:
                src_idx = r_idx
                pos = route.index(customer)
                break
        if src_idx is None:
            continue  # shouldn't happen, but guard against inconsistent state

        src_route = routes[src_idx]
        without = src_route[:pos] + src_route[pos + 1:]

        best_fit = base_fit
        best_routes = None

        for dst_idx, dst_route in enumerate(routes):
            candidate_dst = dst_route if dst_idx != src_idx else without
            for insert_at in range(len(candidate_dst) + 1):
                new_dst = candidate_dst[:insert_at] + [customer] + candidate_dst[insert_at:]

                trial = [r[:] for r in routes]
                trial[src_idx] = without
                trial[dst_idx] = new_dst
                if src_idx == dst_idx:
                    trial[src_idx] = new_dst

                fit = evaluate_solution(problem, trial).fitness
                if fit < best_fit - 1e-9:
                    best_fit = fit
                    best_routes = trial

        if best_routes is not None:
            routes = best_routes
            base_fit = best_fit

    return routes


def local_search_refine(problem: VRPProblem, routes: List[List[int]],
                         max_passes: int = 2) -> List[List[int]]:
    """Alternate 2-opt and or-opt passes until no improvement or max_passes reached."""
    current = routes
    for _ in range(max_passes):
        before_fit = evaluate_solution(problem, current).fitness
        current = two_opt_pass(problem, current)
        current = or_opt_pass(problem, current)
        after_fit = evaluate_solution(problem, current).fitness
        if after_fit >= before_fit - 1e-9:
            break
    return current


def routes_to_chromosome(problem: VRPProblem, routes: List[List[int]]):
    """Re-encode a routes structure back into a random-key chromosome, so a
    locally-refined solution can be reinjected into a continuous-encoding
    metaheuristic's population (Lamarckian learning)."""
    import numpy as np
    node_to_customer_idx = {c.node_id: i for i, c in enumerate(problem.customers)}
    chromosome = np.zeros(len(problem.customers))

    for v_idx, route in enumerate(routes):
        n = len(route)
        for k, node_id in enumerate(route):
            c_idx = node_to_customer_idx[node_id]
            # spread priorities evenly within [v_idx, v_idx+1)
            frac = (k + 0.5) / max(n, 1)
            chromosome[c_idx] = v_idx + frac

    return chromosome


if __name__ == "__main__":
    from app.core.graph_model import generate_synthetic_city_graph
    from app.core.vrp_problem import generate_synthetic_vrp, evaluate_solution
    from app.core.classical_baselines_vrp import run_greedy_nn_vrp

    net = generate_synthetic_city_graph(n_nodes=40, seed=7)
    vrp = generate_synthetic_vrp(net, n_customers=18, depot=0, vehicle_capacity=80, seed=3)

    greedy = run_greedy_nn_vrp(vrp)
    print("Before local search:", greedy.best_fitness)

    refined_routes = local_search_refine(vrp, greedy.best_solution.routes, max_passes=3)
    refined_sol = evaluate_solution(vrp, refined_routes)
    print("After local search: ", refined_sol.fitness, "feasible:", refined_sol.feasible)
