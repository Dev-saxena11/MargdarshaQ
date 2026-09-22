"""
tests/test_exact_vrp.py
------------------------
The exact CVRPTW solver in app/core/exact_vrp.py.

A solver claiming optimality has to be checked against something that cannot be
wrong for the same reason it is. So the central test here does not compare
against a fixture or a remembered number: it enumerates every assignment of
customers to vehicles and every visiting order within each, scores each one
with evaluate_solution, and requires the exact solver to match the best. That
is only affordable for four to six customers, which is the point -- at that
size the naive enumeration is beyond doubt.

The other risk is subtler. An exact optimum under a slightly different
objective than the one the metaheuristics minimise would still look like a
clean baseline while quietly making every reported gap wrong, in whichever
direction the objectives disagree. Several tests below exist to pin the
solver's objective to evaluate_solution's.
"""

from __future__ import annotations

import itertools
from collections import Counter

import pytest

from app.core.exact_vrp import (
    EXACT_MAX_CUSTOMERS,
    ExactSolverTooLarge,
    optimality_gap,
    solve_vrp_exact,
)
from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import evaluate_solution, generate_synthetic_vrp


def make(n_customers, seed=1, **kwargs):
    net = generate_synthetic_city_graph(n_nodes=20, seed=seed)
    kwargs.setdefault("vehicle_capacity", 60)
    return generate_synthetic_vrp(
        net, n_customers=n_customers, depot=0, seed=seed, **kwargs
    )


def brute_force_optimum(problem):
    """
    Every routing there is, scored. Deliberately the dumbest implementation
    that could possibly be correct -- it shares no logic with the solver.
    """
    ids = [c.node_id for c in problem.customers]
    best = float("inf")
    for assignment in itertools.product(range(problem.n_vehicles), repeat=len(ids)):
        buckets = [[] for _ in range(problem.n_vehicles)]
        for node, vehicle in zip(ids, assignment):
            buckets[vehicle].append(node)
        orders = [list(itertools.permutations(b)) or [()] for b in buckets]
        for combination in itertools.product(*orders):
            fitness = evaluate_solution(problem, [list(c) for c in combination]).fitness
            best = min(best, fitness)
    return best


# ---------------------------------------------------------------------------
# The claim: this really is the optimum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_customers,seed", [(4, 1), (5, 2), (5, 7), (6, 3)])
def test_it_matches_an_exhaustive_search(n_customers, seed):
    problem = make(n_customers, seed=seed)
    assert solve_vrp_exact(problem).best_fitness == pytest.approx(
        brute_force_optimum(problem), abs=1e-6
    )


def test_it_matches_exhaustive_search_when_capacity_forces_a_penalty():
    """
    The capacity term is added per vehicle after the per-subset search, because
    it depends on which vehicle serves a subset rather than on the order within
    it. If that reasoning were wrong, a binding capacity is where it would show.
    """
    problem = make(5, seed=4, vehicle_capacity=12)
    assert solve_vrp_exact(problem).best_fitness == pytest.approx(
        brute_force_optimum(problem), abs=1e-6
    )


def test_it_matches_exhaustive_search_with_more_vehicles_than_needed():
    problem = make(4, seed=6, vehicle_capacity=60, n_vehicles=4)
    assert solve_vrp_exact(problem).best_fitness == pytest.approx(
        brute_force_optimum(problem), abs=1e-6
    )


def test_it_matches_exhaustive_search_when_idle_vehicles_are_penalised():
    """require_all_vehicles adds a term that depends on how many routes are empty."""
    problem = make(5, seed=5, vehicle_capacity=60, n_vehicles=3)
    problem.require_all_vehicles = True
    assert solve_vrp_exact(problem).best_fitness == pytest.approx(
        brute_force_optimum(problem), abs=1e-6
    )


# ---------------------------------------------------------------------------
# The returned solution has to be the one that was scored
# ---------------------------------------------------------------------------

def test_the_reported_fitness_is_what_the_returned_routes_actually_score():
    """
    The cost is assembled from per-subset pieces, so the routes are rebuilt at
    the end. If reconstruction picked different routes than the DP costed, the
    number and the plan would disagree and only the plan would be real.
    """
    problem = make(6, seed=3)
    result = solve_vrp_exact(problem)
    assert evaluate_solution(problem, result.best_solution.routes).fitness == pytest.approx(
        result.best_fitness, abs=1e-9
    )


def test_every_customer_is_served_exactly_once():
    problem = make(7, seed=2)
    routes = solve_vrp_exact(problem).best_solution.routes
    served = Counter(node for route in routes for node in route)
    assert served == Counter(c.node_id for c in problem.customers)


def test_one_route_comes_back_per_vehicle():
    problem = make(5, seed=8, n_vehicles=4)
    assert len(solve_vrp_exact(problem).best_solution.routes) == problem.n_vehicles


# ---------------------------------------------------------------------------
# No heuristic can beat it -- that is what "exact" means
# ---------------------------------------------------------------------------

def test_no_heuristic_finds_anything_better():
    from app.core.classical_baselines_vrp import run_ga_vrp, run_greedy_nn_vrp

    problem = make(7, seed=5)
    optimum = solve_vrp_exact(problem).best_fitness
    for result in (run_greedy_nn_vrp(problem),
                   run_ga_vrp(problem, pop_size=30, max_iter=60, seed=1)):
        assert result.best_fitness >= optimum - 1e-6, (
            f"{result.algorithm} beat the supposed optimum, so it is not optimal"
        )


# ---------------------------------------------------------------------------
# The size guard
# ---------------------------------------------------------------------------

def test_an_instance_past_the_limit_is_refused_rather_than_run():
    """
    The work grows like e * n factorial. Silently starting a run that will not
    finish is worse than refusing, because it looks like a hang.
    """
    problem = make(6, seed=1)
    with pytest.raises(ExactSolverTooLarge):
        solve_vrp_exact(problem, max_customers=5)


def test_the_refusal_says_what_to_do_about_it():
    problem = make(6, seed=1)
    with pytest.raises(ExactSolverTooLarge, match="max_customers"):
        solve_vrp_exact(problem, max_customers=5)


def test_the_default_limit_is_documented_and_modest():
    assert EXACT_MAX_CUSTOMERS <= 12


# ---------------------------------------------------------------------------
# Gap arithmetic
# ---------------------------------------------------------------------------

def test_the_gap_is_zero_when_a_heuristic_matches_the_optimum():
    assert optimality_gap(100.0, 100.0) == pytest.approx(0.0)


def test_the_gap_is_positive_when_a_heuristic_is_worse():
    assert optimality_gap(110.0, 100.0) == pytest.approx(10.0)


def test_the_gap_does_not_divide_by_zero_on_a_degenerate_optimum():
    assert optimality_gap(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Degenerate instances
# ---------------------------------------------------------------------------

def test_an_instance_with_no_customers_is_handled():
    problem = make(1, seed=1)
    problem.customers = []
    result = solve_vrp_exact(problem)
    assert result.best_solution.routes == [[] for _ in range(problem.n_vehicles)]


def test_a_single_customer_goes_on_one_route():
    problem = make(1, seed=1)
    routes = solve_vrp_exact(problem).best_solution.routes
    assert sum(len(route) for route in routes) == 1
