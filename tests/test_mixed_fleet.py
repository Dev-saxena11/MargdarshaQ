"""
tests/test_mixed_fleet.py
--------------------------
A fleet whose vehicles are not interchangeable.

An Indian municipal round is rarely one kind of vehicle. A two-wheeler threads
traffic a tempo cannot and carries almost nothing; a truck is the reverse.
Capacity alone does not express that, so vehicles also carry a speed factor
(multiplying travel time) and a running cost (weighting distance).

The subtle part is what this does to exact_vrp. Its first stage finds the
cheapest ORDER for each subset of customers, and it used to find one per
subset, full stop. With a mixed fleet that is wrong: a faster van reaches the
same stops earlier, waits differently against the same time windows, and the
penalty does not simply scale -- so the cheapest order genuinely differs
between vehicles. Reusing one table across a mixed fleet returns a number that
is not the optimum, and nothing about it looks wrong.

So the central test here is the same one used for the uniform case: enumerate
every assignment and every order by brute force, and require the solver to
match. Verified that it bites -- with one shared table the mixed cases come
back 12 to 21 percent above the true optimum while the uniform control still
passes.
"""

from __future__ import annotations

import itertools

import pytest

from app.core.exact_vrp import solve_vrp_exact
from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import evaluate_solution, generate_synthetic_vrp


def make(n_customers=5, n_vehicles=3, seed=5, speeds=None, costs=None):
    net = generate_synthetic_city_graph(n_nodes=20, seed=seed)
    problem = generate_synthetic_vrp(
        net, n_customers=n_customers, depot=0, vehicle_capacity=60,
        n_vehicles=n_vehicles, seed=seed,
    )
    problem.vehicle_speed_factors = speeds
    problem.vehicle_cost_per_km = costs
    return problem


def brute_force_optimum(problem):
    ids = [c.node_id for c in problem.customers]
    best = float("inf")
    for assignment in itertools.product(range(problem.n_vehicles), repeat=len(ids)):
        buckets = [[] for _ in range(problem.n_vehicles)]
        for node, vehicle in zip(ids, assignment):
            buckets[vehicle].append(node)
        orders = [list(itertools.permutations(b)) or [()] for b in buckets]
        for combination in itertools.product(*orders):
            best = min(best, evaluate_solution(problem, [list(c) for c in combination]).fitness)
    return best


# ---------------------------------------------------------------------------
# The accessors, and the uniform default
# ---------------------------------------------------------------------------

def test_a_uniform_fleet_reports_neutral_factors():
    problem = make()
    assert problem.speed_factor_for(0) == 1.0
    assert problem.cost_per_km_for(0) == 1.0
    assert problem.has_mixed_fleet() is False


def test_per_vehicle_values_are_read_by_index():
    problem = make(speeds=[0.6, 1.0, 1.4], costs=[2.0, 1.0, 0.5])
    assert [problem.speed_factor_for(v) for v in range(3)] == [0.6, 1.0, 1.4]
    assert [problem.cost_per_km_for(v) for v in range(3)] == [2.0, 1.0, 0.5]
    assert problem.has_mixed_fleet() is True


def test_a_vehicle_past_the_end_of_the_list_falls_back():
    """
    The fleet is resized mid-replan (see dynamic_vrp), and a van with no entry
    should cost the ordinary amount rather than stop the solve.
    """
    problem = make(speeds=[0.6])
    assert problem.speed_factor_for(99) == 1.0


def test_a_list_of_identical_values_is_not_a_mixed_fleet():
    assert make(speeds=[1.0, 1.0, 1.0]).has_mixed_fleet() is False


# ---------------------------------------------------------------------------
# The factors reach the cost
# ---------------------------------------------------------------------------

def test_a_faster_vehicle_records_less_travel_time():
    problem = make()
    routes = [[c.node_id for c in problem.customers], [], []]
    baseline = evaluate_solution(problem, routes).total_time

    problem.vehicle_speed_factors = [0.5, 1.0, 1.0]
    assert evaluate_solution(problem, routes).total_time < baseline


def test_a_costlier_vehicle_records_more_distance():
    problem = make()
    routes = [[c.node_id for c in problem.customers], [], []]
    baseline = evaluate_solution(problem, routes).total_distance

    problem.vehicle_cost_per_km = [2.0, 1.0, 1.0]
    assert evaluate_solution(problem, routes).total_distance > baseline


def test_the_factors_apply_to_the_vehicle_that_drives_not_the_whole_fleet():
    """Loading the slow van should cost more than loading the fast one."""
    problem = make(speeds=[2.0, 0.5, 1.0])
    stops = [c.node_id for c in problem.customers]
    on_slow = evaluate_solution(problem, [stops, [], []]).total_time
    on_fast = evaluate_solution(problem, [[], stops, []]).total_time
    assert on_slow > on_fast


def test_a_uniform_fleet_scores_exactly_as_before():
    """Setting no factors must not move a single number."""
    problem = make()
    routes = [[c.node_id for c in problem.customers], [], []]
    before = evaluate_solution(problem, routes).fitness
    problem.vehicle_speed_factors = [1.0, 1.0, 1.0]
    problem.vehicle_cost_per_km = [1.0, 1.0, 1.0]
    assert evaluate_solution(problem, routes).fitness == pytest.approx(before, abs=1e-12)


# ---------------------------------------------------------------------------
# The exact solver stays exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,speeds,costs", [
    ("uniform", None, None),
    ("mixed speed", [0.6, 1.0, 1.4], None),
    ("mixed cost", None, [0.5, 1.0, 2.0]),
    ("mixed speed and cost", [0.6, 1.0, 1.4], [2.0, 1.0, 0.5]),
])
def test_the_exact_solver_matches_brute_force_on_a_mixed_fleet(label, speeds, costs):
    """
    The one that matters. A faster van waits differently against the same time
    windows, so the cheapest order for a set of stops is not the same for every
    vehicle -- and a solver that assumes it is returns a number that is not the
    optimum while still calling it one.
    """
    problem = make(speeds=speeds, costs=costs)
    assert solve_vrp_exact(problem).best_fitness == pytest.approx(
        brute_force_optimum(problem), abs=1e-6
    )


def test_the_exact_solution_scores_what_it_claims_on_a_mixed_fleet():
    """Routes are rebuilt per vehicle class, so reconstruction can disagree."""
    problem = make(speeds=[0.6, 1.0, 1.4], costs=[2.0, 1.0, 0.5])
    result = solve_vrp_exact(problem)
    assert evaluate_solution(problem, result.best_solution.routes).fitness == pytest.approx(
        result.best_fitness, abs=1e-9
    )


def test_a_mixed_fleet_prices_one_route_table_per_class():
    """
    Three distinct classes means three stage-one tables, not one. Reported via
    n_evaluations, which counts the subsets actually priced.
    """
    uniform = solve_vrp_exact(make()).n_evaluations
    mixed = solve_vrp_exact(make(speeds=[0.6, 1.0, 1.4])).n_evaluations
    assert mixed == uniform * 3
