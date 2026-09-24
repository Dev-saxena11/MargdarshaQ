"""
tests/test_formulation_matches_code.py
---------------------------------------
Keeps docs/FORMULATION.md honest.

The formulation document is the deliverable the problem statement asks for, and
a formulation that has drifted from the code is worse than none: it is a précis
that reads as authoritative while describing a different optimiser. That is not
hypothetical here. Before this test existed the document stated the objective as

    F = T_total + 50*P_cap + 10*P_time

which omitted the distance term and its weight, the waiting time folded into
T_total, the per-vehicle cost and speed factors, the idle-vehicle penalty, and
measured lateness from arrival rather than from service start. Every one of
those is a real term in evaluate_solution.

So this file transcribes the equations of FORMULATION.md section 4 directly --
deliberately as a naive, readable walk of the route rather than by reusing any
helper from vrp_problem -- and requires that they reproduce evaluate_solution's
fitness exactly. If someone changes the objective and not the document, or the
document and not the objective, this fails and names the term that moved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import (
    decode_chromosome,
    evaluate_solution,
    generate_synthetic_vrp,
)

# Section 4.4 of the document, and the defaults in evaluate_solution's signature.
LAMBDA_CAP = 50.0
LAMBDA_TIME = 10.0
LAMBDA_IDLE = 200.0
UNREACHABLE_PENALTY = 1000.0


def fitness_from_the_document(problem, routes):
    """
    Section 4 of docs/FORMULATION.md, implemented literally.

    Reads the route the way the document describes a vehicle driving it, so the
    correspondence can be checked line by line against the equations rather than
    trusted.
    """
    customer = {c.node_id: c for c in problem.customers}

    T_total = 0.0          # 4.2
    D_total = 0.0          # 4.2
    P_cap = 0.0            # 4.3, capacity
    P_time = 0.0           # 4.3, time window + unreachable

    for k, route in enumerate(routes):
        if not route:
            continue

        # 4.3 capacity: per vehicle, against that vehicle's own capacity Q_k
        load = sum(customer[n].demand for n in route)
        P_cap += max(0.0, load - problem.capacity_for(k))

        sigma_k = problem.speed_factor_for(k)     # 4.1
        gamma_k = problem.cost_per_km_for(k)      # 4.2

        prev = problem.depot
        tau = 0.0                                  # 4.1, vehicle leaves at tau_{k,0} = 0

        for node in route:
            # 4.1: travel priced at the moment of departure
            theta = problem.travel_time(prev, node, depart_at=tau) * sigma_k

            if not math.isfinite(theta):
                # 4.3: a flat 1000, no time or distance, clock does not advance
                P_time += UNREACHABLE_PENALTY
                prev = node
                continue

            c = customer[node]
            arrival = tau + theta                          # a_{k,j}
            wait = max(0.0, c.ready_time - arrival)        # w_{k,j}
            start = arrival + wait                         # beta_{k,j}

            P_time += max(0.0, start - c.due_time)         # 4.3, from service start
            T_total += theta + wait                        # 4.2, waiting is charged
            D_total += problem.travel_distance(prev, node) * gamma_k

            tau = start + c.service_time                   # tau_{k,j}
            prev = node

        # 4.1: return leg, no wait and no service at the depot
        back = problem.travel_time(prev, problem.depot, depart_at=tau) * sigma_k
        if math.isfinite(back):
            T_total += back
            D_total += problem.travel_distance(prev, problem.depot) * gamma_k

    # 4.3: idle vehicles, only when the fleet must all go out
    P_idle = 0.0
    if getattr(problem, "require_all_vehicles", False):
        P_idle = float(sum(1 for r in routes if not r))

    # 4.4
    w_T = getattr(problem, "objective_w_time", 0.6)
    w_D = getattr(problem, "objective_w_distance", 0.4)
    return (
        w_T * T_total
        + w_D * D_total
        + LAMBDA_CAP * P_cap
        + LAMBDA_TIME * P_time
        + LAMBDA_IDLE * P_idle
    )


def routes_for(problem, seed):
    """A spread of route shapes, including empty vehicles, via the real decoder."""
    rng = np.random.default_rng(seed)
    chromosome = rng.uniform(0, problem.n_vehicles, size=len(problem.customers))
    return decode_chromosome(problem, chromosome)


def base_problem(**kwargs):
    net = generate_synthetic_city_graph(n_nodes=30, seed=11)
    params = dict(net=net, n_customers=10, depot=0, vehicle_capacity=80, seed=5)
    params.update(kwargs)
    return generate_synthetic_vrp(**params)


# ---------------------------------------------------------------------------
# The default instance: bi-objective, uniform fleet, static travel times
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_documented_fitness_matches_evaluate_solution(seed):
    problem = base_problem()
    routes = routes_for(problem, seed)

    assert evaluate_solution(problem, routes).fitness == pytest.approx(
        fitness_from_the_document(problem, routes)
    )


# ---------------------------------------------------------------------------
# The terms the old document left out. Each of these would pass against a
# time-only objective and fail against the real one, or vice versa.
# ---------------------------------------------------------------------------
def test_objective_is_bi_objective_not_time_alone():
    """w_D is a real term: zeroing it must change the answer."""
    problem = base_problem()
    routes = routes_for(problem, 3)

    blended = evaluate_solution(problem, routes).fitness
    time_only = evaluate_solution(problem, routes, w_time=1.0, w_distance=0.0).fitness

    assert blended != pytest.approx(time_only)
    assert problem.objective_w_time == pytest.approx(0.6)
    assert problem.objective_w_distance == pytest.approx(0.4)


@pytest.mark.parametrize("seed", range(4))
def test_mixed_fleet_speed_and_cost_factors(seed):
    """sigma_k on travel time and gamma_k on distance, as section 4.1/4.2 state."""
    problem = base_problem()
    problem.vehicle_speed_factors = [0.7, 1.0, 1.4][: problem.n_vehicles]
    problem.vehicle_cost_per_km = [1.6, 1.0, 0.5][: problem.n_vehicles]
    routes = routes_for(problem, seed)

    assert evaluate_solution(problem, routes).fitness == pytest.approx(
        fitness_from_the_document(problem, routes)
    )


@pytest.mark.parametrize("seed", range(4))
def test_idle_vehicle_penalty(seed):
    """The 200-per-parked-van term only exists when the fleet must all go out."""
    problem = base_problem(require_all_vehicles=True)
    routes = routes_for(problem, seed)

    # Force at least one van to be idle so the term is actually exercised.
    routes = routes + [[]]
    problem.n_vehicles = len(routes)

    assert evaluate_solution(problem, routes).fitness == pytest.approx(
        fitness_from_the_document(problem, routes)
    )


@pytest.mark.parametrize("seed", range(4))
def test_time_dependent_travel_times(seed):
    """t(i, j, tau) is read at the departure clock, not as a static matrix."""
    problem = base_problem(time_dependent=True, bucket_minutes=30.0)
    routes = routes_for(problem, seed)

    assert evaluate_solution(problem, routes).fitness == pytest.approx(
        fitness_from_the_document(problem, routes)
    )


def test_waiting_time_is_charged_to_total_time():
    """
    Section 4.2 says T_total includes waiting. A vehicle sent to a customer whose
    window opens long after it can arrive must be charged for the idling.
    """
    problem = base_problem()
    target = problem.customers[0]
    target.ready_time = 400.0
    target.due_time = 600.0
    problem.recompute_matrices()

    solution = evaluate_solution(problem, [[target.node_id]])
    pure_travel = (
        problem.travel_time(problem.depot, target.node_id)
        + problem.travel_time(target.node_id, problem.depot)
    )

    assert solution.total_time > pure_travel
    assert solution.total_time == pytest.approx(
        pure_travel + (target.ready_time - problem.travel_time(problem.depot, target.node_id))
    )


@pytest.mark.parametrize("seed", range(4))
def test_conformance_when_waiting_causes_the_lateness(seed):
    """
    The random instances above never happen to produce a customer whose window
    opens late and shuts before service can start, so they cannot tell lateness
    measured from arrival apart from lateness measured from service start --
    checked by mutation: swapping one for the other leaves all of them passing.
    This builds that case on purpose, so the distinction is actually covered.
    """
    problem = base_problem()
    routes = routes_for(problem, seed)

    served = [n for route in routes for n in route]
    lookup = {c.node_id: c for c in problem.customers}
    for offset, node in enumerate(served[:4]):
        c = lookup[node]
        # Opens well after any plausible arrival, and shuts before it opens plus
        # the wait -- so the vehicle waits, and the wait is what makes it late.
        c.ready_time = 300.0 + 20.0 * offset
        c.due_time = c.ready_time - 15.0
    problem.recompute_matrices()

    assert evaluate_solution(problem, routes).fitness == pytest.approx(
        fitness_from_the_document(problem, routes)
    )


def test_lateness_is_measured_from_service_start_not_arrival():
    """
    Section 4.3. With a window that opens late and closes immediately after, a
    vehicle arriving early is still late -- the wait pushes service start past
    the deadline. Measuring from arrival, as the document used to, would report
    this solution as on time.
    """
    problem = base_problem()
    target = problem.customers[0]
    arrival = problem.travel_time(problem.depot, target.node_id)
    target.ready_time = arrival + 100.0
    target.due_time = arrival + 50.0      # closes before the vehicle may be served
    problem.recompute_matrices()

    solution = evaluate_solution(problem, [[target.node_id]])

    assert solution.time_window_violation > 0
    assert solution.time_window_violation == pytest.approx(
        target.ready_time - target.due_time
    )
    assert not solution.feasible
