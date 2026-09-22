"""
tests/test_local_search.py
---------------------------
Regression cover for the 2-opt and or-opt operators in app/core/local_search.py.

The defect these exist for: or_opt_pass used to walk a (route_index, position)
snapshot taken before any move was applied. Routes mutate as the pass proceeds,
so once an earlier relocation shifted list positions, a later index pointed at
the wrong customer -- silently duplicating one and dropping another. The
solution still scored, still looked plausible on a map, and still reported a
fitness; it just no longer served every customer.

Nothing in the fitness function catches that, because a dropped customer simply
stops contributing distance and time: losing one makes the solution look
*better*. So the property worth asserting is not "fitness improved" but "the
multiset of customers is exactly preserved".
"""

from __future__ import annotations

from collections import Counter

import pytest

from app.core.graph_model import generate_synthetic_city_graph
from app.core.local_search import (
    local_search_refine,
    or_opt_pass,
    routes_to_chromosome,
    two_opt_pass,
)
from app.core.vrp_problem import (
    decode_chromosome,
    evaluate_solution,
    generate_synthetic_vrp,
)


@pytest.fixture(scope="module")
def problem():
    """A small instance that still has enough customers to move between routes."""
    net = generate_synthetic_city_graph(n_nodes=30, seed=7)
    return generate_synthetic_vrp(
        net, n_customers=12, depot=0, vehicle_capacity=80, seed=3
    )


def served(routes):
    """Every customer across every route, as a multiset, so duplicates show up."""
    return Counter(node for route in routes for node in route)


def expected(problem):
    return Counter(c.node_id for c in problem.customers)


def starting_routes(problem):
    """A deterministic non-trivial assignment: spread customers over the fleet."""
    routes = [[] for _ in range(problem.n_vehicles)]
    for i, customer in enumerate(problem.customers):
        routes[i % problem.n_vehicles].append(customer.node_id)
    return routes


# ---------------------------------------------------------------------------
# The regression this file exists for
# ---------------------------------------------------------------------------

def test_or_opt_serves_every_customer_exactly_once(problem):
    """The stale-index bug: a customer duplicated, another dropped."""
    result = or_opt_pass(problem, starting_routes(problem))
    assert served(result) == expected(problem)


def test_or_opt_conserves_customers_over_repeated_passes(problem):
    """
    One pass can look right by luck. The bug needed an earlier move to shift a
    later index, so it surfaces most reliably after several passes have each
    had the chance to relocate something.
    """
    routes = starting_routes(problem)
    for _ in range(5):
        routes = or_opt_pass(problem, routes)
        assert served(routes) == expected(problem)


def test_two_opt_serves_every_customer_exactly_once(problem):
    result = two_opt_pass(problem, starting_routes(problem))
    assert served(result) == expected(problem)


def test_local_search_refine_conserves_customers(problem):
    """The combined loop, which is what callers actually use."""
    result = local_search_refine(problem, starting_routes(problem), max_passes=3)
    assert served(result) == expected(problem)


def test_or_opt_does_not_mutate_the_routes_it_was_given(problem):
    """Callers keep the pre-refinement solution; the pass must copy, not alias."""
    routes = starting_routes(problem)
    before = [route[:] for route in routes]
    or_opt_pass(problem, routes)
    assert routes == before


# ---------------------------------------------------------------------------
# The operators must also actually be improvements
# ---------------------------------------------------------------------------

def test_local_search_never_returns_a_worse_solution(problem):
    routes = starting_routes(problem)
    before = evaluate_solution(problem, routes).fitness
    after = evaluate_solution(
        problem, local_search_refine(problem, routes, max_passes=3)
    ).fitness
    assert after <= before + 1e-9


def test_or_opt_keeps_the_fleet_size_fixed(problem):
    """
    A relocation empties a route rather than deleting it. Dropping the entry
    would renumber every later vehicle, so a route index would stop referring
    to the vehicle whose capacity it is checked against.
    """
    routes = starting_routes(problem)
    assert len(or_opt_pass(problem, routes)) == len(routes)


def test_or_opt_handles_an_empty_route_in_the_fleet(problem):
    routes = starting_routes(problem)
    routes.append([])
    result = or_opt_pass(problem, routes)
    assert served(result) == expected(problem)


def test_or_opt_handles_every_customer_on_one_route(problem):
    """The degenerate start: one loaded route, the rest empty."""
    routes = [[c.node_id for c in problem.customers]]
    routes += [[] for _ in range(problem.n_vehicles - 1)]
    result = or_opt_pass(problem, routes)
    assert served(result) == expected(problem)


# ---------------------------------------------------------------------------
# Re-encoding back to a chromosome (Lamarckian reinjection)
# ---------------------------------------------------------------------------

def test_routes_to_chromosome_round_trips_through_the_decoder(problem):
    """
    A refined solution is re-encoded and pushed back into the population. If the
    encoding does not decode to the same routes, the optimiser reinjects a
    different solution from the one local search just proved was better.
    """
    refined = local_search_refine(problem, starting_routes(problem), max_passes=2)
    chromosome = routes_to_chromosome(problem, refined)
    assert decode_chromosome(problem, chromosome) == refined
