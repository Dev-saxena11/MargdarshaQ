"""
tests/test_chromosome_encoding.py
----------------------------------
The random-key encoding shared by QPSO, GA, SA and standard PSO.

A chromosome holds one float per customer. Its integer part names the vehicle,
its fractional part is a priority that orders that vehicle's stops. Every
metaheuristic here searches in that continuous space and only becomes a routing
decision through decode_chromosome, so a decoder that loses or reorders a
customer corrupts all four algorithms at once -- and does it quietly, because
the result is still a well-formed routes structure that still scores.

These pin the decoder's contract rather than any particular route it produces.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import (
    decode_chromosome,
    evaluate_chromosome,
    generate_synthetic_vrp,
)


@pytest.fixture(scope="module")
def problem():
    net = generate_synthetic_city_graph(n_nodes=30, seed=11)
    return generate_synthetic_vrp(
        net, n_customers=10, depot=0, vehicle_capacity=80, seed=5
    )


def served(routes):
    return Counter(node for route in routes for node in route)


def expected(problem):
    return Counter(c.node_id for c in problem.customers)


# ---------------------------------------------------------------------------
# Conservation: every customer, exactly once, whatever the genes say
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_a_random_chromosome_serves_every_customer_exactly_once(problem, seed):
    rng = np.random.default_rng(seed)
    chromosome = rng.uniform(0.0, problem.n_vehicles, size=len(problem.customers))
    assert served(decode_chromosome(problem, chromosome)) == expected(problem)


def test_the_decoder_always_returns_one_route_per_vehicle(problem):
    """
    Route index is vehicle index -- capacity_for(v) is looked up by position.
    An empty vehicle must stay an empty list rather than vanish, or every later
    route silently shifts onto a different vehicle's capacity.
    """
    chromosome = np.zeros(len(problem.customers))       # everyone on vehicle 0
    routes = decode_chromosome(problem, chromosome)
    assert len(routes) == problem.n_vehicles
    assert len(routes[0]) == len(problem.customers)
    assert all(route == [] for route in routes[1:])


def test_decoding_is_deterministic(problem):
    rng = np.random.default_rng(99)
    chromosome = rng.uniform(0.0, problem.n_vehicles, size=len(problem.customers))
    assert decode_chromosome(problem, chromosome) == decode_chromosome(problem, chromosome)


# ---------------------------------------------------------------------------
# The two halves of a gene: integer part picks the vehicle, fraction orders
# ---------------------------------------------------------------------------

def test_the_integer_part_selects_the_vehicle(problem):
    """Customer i goes to vehicle i % n_vehicles, by construction."""
    n_v = problem.n_vehicles
    chromosome = np.array([float(i % n_v) for i in range(len(problem.customers))])
    routes = decode_chromosome(problem, chromosome)
    for i, customer in enumerate(problem.customers):
        assert customer.node_id in routes[i % n_v]


def test_the_fractional_part_orders_the_route_ascending(problem):
    """All on one vehicle, priorities descending -> the route comes back reversed."""
    n = len(problem.customers)
    chromosome = np.array([0.9 - 0.05 * i for i in range(n)])
    route = decode_chromosome(problem, chromosome)[0]
    assert route == [c.node_id for c in reversed(problem.customers)]


def test_ordering_depends_only_on_the_fraction_not_the_vehicle_number(problem):
    """
    Adding a whole number moves a customer to another vehicle without changing
    where it sits in that vehicle's sequence.
    """
    n = len(problem.customers)
    fracs = np.linspace(0.05, 0.95, n)
    first = decode_chromosome(problem, fracs.copy())[0]
    if problem.n_vehicles > 1:
        second = decode_chromosome(problem, fracs + 1.0)[1]
        assert first == second


# ---------------------------------------------------------------------------
# Genes at and beyond the edges of the range
# ---------------------------------------------------------------------------

def test_a_gene_at_the_upper_bound_lands_on_the_last_vehicle(problem):
    """
    floor(n_vehicles) is one past the last index. It is clipped rather than
    allowed to raise, because the optimisers clip positions to
    upper - 1e-9 and floating point lets a value reach the bound exactly.
    """
    chromosome = np.full(len(problem.customers), float(problem.n_vehicles))
    routes = decode_chromosome(problem, chromosome)
    assert served(routes) == expected(problem)
    assert len(routes[-1]) == len(problem.customers)


def test_genes_outside_the_range_are_clipped_rather_than_dropped(problem):
    """A gene that escaped its bounds must not cost us a customer."""
    n = len(problem.customers)
    chromosome = np.array([-5.0 if i % 2 else problem.n_vehicles + 5.0 for i in range(n)])
    assert served(decode_chromosome(problem, chromosome)) == expected(problem)


def test_equal_priorities_keep_every_customer(problem):
    """Ties must not collapse two customers into one."""
    chromosome = np.full(len(problem.customers), 0.5)
    assert served(decode_chromosome(problem, chromosome)) == expected(problem)


# ---------------------------------------------------------------------------
# What the optimisers actually call
# ---------------------------------------------------------------------------

def test_evaluate_chromosome_scores_the_routes_the_decoder_produces(problem):
    rng = np.random.default_rng(3)
    chromosome = rng.uniform(0.0, problem.n_vehicles, size=len(problem.customers))
    solution = evaluate_chromosome(problem, chromosome)
    assert solution.routes == decode_chromosome(problem, chromosome)
    assert np.isfinite(solution.fitness)
