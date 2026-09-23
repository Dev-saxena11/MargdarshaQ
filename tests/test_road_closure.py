"""
tests/test_road_closure.py
---------------------------
Closing a road, and re-planning around it.

A closure is deliberately not a large congestion factor. A road that is merely
slow is still a road: given a bad enough detour the optimiser will drive down
it anyway, which is the right answer for a jam and the wrong one for a street
barricaded for a festival. So a closure removes the edge and the routing has to
find another way.

The failure worth guarding against is the one that still returns a number.
Closing a road can cut a stop off entirely rather than merely make it
expensive, and the scoring treats an unreachable leg as a large penalty rather
than an error -- so a plan that strands a customer still scores, still draws on
a map, and still looks like a plan. It is not one, and the only way to know is
to ask whether the stop is still reachable.
"""

from __future__ import annotations

import pytest

from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import evaluate_solution, generate_synthetic_vrp


@pytest.fixture
def net():
    return generate_synthetic_city_graph(n_nodes=30, seed=3)


def a_road(net):
    return next(net.road_pairs())


# ---------------------------------------------------------------------------
# Closing and reopening
# ---------------------------------------------------------------------------

def test_closing_a_road_removes_both_directions(net):
    u, v = a_road(net)
    before = net.graph.number_of_edges()
    assert net.close_road(u, v) == 2
    assert net.graph.number_of_edges() == before - 2
    assert not net.graph.has_edge(u, v)
    assert not net.graph.has_edge(v, u)


def test_a_one_way_street_loses_only_the_direction_it_has(net):
    net.add_edge(900, 901, distance=1.0, bidirectional=False)
    assert net.close_road(900, 901) == 1


def test_closing_one_direction_leaves_the_other_open(net):
    u, v = a_road(net)
    assert net.close_road(u, v, both_directions=False) == 1
    assert net.graph.has_edge(v, u)


def test_reopening_restores_the_road_exactly(net):
    u, v = a_road(net)
    original = dict(net.graph[u][v])
    net.close_road(u, v)
    assert net.reopen_road(u, v) == 2
    assert net.graph[u][v] == original


def test_closing_an_already_closed_road_is_a_no_op(net):
    """
    A repeated click must not stack duplicate saved copies, or the road becomes
    unreopenable and the saved state is a lie.
    """
    u, v = a_road(net)
    net.close_road(u, v)
    assert net.close_road(u, v) == 0
    assert net.closed_roads() == [(u, v)]


def test_reopening_a_road_that_is_not_closed_changes_nothing(net):
    u, v = a_road(net)
    assert net.reopen_road(u, v) == 0


def test_closed_roads_are_listed_once_not_once_per_direction(net):
    u, v = a_road(net)
    net.close_road(u, v)
    assert net.closed_roads() == [(u, v)]


def test_reopen_all_lifts_every_closure(net):
    roads = [r for _, r in zip(range(3), net.road_pairs())]
    for u, v in roads:
        net.close_road(u, v)
    before = net.graph.number_of_edges()
    assert net.reopen_all_roads() == 6
    assert net.graph.number_of_edges() == before + 6
    assert net.closed_roads() == []


# ---------------------------------------------------------------------------
# What it does to routing
# ---------------------------------------------------------------------------

def test_a_closure_changes_the_plan(net):
    """
    The point of the feature. Matrices are rebuilt against the graph as it now
    stands, so the fleet does not keep driving down a road that is gone.
    """
    problem = generate_synthetic_vrp(net, n_customers=8, depot=0,
                                     vehicle_capacity=100, seed=3)
    routes = [[c.node_id for c in problem.customers]] + \
             [[] for _ in range(problem.n_vehicles - 1)]
    before = evaluate_solution(problem, routes).total_distance

    # Close a road the plan actually uses, so the detour has to show up.
    path = problem.path_matrix.get((problem.depot, problem.customers[0].node_id), [])
    assert len(path) >= 2, "no multi-hop path to close a road on"
    net.close_road(path[0], path[1])
    problem.recompute_matrices()

    assert evaluate_solution(problem, routes).total_distance != pytest.approx(before)


def test_reopening_restores_the_original_plan_cost(net):
    problem = generate_synthetic_vrp(net, n_customers=8, depot=0,
                                     vehicle_capacity=100, seed=3)
    routes = [[c.node_id for c in problem.customers]] + \
             [[] for _ in range(problem.n_vehicles - 1)]
    before = evaluate_solution(problem, routes).total_distance

    path = problem.path_matrix[(problem.depot, problem.customers[0].node_id)]
    net.close_road(path[0], path[1])
    problem.recompute_matrices()
    net.reopen_road(path[0], path[1])
    problem.recompute_matrices()

    assert evaluate_solution(problem, routes).total_distance == pytest.approx(before)


# ---------------------------------------------------------------------------
# Stranding -- the failure that still returns a number
# ---------------------------------------------------------------------------

def test_a_reachable_network_strands_nobody(net):
    assert net.unreachable_from(0, list(net.graph.nodes())) == []


def test_cutting_every_road_into_a_stop_strands_it(net):
    """
    The scoring would price this as a large penalty and return a plan anyway.
    Nothing about the number says a customer cannot be served at all.
    """
    victim = next(n for n in net.graph.nodes() if n != 0)
    for neighbour in list(net.graph.predecessors(victim)):
        net.close_road(neighbour, victim, both_directions=True)
    assert victim in net.unreachable_from(0, list(net.graph.nodes()))


def test_a_stranded_stop_still_scores_which_is_why_it_must_be_checked(net):
    """
    Demonstrates the hazard directly: an unreachable customer produces a
    finite fitness, so a caller that does not ask about reachability gets a
    plausible-looking plan for a round that cannot be driven.
    """
    problem = generate_synthetic_vrp(net, n_customers=5, depot=0,
                                     vehicle_capacity=100, seed=3)
    victim = problem.customers[0].node_id
    for neighbour in list(net.graph.predecessors(victim)):
        net.close_road(neighbour, victim, both_directions=True)
    problem.recompute_matrices()

    routes = [[c.node_id for c in problem.customers]] + \
             [[] for _ in range(problem.n_vehicles - 1)]
    solution = evaluate_solution(problem, routes)

    import math
    assert math.isfinite(solution.fitness)          # it scores...
    assert net.unreachable_from(problem.depot, [victim]) == [victim]   # ...but cannot be driven
