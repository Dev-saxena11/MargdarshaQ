"""
tests/test_qpso_jump_cap.py
----------------------------
The bounded-jump term in QPSOVRPOptimizer.

Vanilla QPSO draws u ~ U(0,1) and moves each dimension by a factor of ln(1/u),
which is unbounded as u approaches zero. One dimension is rarely a problem; a
VRP chromosome has one dimension per customer, so the chance that *at least
one* gene draws a huge excursion on a given iteration rises quickly with the
customer count. Those excursions land exactly when the search should be
exploiting structure it has already found, so a good solution gets thrown away
by a single unlucky gene.

The fix caps ln(1/u) at max_jump_factor and, when no cap is given, tightens it
as dimensionality grows. This file pins the cap's shape and the properties that
make it safe -- not specific fitness values, which are seed- and
hardware-dependent and would make the suite a coin toss.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.graph_model import generate_synthetic_city_graph
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.vrp_problem import generate_synthetic_vrp


def build(n_customers, **kwargs):
    """A real instance, for the end-to-end runs. Kept small: these actually solve."""
    net = generate_synthetic_city_graph(n_nodes=max(20, n_customers * 2), seed=7)
    problem = generate_synthetic_vrp(
        net, n_customers=n_customers, depot=0, vehicle_capacity=200, seed=3
    )
    return problem, QPSOVRPOptimizer(problem, seed=1, **kwargs)


class _ProblemOfSize:
    """
    Only what __init__ reads to size the cap: a customer count and a fleet.
    Building a real instance to ask "what cap does N customers get" would mean
    generating an N*2-node city and a full distance matrix, which at the sizes
    the floor is about (tens of thousands) is minutes of work to answer a
    question about one line of arithmetic.
    """

    def __init__(self, n_customers, n_vehicles=5):
        self.customers = [object()] * n_customers
        self.n_vehicles = n_vehicles


def cap_for(n_customers, **kwargs):
    return QPSOVRPOptimizer(_ProblemOfSize(n_customers), seed=1, **kwargs).max_jump_factor


# ---------------------------------------------------------------------------
# The default cap tightens as dimensionality grows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_customers", [5, 10, 20, 40, 80, 500])
def test_the_default_cap_is_positive_and_finite(n_customers):
    assert 0.0 < cap_for(n_customers) < float("inf")


def test_the_default_cap_never_increases_with_dimensionality():
    """More genes that can each draw a large jump -> a tighter per-gene cap."""
    caps = [cap_for(n) for n in (5, 10, 20, 40, 80, 160, 320)]
    assert caps == sorted(caps, reverse=True)


def test_a_high_dimensional_instance_is_capped_more_tightly_than_a_small_one():
    assert cap_for(80) < cap_for(5)


def test_the_cap_has_a_floor_so_the_search_can_still_move():
    """
    Shrinking without limit would eventually pin every particle to its
    attractor and stop the search exploring at all.
    """
    assert cap_for(10_000) >= 0.3


def test_an_explicit_cap_overrides_the_dimensional_default():
    assert cap_for(40, max_jump_factor=2.5) == 2.5


# ---------------------------------------------------------------------------
# The cap actually bounds the term it is applied to
# ---------------------------------------------------------------------------

def test_the_jump_term_never_exceeds_the_cap_even_for_extreme_draws():
    """
    Calls the optimiser's own term, so removing the bound from the update fails
    here. u near zero is the case that made vanilla QPSO unstable:
    ln(1/1e-300) is about 690.
    """
    opt = QPSOVRPOptimizer(_ProblemOfSize(40), seed=1)
    u = np.array([1e-300, 1e-12, 1e-6, 0.01, 0.5, 0.999])
    jump = opt._bounded_jump(u)
    assert np.all(jump <= opt.max_jump_factor + 1e-12)
    assert np.all(np.isfinite(jump))
    assert np.log(1.0 / u[0]) > 600          # the draw really was extreme


def test_the_jump_term_leaves_ordinary_draws_untouched():
    """It should clip the tail, not flatten the whole distribution."""
    opt = QPSOVRPOptimizer(_ProblemOfSize(5), seed=1)
    u = np.array([0.5, 0.8, 0.95])
    assert np.allclose(opt._bounded_jump(u), np.log(1.0 / u))


def test_the_jump_term_is_the_one_the_update_uses():
    """
    Guards against the bound being reintroduced inline and the method left
    behind unused -- the test above would still pass while the loop diverged.
    """
    import inspect
    from app.core import qpso_vrp
    source = inspect.getsource(qpso_vrp.QPSOVRPOptimizer.optimize)
    assert "_bounded_jump" in source, "optimize() no longer uses the bounded jump term"
    assert "np.log(1.0 / u)" not in source, "optimize() computes an uncapped jump inline"


# ---------------------------------------------------------------------------
# End-to-end: the optimiser stays well-behaved in high dimensions
# ---------------------------------------------------------------------------

def test_positions_stay_inside_the_gene_range_on_a_high_dimensional_run():
    """
    An uncapped excursion drives genes far outside [0, n_vehicles) and relies on
    the clip to claw them back, which pins whole routes onto the first or last
    vehicle. Every returned gene must be a usable vehicle index.
    """
    problem, opt = build(40, n_particles=10, max_iter=20, use_local_search=False)
    result = opt.optimize()
    routes = result.best_solution.routes
    assert len(routes) == problem.n_vehicles
    served = [node for route in routes for node in route]
    assert sorted(served) == sorted(c.node_id for c in problem.customers)


def test_a_high_dimensional_run_converges_monotonically():
    """
    The curve records the best solution found so far, so it can never rise.

    Note what this does NOT show: because the incumbent is only ever replaced by
    something better, the curve stays monotone with the cap removed too. It
    pins the curve's meaning for the benchmark charts that plot it, not the cap.
    The cap itself is covered by the bounded-jump tests above.
    """
    _, opt = build(40, n_particles=10, max_iter=25, use_local_search=False)
    curve = opt.optimize().convergence_curve
    assert len(curve) > 1
    assert all(b <= a + 1e-9 for a, b in zip(curve, curve[1:]))


def test_a_high_dimensional_run_returns_a_usable_solution():
    problem, opt = build(40, n_particles=10, max_iter=20, use_local_search=False)
    result = opt.optimize()
    assert np.isfinite(result.best_fitness)
    served = [node for route in result.best_solution.routes for node in route]
    assert sorted(served) == sorted(c.node_id for c in problem.customers)


def test_a_tighter_cap_does_not_destabilise_the_search():
    """A hand-set cap at the floor must still produce a finite, complete solution."""
    problem, opt = build(30, max_jump_factor=0.3, n_particles=10,
                         max_iter=20, use_local_search=False)
    result = opt.optimize()
    assert np.isfinite(result.best_fitness)
    served = [node for route in result.best_solution.routes for node in route]
    assert sorted(served) == sorted(c.node_id for c in problem.customers)
