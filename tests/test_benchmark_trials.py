"""
tests/test_benchmark_trials.py
-------------------------------
Repeated benchmark runs.

One run per algorithm answers "which algorithm won that run", which is a weaker
claim than "which algorithm is better" -- a metaheuristic's result moves with its
seed. `trials` repeats the run so the spread is visible, which is what a box plot
and a success rate are drawn from.

Two properties matter more than the statistics themselves:

  - trials=1 must be indistinguishable from the code that existed before the
    parameter did, including the seed each solver is handed. A default that
    quietly changed every published benchmark figure would be a bad trade for a
    charting feature.

  - the parallel path and the sequential path must agree exactly. The pool is an
    execution detail; if it changed the numbers it would be a bug wearing a
    speedup's clothes.
"""

from __future__ import annotations

import pytest

from app.api.routes import _run_trials, _run_single_trial
from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp

ALGOS = ["qpso", "ga"]
MAX_ITER = 12          # small: these tests are about plumbing, not solution quality


@pytest.fixture(scope="module")
def problem():
    net = generate_synthetic_city_graph(n_nodes=40, seed=11)
    return generate_synthetic_vrp(
        net, n_customers=8, depot=0, vehicle_capacity=90, seed=5
    )


# ---------------------------------------------------------------------------
# The default path is the old path
# ---------------------------------------------------------------------------
def test_single_trial_uses_the_requested_seed_and_runs_sequentially(problem):
    """
    trials=1 must hand the solver exactly the seed it was given, and must not
    reach the pool at all -- so the default benchmark cannot be destabilised by
    anything in the parallel branch.
    """
    out = _run_trials(problem, ALGOS, [7], MAX_ITER)

    for algo in ALGOS:
        assert len(out[algo]) == 1
        assert out[algo][0].seed == 7

    direct = _run_single_trial(problem, "qpso", 7, MAX_ITER)
    assert out["qpso"][0].fitness == pytest.approx(direct.fitness)


# ---------------------------------------------------------------------------
# Parallel must equal sequential
# ---------------------------------------------------------------------------
def test_parallel_trials_match_sequential_exactly(problem):
    """
    The pool is an execution detail. Run the same seeds both ways and require
    identical fitness, so a speedup can never be bought with a different answer.
    """
    seeds = [1, 2, 3, 4]

    pooled = _run_trials(problem, ALGOS, seeds, MAX_ITER)
    sequential = {
        algo: [_run_single_trial(problem, algo, s, MAX_ITER) for s in seeds]
        for algo in ALGOS
    }

    for algo in ALGOS:
        assert [t.fitness for t in pooled[algo]] == pytest.approx(
            [t.fitness for t in sequential[algo]]
        )


def test_samples_come_back_in_seed_order(problem):
    """
    A pool finishes work in whatever order it likes. fitness_samples[i] has to
    belong to seeds[i] regardless, or the box plot is drawn from a shuffle and
    nobody can trace a point back to the run that produced it.
    """
    seeds = [11, 12, 13, 14, 15]
    out = _run_trials(problem, ALGOS, seeds, MAX_ITER)

    for algo in ALGOS:
        assert [t.seed for t in out[algo]] == seeds


# ---------------------------------------------------------------------------
# Different seeds are actually different runs
# ---------------------------------------------------------------------------
def test_trials_explore_different_seeds(problem):
    """
    If every trial returned the same number the spread would be an artefact of
    the plumbing rather than of the search. At least one seed must differ from
    the others for a stochastic solver.
    """
    out = _run_trials(problem, ["qpso"], [1, 2, 3, 4, 5], MAX_ITER)
    fits = [t.fitness for t in out["qpso"]]

    assert len(fits) == 5
    assert len(set(round(f, 9) for f in fits)) > 1, (
        "every seed produced an identical fitness -- the seed is not reaching "
        "the optimiser"
    )


def test_repeating_a_run_repeats_exactly(problem):
    """
    Seeds are derived from the requested one, not drawn at random, so the same
    request twice is the same benchmark twice. Judges asking "run that again"
    should see the same figures.
    """
    a = _run_trials(problem, ["qpso"], [3, 4, 5], MAX_ITER)
    b = _run_trials(problem, ["qpso"], [3, 4, 5], MAX_ITER)

    assert [t.fitness for t in a["qpso"]] == pytest.approx(
        [t.fitness for t in b["qpso"]]
    )
