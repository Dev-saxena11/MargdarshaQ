"""
tests/test_solomon.py
----------------------
Loading Solomon's CVRPTW instances, and scoring runs against them.

Two things here can go wrong quietly, and both would produce a number that
looks like a result:

The instance can be loaded under this project's own assumptions rather than
Solomon's. Solomon is Euclidean, untimed, and scored on distance alone; apply
the congestion curve or the time/distance blend used elsewhere here and the
solver is answering a different question from the one the published figures
answer. The gap would then measure the disagreement between two objectives, not
the quality of the search, and nothing about the output would say so.

The comparison can be made against an infeasible run. Every published figure is
feasible, so a solution that breaks a time window has no gap -- but it does have
a distance, often a flatteringly short one, because skipping the window is what
made it short.

These pin both. Solving is left to the benchmark script: the point here is that
what gets solved, and how it is scored, is right.
"""

from __future__ import annotations

import math
import os

import pytest

from app.core.benchmark_solomon import SolomonRun
from app.core.solomon import (
    BEST_KNOWN,
    INSTANCE_DIR,
    SolomonParseError,
    available_instances,
    best_known,
    load_problem,
    load_solomon_file,
    parse_solomon,
    to_vrp_problem,
)

SAMPLE = """TESTINST

VEHICLE
NUMBER     CAPACITY
  25         200

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME

    0      40         50          0          0       1236          0
    1      45         68         10        912        967         90
    2      45         70         30        825        870         90
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_the_fleet_block_is_read():
    inst = parse_solomon(SAMPLE)
    assert (inst.n_vehicles, inst.capacity) == (25, 200.0)


def test_the_depot_is_row_zero_and_is_not_a_customer():
    inst = parse_solomon(SAMPLE)
    assert (inst.x[0], inst.y[0]) == (40.0, 50.0)
    assert inst.n_customers == 2


def test_customer_fields_land_in_the_right_columns():
    inst = parse_solomon(SAMPLE)
    assert inst.demand[1] == 10.0
    assert inst.ready[1] == 912.0
    assert inst.due[1] == 967.0
    assert inst.service[1] == 90.0


def test_irregular_spacing_is_tolerated():
    """
    The files in circulation differ in spacing and header wording, so rows are
    taken as "seven numbers on a line" rather than by column position.
    """
    squashed = SAMPLE.replace("    0      40         50", "0 40 50")
    assert parse_solomon(squashed).n_customers == 2


def test_a_file_with_no_vehicle_block_is_refused():
    with pytest.raises(SolomonParseError, match="VEHICLE"):
        parse_solomon("JUST A NAME\n\nCUSTOMER\n 0 1 2 3 4 5 6\n")


def test_a_file_with_no_customers_is_refused():
    with pytest.raises(SolomonParseError):
        parse_solomon("NAME\n\nVEHICLE\nNUMBER CAPACITY\n 25 200\n\nCUSTOMER\n")


# ---------------------------------------------------------------------------
# The shipped instances
# ---------------------------------------------------------------------------

def test_the_shipped_instances_are_the_hundred_customer_set():
    names = available_instances()
    assert names, "no Solomon instances shipped"
    for name in names:
        inst = load_solomon_file(os.path.join(INSTANCE_DIR, f"{name}.txt"))
        assert inst.n_customers == 100, f"{name} has {inst.n_customers} customers"


def test_every_shipped_instance_has_a_best_known_value():
    """A gap cannot be reported for an instance with nothing to compare to."""
    for name in available_instances():
        assert best_known(name) is not None, f"no best-known recorded for {name}"


def test_all_three_families_are_represented():
    """C, R and RC differ in how customers are spread; one family is not a benchmark."""
    names = available_instances()
    assert any(n.startswith("C") and not n.startswith("RC") for n in names)
    assert any(n.startswith("R") and not n.startswith("RC") for n in names)
    assert any(n.startswith("RC") for n in names)


# ---------------------------------------------------------------------------
# Solomon's conventions, which the published figures depend on
# ---------------------------------------------------------------------------

def test_travel_time_equals_euclidean_distance():
    """
    Solomon's convention, and the only one under which best-known values mean
    anything. If the loader ever put a speed or congestion factor in the way,
    every gap in the table would silently be against a different problem.
    """
    inst = load_solomon_file(os.path.join(INSTANCE_DIR, "R101.txt"))
    problem = to_vrp_problem(inst)
    for i, j in ((0, 1), (1, 2), (5, 40), (99, 100)):
        euclid = math.hypot(inst.x[i] - inst.x[j], inst.y[i] - inst.y[j])
        assert problem.travel_time(i, j) == pytest.approx(euclid, abs=1e-9)
        assert problem.travel_distance(i, j) == pytest.approx(euclid, abs=1e-9)


def test_the_instance_is_scored_on_distance_alone():
    """Solomon minimises vehicles then distance -- not this project's time blend."""
    problem = load_problem("R101")
    assert problem.objective_w_time == 0.0
    assert problem.objective_w_distance == 1.0


def test_the_congestion_clock_is_off():
    """There is no time of day in a Solomon instance to be congested by."""
    assert load_problem("R101").time_dependent is False


def test_the_declared_fleet_is_a_ceiling_not_the_best_known_count():
    """
    Passing the best-known vehicle count would hand the solver half the answer,
    since minimising vehicles is the first half of Solomon's objective.
    """
    problem = load_problem("C101")
    assert problem.n_vehicles == 25
    assert problem.n_vehicles > BEST_KNOWN["C101"][0]


def test_an_unknown_instance_name_says_what_is_available():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_problem("NOPE999")


# ---------------------------------------------------------------------------
# Gap arithmetic
# ---------------------------------------------------------------------------

def _run(**kw):
    base = dict(instance="R101", algorithm="X", feasible=True, vehicles_used=19,
                distance=1650.8, runtime_sec=1.0,
                best_known_vehicles=19, best_known_distance=1650.8)
    base.update(kw)
    return SolomonRun(**base)


def test_matching_the_best_known_distance_is_a_zero_gap():
    assert _run().distance_gap_pct == pytest.approx(0.0)


def test_a_longer_solution_reports_a_positive_gap():
    assert _run(distance=1815.88).distance_gap_pct == pytest.approx(10.0, abs=1e-6)


def test_an_infeasible_run_has_no_gap():
    """
    Breaking a time window is often what made the distance short. Published
    figures are all feasible, so there is nothing to compare against -- and a
    percentage here would be read as a result.
    """
    assert _run(feasible=False, distance=900.0).distance_gap_pct is None


def test_an_instance_with_no_best_known_value_has_no_gap():
    assert _run(best_known_distance=None).distance_gap_pct is None


def test_extra_vehicles_are_reported_alongside_distance():
    """
    Vehicles come first in Solomon's objective, so a shorter distance bought
    with more vans is worse, not better. Reporting distance alone would hide it.
    """
    assert _run(vehicles_used=24).extra_vehicles == 5
    assert _run(vehicles_used=19).extra_vehicles == 0
