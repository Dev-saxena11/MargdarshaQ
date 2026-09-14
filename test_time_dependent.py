"""
test_time_dependent.py
----------------------
Tests for time-dependent travel times — routes priced by when a vehicle
actually departs, rather than by a single snapshot of the network.

Background: the network could already vary congestion with a clock, but the
solver never used it. `_precompute_matrices()` ran once with
`current_time=None`, so a van crossing a road at 09:00 and at 14:00 was charged
the same travel time. Rush hour existed on the chart and nowhere in the routing.

Runs offline.

Run with:  python test_time_dependent.py
"""

import sys

from app.core.traffic_profile import TrafficProfile, DEFAULT_PROFILE
from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp, evaluate_solution
from app.core.qpso_vrp import QPSOVRPOptimizer

failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


print("=" * 76)
print("1. The traffic profile has the shape of a working day")
print("=" * 76)

p = DEFAULT_PROFILE
midday = p.multiplier(360)      # 12:00
morning = p.multiplier(150)     # 08:30
evening = p.multiplier(720)     # 18:00
early = p.multiplier(0)         # 06:00

check("morning peak is busier than midday", morning > midday, f"{morning:.2f} vs {midday:.2f}")
check("evening peak is busier than midday", evening > midday, f"{evening:.2f} vs {midday:.2f}")
check("evening is the heavier peak", evening > morning, f"{evening:.2f} vs {morning:.2f}")
check("early morning is near free-flow", early < 1.1, f"{early:.2f}")
check("never faster than free-flow", all(m >= 1.0 for _, m in p.curve()),
      f"min={min(m for _, m in p.curve()):.3f}")
check("curve is sampled across the day", len(p.curve(horizon_min=720, step_min=30)) == 25)
check("assumptions are stated, not hidden", "not measured Delhi data" in p.describe())

print()
print("=" * 76)
print("2. The same road costs more in rush hour")
print("=" * 76)

net = generate_synthetic_city_graph(n_nodes=30, seed=42)
u, v = next(iter(net.road_pairs()))
free = net.travel_time(u, v, current_time=360)
rush = net.travel_time(u, v, current_time=720)
static = net.travel_time(u, v)

check("evening rush costs more than midday", rush > free, f"{rush:.2f} vs {free:.2f}")
check("no clock means static behaviour is unchanged", abs(static - free) < 0.05,
      f"static={static:.3f} midday={free:.3f}")

print()
print("=" * 76)
print("3. The solver prices a leg by when the vehicle departs")
print("=" * 76)

kw = dict(n_customers=12, depot=0, vehicle_capacity=80, seed=1)
static_problem = generate_synthetic_vrp(net, **kw)
td_problem = generate_synthetic_vrp(net, time_dependent=True, **kw)

a, b = td_problem.all_nodes[0], td_problem.all_nodes[1]
# Compare times that are BOTH inside the operating horizon. The default day is
# 480 minutes from 06:00, i.e. 06:00-14:00, so the evening peak falls outside it
# and clamps to the last (quiet) bucket — comparing against it would make this
# test pass on rounding noise rather than on the effect being modelled.
t_quiet = td_problem.travel_time(a, b, depart_at=360)   # 12:00, off-peak
t_rush = td_problem.travel_time(a, b, depart_at=150)    # 08:30, morning peak

check("buckets were precomputed", len(td_problem.time_matrices) > 1,
      f"{len(td_problem.time_matrices)} buckets")
check("leaving in the morning peak costs materially more",
      t_rush > t_quiet * 1.2, f"rush={t_rush:.1f} vs quiet={t_quiet:.1f} min")
check("a departure beyond the horizon clamps rather than wrapping to morning",
      td_problem.travel_time(a, b, depart_at=9999) == td_problem.travel_time(a, b, depart_at=td_problem.horizon_minutes - 1))
check("a static problem ignores the departure time",
      static_problem.travel_time(a, b, depart_at=720) == static_problem.travel_time(a, b))
check("departures past the horizon clamp to the last bucket",
      td_problem.bucket_for(99999) == td_problem._bucket_count() - 1)
check("negative departure clamps to the first bucket", td_problem.bucket_for(-5) == 0)

print()
print("=" * 76)
print("4. Knowing the clock produces a better plan under real traffic")
print("=" * 76)

blind = QPSOVRPOptimizer(static_problem, n_particles=30, max_iter=60, seed=1).optimize().best_solution
aware = QPSOVRPOptimizer(td_problem, n_particles=30, max_iter=60, seed=1).optimize().best_solution

check("time-awareness changes the chosen routes",
      [r for r in blind.routes if r] != [r for r in aware.routes if r])

# Score the clock-blind plan under time-varying traffic — the fair comparison.
blind_under_traffic = evaluate_solution(td_problem, blind.routes)
check("the clock-aware plan is better under time-varying traffic",
      aware.total_time < blind_under_traffic.total_time,
      f"aware={aware.total_time:.1f} vs blind={blind_under_traffic.total_time:.1f} min")

print()
print("=" * 76)
print("5. Existing (static) results are untouched")
print("=" * 76)

# Pinned from before time-dependence existed. Time-dependence is opt-in
# precisely so published benchmark figures are never silently redefined.
sol = QPSOVRPOptimizer(static_problem, n_particles=20, max_iter=40, seed=1).optimize().best_solution
for field_name, expected in (("total_time", 430.0415), ("total_distance", 144.68),
                             ("fitness", 315.8969)):
    got = getattr(sol, field_name)
    check(f"static {field_name} unchanged", abs(got - expected) < 0.01,
          f"expected {expected}, got {got:.4f}")

check("time_dependent is off unless asked for", static_problem.time_dependent is False)

print()
print("=" * 76)
print("6. A custom profile is honoured")
print("=" * 76)

flat = TrafficProfile(morning_peak_strength=0.0, evening_peak_strength=0.0)
check("a flat profile removes rush hour entirely",
      abs(flat.multiplier(720) - 1.0) < 1e-9, f"{flat.multiplier(720):.4f}")
check("a flat profile is still never below free-flow", flat.multiplier(0) >= 1.0)

print()
print("=" * 76)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
