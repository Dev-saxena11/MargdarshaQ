"""
test_dynamic_correctness.py
---------------------------
Regression tests for the dynamic-traffic re-optimization engine (issue #2).

Covers three defects that the original implementation shipped with:
  1. Re-planning ignored already-consumed vehicle load, producing routes that
     beat the static plan only by exceeding vehicle capacity.
  2. The simulation mutated the caller's VRPProblem (and, via the in-memory
     store, every later solve/benchmark on that instance).
  3. clear_incidents() cleared the incident list but left edge congestion
     factors permanently rewritten, so there was no way to undo an incident.

Run with:  python test_dynamic_correctness.py
"""

import sys

from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp
from app.core.dynamic_vrp import simulate_dynamic_reroute
from app.core.qpso_vrp import QPSOVRPOptimizer

failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


def _instance(n_nodes=30, n_customers=12, capacity=80, seed=42):
    net = generate_synthetic_city_graph(n_nodes=n_nodes, seed=seed)
    vrp = generate_synthetic_vrp(
        net, n_customers=n_customers, depot=0, vehicle_capacity=capacity, seed=1
    )
    return net, vrp


print("=" * 70)
print("1. Re-planning must respect each vehicle's REMAINING capacity")
print("=" * 70)
for cap in (80, 50, 40):
    _, vrp = _instance(capacity=cap)
    res = simulate_dynamic_reroute(
        problem=vrp, incident_factor=4.0, trigger_time_min=90.0,
        algorithm="qpso", n_particles=20, max_iter=40, seed=1,
    )
    viol = res.dynamic_solution.capacity_violation
    # The penalty-based solver may leave a small residual; the pre-fix bug
    # produced violations of ~20 units against the full-capacity assumption.
    check(
        f"capacity={cap}: dynamic plan does not overload vehicles",
        viol < 1.0,
        f"violation={viol:.2f}",
    )

print()
print("=" * 70)
print("2. Simulation must not mutate the caller's problem")
print("=" * 70)
net, vrp = _instance()


def _solve():
    return QPSOVRPOptimizer(vrp, n_particles=20, max_iter=40, seed=1).optimize().best_solution.total_time


before = _solve()
res = simulate_dynamic_reroute(
    problem=vrp, incident_factor=4.0, trigger_time_min=60.0,
    algorithm="qpso", n_particles=20, max_iter=40, seed=1,
)
after = _solve()

check("solve result unchanged after a dynamic run", abs(before - after) < 1e-6,
      f"{before:.2f} -> {after:.2f} min")
check("no incident leaked onto the caller's network", len(net.incidents) == 0,
      f"{len(net.incidents)} incident(s)")
check("incident was applied to the returned simulated copy",
      res.simulated_problem is not None and len(res.simulated_problem.net.incidents) == 1)

print()
print("=" * 70)
print("3. clear_incidents() must restore original congestion factors")
print("=" * 70)
net2 = generate_synthetic_city_graph(n_nodes=20, seed=42)
u, v = list(net2.graph.edges())[0]
original = net2.graph[u][v]["congestion_factor"]
net2.apply_incident(u, v, 4.0)
net2.apply_incident(u, v, 6.0)   # overlapping incident on the same edge
net2.clear_incidents()
restored = net2.graph[u][v]["congestion_factor"]

check("edge factor restored after clear_incidents()", abs(restored - original) < 1e-9,
      f"{original:.4f} -> {restored:.4f}")
check("incident list emptied", len(net2.incidents) == 0)

print()
print("=" * 70)
print("4. Savings are reported signed (a worse re-plan must be visible)")
print("=" * 70)
# This configuration is one where re-planning genuinely loses; the pre-fix code
# clamped it to 0.0 and so could never display a regression.
net3 = generate_synthetic_city_graph(n_nodes=25, seed=2)
vrp3 = generate_synthetic_vrp(net3, n_customers=10, depot=0, vehicle_capacity=60, seed=2)
res3 = simulate_dynamic_reroute(
    problem=vrp3, incident_factor=2.0, trigger_time_min=120.0,
    algorithm="qpso", n_particles=15, max_iter=25, seed=2,
)
check("regression is reported as a negative saving, not clamped to zero",
      res3.time_saved_min < 0,
      f"time_saved={res3.time_saved_min:+.2f} min")

print()
print("=" * 70)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
