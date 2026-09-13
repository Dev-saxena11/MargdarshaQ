"""
test_directed_network.py
------------------------
Tests that TrafficNetwork is directed and that one-way streets survive.

Background: the network used to be an undirected nx.Graph. The OSM loader
iterated OpenStreetMap's directed graph and claimed in a comment to "preserve
real-world one-way streets", but feeding directed edges into an undirected
graph made every one-way street two-way — so the router could send a vehicle
the wrong way up a one-way road on a real city map.

Runs offline; no OSM download or network access required.

Run with:  python test_directed_network.py
"""

import sys

import networkx as nx

from app.core.graph_model import TrafficNetwork, generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp
from app.core.qpso_vrp import QPSOVRPOptimizer

failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


print("=" * 74)
print("1. One-way streets stay one-way")
print("=" * 74)

net = TrafficNetwork()
for n in (1, 2, 3):
    net.add_node(n, x=n, y=0)
net.add_edge(1, 2, distance=1.0, bidirectional=False)   # one-way, as the OSM loader adds
net.add_edge(2, 3, distance=1.0)                        # ordinary two-way road

check("graph is directed", net.graph.is_directed())
check("one-way edge exists in its own direction", net.graph.has_edge(1, 2))
check("one-way edge has NO reverse", not net.graph.has_edge(2, 1),
      "this is the bug the DiGraph change fixes")
check("two-way road exists both ways",
      net.graph.has_edge(2, 3) and net.graph.has_edge(3, 2))
check("road_pairs() counts each road once", list(net.road_pairs()) == [(1, 2), (2, 3)],
      str(list(net.road_pairs())))

print()
print("=" * 74)
print("2. A router cannot drive the wrong way up a one-way street")
print("=" * 74)

# 1 -> 2 only. Going 2 -> 1 must be impossible.
check("path exists along the one-way direction", nx.has_path(net.graph, 1, 2))
check("no path against the one-way direction", not nx.has_path(net.graph, 2, 1),
      "a vehicle must not be routed backwards up a one-way road")

print()
print("=" * 74)
print("3. Synthetic networks remain fully two-way and connected")
print("=" * 74)

syn = generate_synthetic_city_graph(n_nodes=30, seed=42)
g = syn.graph
one_way = [(u, v) for u, v in syn.road_pairs() if not g.has_edge(v, u)]
mismatched = [(u, v) for u, v in syn.road_pairs()
              if abs(g[u][v]["congestion_factor"] - g[v][u]["congestion_factor"]) > 1e-12]

check("no accidental one-way synthetic roads", not one_way, f"{len(one_way)} found")
check("strongly connected (every stop reachable)", nx.is_strongly_connected(g))
check("both directions share a congestion level", not mismatched, f"{len(mismatched)} differ")
check("directed edges are exactly twice the roads",
      g.number_of_edges() == 2 * len(list(syn.road_pairs())),
      f"{g.number_of_edges()} edges / {len(list(syn.road_pairs()))} roads")

print()
print("=" * 74)
print("4. Incidents act on the whole road and are reversible")
print("=" * 74)

u, v = next(iter(syn.road_pairs()))
before_fwd = g[u][v]["congestion_factor"]
before_rev = g[v][u]["congestion_factor"]
syn.apply_incident(u, v, 4.0)
check("incident slows the forward direction", g[u][v]["congestion_factor"] == 4.0)
check("incident slows the reverse direction too", g[v][u]["congestion_factor"] == 4.0,
      "a jam on a two-way street blocks both ways")
syn.clear_incidents()
check("forward direction restored", abs(g[u][v]["congestion_factor"] - before_fwd) < 1e-12)
check("reverse direction restored", abs(g[v][u]["congestion_factor"] - before_rev) < 1e-12)

print()
print("=" * 74)
print("5. Seeded results are unchanged by the refactor")
print("=" * 74)

# Values recorded from the undirected implementation immediately before the
# DiGraph change. They must not move: randomize_congestion draws once per road
# (not once per direction), so the RNG sequence is identical.
EXPECTED = {"total_time": 430.0415, "total_distance": 144.6800, "fitness": 315.8969}

vrp = generate_synthetic_vrp(generate_synthetic_city_graph(n_nodes=30, seed=42),
                             n_customers=12, depot=0, vehicle_capacity=80, seed=1)
sol = QPSOVRPOptimizer(vrp, n_particles=20, max_iter=40, seed=1).optimize().best_solution

for field, expected in EXPECTED.items():
    got = getattr(sol, field)
    check(f"{field} matches the pre-refactor value", abs(got - expected) < 0.01,
          f"expected {expected}, got {got:.4f}")

print()
print("=" * 74)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
