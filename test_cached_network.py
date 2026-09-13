"""
test_cached_network.py
----------------------
Tests for the pre-downloaded real-city road network (data/networks/).

Why this exists: downloading from OpenStreetMap at request time was measured
against the live backend at 44s, 169s and outright failure (502) within the
same hour — OSM rate-limits by IP and the deployed host shares one. The cache
removes that dependency from the demo path.

These checks run offline and need no network access.

Run with:  python test_cached_network.py
"""

import sys
import time

import networkx as nx

from app.core.cached_network import (
    load_cached_network, cache_metadata, available_networks, CachedNetworkNotFound,
)
from app.core.vrp_problem import generate_synthetic_vrp
from app.core.qpso_vrp import QPSOVRPOptimizer

failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


print("=" * 76)
print("1. The cached network ships with the repo and loads")
print("=" * 76)

nets = available_networks()
check("at least one cached network is present", len(nets) >= 1, f"{[n['name'] for n in nets]}")
check("delhi_central is one of them", any(n["name"] == "delhi_central" for n in nets))

started = time.time()
net = load_cached_network("delhi_central")
elapsed_ms = (time.time() - started) * 1000

# The whole point is that this is instant compared with a live download.
check("loads in well under a second", elapsed_ms < 1000, f"{elapsed_ms:.0f} ms")

g = net.graph
check("has a usable number of nodes", g.number_of_nodes() >= 100, f"{g.number_of_nodes()} nodes")
check("has edges", g.number_of_edges() > 0, f"{g.number_of_edges()} directed edges")

print()
print("=" * 76)
print("2. It is a real road network, not a synthetic one")
print("=" * 76)

check("graph is directed", g.is_directed())
check("every stop is reachable from every other", nx.is_strongly_connected(g))

one_way = [(u, v) for u, v in net.road_pairs() if not g.has_edge(v, u)]
check("real one-way streets are preserved", len(one_way) > 0,
      f"{len(one_way)} one-way of {len(list(net.road_pairs()))} roads")

# Real OSM ids are large; synthetic networks number their nodes 0..n-1.
sample = list(g.nodes())[:20]
check("node ids look like OpenStreetMap ids", all(n > 1000 for n in sample),
      f"e.g. {sample[:3]}")

lats = [d["y"] for _, d in g.nodes(data=True)]
lons = [d["x"] for _, d in g.nodes(data=True)]
check("coordinates sit inside the Delhi bounding box",
      28.6 < min(lats) and max(lats) < 28.7 and 77.1 < min(lons) and max(lons) < 77.3,
      f"lat {min(lats):.3f}-{max(lats):.3f}, lon {min(lons):.3f}-{max(lons):.3f}")

print()
print("=" * 76)
print("3. Attribution is carried with the data (OSM is ODbL-licensed)")
print("=" * 76)

meta = cache_metadata("delhi_central")
check("attribution names OpenStreetMap", "OpenStreetMap" in meta["attribution"], meta["attribution"])
check("has a human-readable label", bool(meta["label"]), meta["label"])
check("records when it was fetched", bool(meta["fetched_utc"]), meta["fetched_utc"])

print()
print("=" * 76)
print("4. Congestion still varies per run — only the roads are cached")
print("=" * 76)

a = load_cached_network("delhi_central", congestion_seed=1)
b = load_cached_network("delhi_central", congestion_seed=2)
u, v = next(iter(a.road_pairs()))
differs = any(
    abs(a.graph[x][y]["congestion_factor"] - b.graph[x][y]["congestion_factor"]) > 1e-9
    for x, y in list(a.road_pairs())[:50]
)
check("different seeds give different traffic", differs)

c = load_cached_network("delhi_central", congestion_seed=1)
same = all(
    abs(a.graph[x][y]["congestion_factor"] - c.graph[x][y]["congestion_factor"]) < 1e-12
    for x, y in list(a.road_pairs())[:50]
)
check("the same seed reproduces the same traffic", same)

print()
print("=" * 76)
print("5. It solves end to end")
print("=" * 76)

vrp = generate_synthetic_vrp(net, n_customers=12, depot=0, vehicle_capacity=80, seed=1)
check("depot falls back to a real node id", vrp.depot in g.nodes(), str(vrp.depot))

sol = QPSOVRPOptimizer(vrp, n_particles=15, max_iter=25, seed=1).optimize().best_solution
check("produces a solution", sol is not None)
check("routes cover every customer",
      sorted(n for r in sol.routes for n in r) == sorted(c.node_id for c in vrp.customers))

print()
print("=" * 76)
print("6. A missing cache fails clearly")
print("=" * 76)

try:
    load_cached_network("does_not_exist")
    check("unknown name raises", False, "no exception")
except CachedNetworkNotFound as e:
    check("unknown name raises CachedNetworkNotFound", True)
    check("the error says how to build one", "build_osm_cache" in str(e))

print()
print("=" * 76)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
