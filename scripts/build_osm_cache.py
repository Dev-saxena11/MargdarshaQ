"""
build_osm_cache.py
------------------
Fetches a real road network from OpenStreetMap once and writes it to
`data/networks/<name>.json`, so the demo can serve real city roads instantly
instead of downloading them live.

Why this exists
===============
Downloading the network at demo time proved unreliable from the deployed host:
measured against the live backend, the same request returned 426 nodes in 44s,
316 nodes in 169s, and outright failure (502) minutes apart. OpenStreetMap's
public endpoints rate-limit by IP, and shared hosting means another tenant can
get you blocked. That is a bad thing to depend on in front of judges.

Run this once from a machine that OSM is happy to talk to, commit the result,
and the backend serves it from disk in milliseconds.

This talks to the Overpass API directly rather than going through osmnx, so it
needs no geospatial dependencies and can run anywhere Python can make an HTTPS
request.

Usage
=====
    python scripts/build_osm_cache.py --name delhi_central
    python scripts/build_osm_cache.py --name mumbai --south 19.06 --north 19.10 \
        --west 72.83 --east 72.88

Data © OpenStreetMap contributors, ODbL. Attribution is written into the cache
file and must be displayed wherever the map is shown.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Road classes worth routing a delivery van over. Ordered fastest first; the
# speeds are typical urban free-flow values in km/h and are only used when the
# way carries no explicit maxspeed tag.
ROAD_SPEEDS = {
    "motorway": 80, "motorway_link": 50,
    "trunk": 60, "trunk_link": 40,
    "primary": 50, "primary_link": 35,
    "secondary": 40, "secondary_link": 30,
    "tertiary": 35, "tertiary_link": 25,
    "unclassified": 30, "residential": 25,
    "living_street": 15, "service": 20,
}

ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_overpass(south: float, west: float, north: float, east: float,
                   timeout: int = 180) -> dict:
    """Download drivable ways (and their nodes) inside the bounding box."""
    classes = "|".join(ROAD_SPEEDS)
    query = (
        f"[out:json][timeout:{timeout}];"
        f'way["highway"~"^({classes})$"]({south},{west},{north},{east});'
        f"(._;>;);out body;"
    )
    req = urllib.request.Request(
        OVERPASS_URL,
        data=query.encode("utf-8"),
        headers={"User-Agent": "SIH26137-cache-builder/1.0 (academic project)"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    print(f"  Overpass responded in {time.time() - started:.1f}s "
          f"({len(payload.get('elements', []))} elements)")
    return payload


def is_oneway(tags: dict) -> Tuple[bool, bool]:
    """
    Returns (oneway, reversed).

    `reversed` marks `oneway=-1`, meaning traffic flows against the order the
    way's nodes are listed in. Roundabouts are one-way even without the tag.
    """
    val = str(tags.get("oneway", "")).strip().lower()
    if val in ("yes", "true", "1"):
        return True, False
    if val == "-1":
        return True, True
    if val in ("no", "false", "0"):
        return False, False
    if tags.get("junction") in ("roundabout", "circular"):
        return True, False
    return False, False


def speed_for(tags: dict) -> float:
    """Posted speed if the way has one, else a default for its road class."""
    raw = tags.get("maxspeed")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw:
        token = str(raw).split()[0]
        try:
            kmh = float(token)
            if "mph" in str(raw).lower():
                kmh *= 1.609
            if 5 <= kmh <= 130:
                return kmh
        except ValueError:
            pass
    return float(ROAD_SPEEDS.get(tags.get("highway"), 30))


def build_graph(payload: dict):
    """
    Turn the Overpass response into a directed adjacency map.

    Returns (coords, adjacency) where adjacency[u][v] = (distance_km, speed_kph).
    Every consecutive node pair in a way becomes an edge; one-way tags decide
    whether the reverse edge is added.
    """
    coords: Dict[int, Tuple[float, float]] = {}
    ways: List[dict] = []
    for el in payload.get("elements", []):
        if el["type"] == "node":
            coords[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way" and el.get("nodes"):
            ways.append(el)

    adj: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for way in ways:
        tags = way.get("tags", {})
        oneway, reverse = is_oneway(tags)
        speed = speed_for(tags)
        nodes = way["nodes"]
        if reverse:
            nodes = list(reversed(nodes))
        for a, b in zip(nodes, nodes[1:]):
            if a not in coords or b not in coords or a == b:
                continue
            d = haversine_km(*coords[a], *coords[b])
            if d <= 0:
                continue
            adj[a][b] = (d, speed)
            if not oneway:
                adj[b][a] = (d, speed)
    return coords, adj


def simplify(coords, adj):
    """
    Collapse chains of pass-through nodes into single edges.

    Raw OSM ways carry many intermediate points that exist only to draw the
    road's curve. Keeping them would make the graph mostly geometry and slow the
    solver down for no routing benefit, so any node that isn't a junction or a
    dead end is dissolved and the distances either side are summed.
    """
    undirected = defaultdict(set)
    for u, nbrs in adj.items():
        for v in nbrs:
            undirected[u].add(v)
            undirected[v].add(u)

    keep = {n for n, nbrs in undirected.items() if len(nbrs) != 2}
    if not keep:                      # a perfect ring has no junctions at all
        keep = set(list(undirected)[:1])

    simple: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for start in keep:
        for first in list(adj.get(start, {})):
            dist, speed = adj[start][first]
            prev, node = start, first
            # walk forward until the next junction
            while node not in keep:
                nxt = [w for w in adj.get(node, {}) if w != prev]
                if len(nxt) != 1:
                    break
                d2, s2 = adj[node][nxt[0]]
                dist += d2
                speed = min(speed, s2)
                prev, node = node, nxt[0]
            if node != start and dist > 0:
                existing = simple[start].get(node)
                if existing is None or dist < existing[0]:
                    simple[start][node] = (dist, speed)

    kept_coords = {n: coords[n] for n in keep if n in coords}
    return kept_coords, simple


def largest_scc(adj, nodes):
    """Biggest set of nodes that can all reach each other (Tarjan, iterative)."""
    index, low, on_stack, stack, comps = {}, {}, set(), [], []
    counter = [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, {})))]
        index[root] = low[root] = counter[0]; counter[0] += 1
        stack.append(root); on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter[0]; counter[0] += 1
                    stack.append(nxt); on_stack.add(nxt)
                    work.append((nxt, iter(adj.get(nxt, {}))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop(); on_stack.discard(w); comp.append(w)
                    if w == node:
                        break
                comps.append(comp)
    return set(max(comps, key=len)) if comps else set()


def crop(adj, nodes, max_nodes):
    """Breadth-first neighbourhood of at most max_nodes, kept strongly connected."""
    if len(nodes) <= max_nodes:
        return nodes
    start = next(iter(nodes))
    seen, queue = {start}, [start]
    while queue and len(seen) < max_nodes:
        cur = queue.pop(0)
        for nxt in adj.get(cur, {}):
            if nxt in nodes and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
                if len(seen) >= max_nodes:
                    break
    # cropping severs edges, so re-run SCC or some stops become unreachable
    sub = {u: {v: d for v, d in adj[u].items() if v in seen} for u in seen if u in adj}
    return largest_scc(sub, seen)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache an OSM road network to JSON.")
    ap.add_argument("--name", default="delhi_central", help="cache file name (no extension)")
    ap.add_argument("--label", default="Central New Delhi", help="human-readable label")
    ap.add_argument("--south", type=float, default=28.615)
    ap.add_argument("--north", type=float, default=28.655)
    ap.add_argument("--west", type=float, default=77.195)
    ap.add_argument("--east", type=float, default=77.245)
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument("--out-dir", default="data/networks")
    args = ap.parse_args()

    print(f"Fetching {args.label} "
          f"({args.south},{args.west}) -> ({args.north},{args.east}) …")
    try:
        payload = fetch_overpass(args.south, args.west, args.north, args.east)
    except Exception as exc:
        print(f"  Overpass request failed: {type(exc).__name__}: {exc}")
        print("  OSM rate-limits by IP; wait a few minutes and try again.")
        return 1

    coords, adj = build_graph(payload)
    print(f"  raw graph        : {len(coords)} nodes, {sum(len(v) for v in adj.values())} directed edges")

    coords, adj = simplify(coords, adj)
    print(f"  after simplify   : {len(coords)} nodes, {sum(len(v) for v in adj.values())} directed edges")

    keep = largest_scc(adj, set(coords))
    print(f"  strongly connected: {len(keep)} nodes")

    keep = crop(adj, keep, args.max_nodes)
    print(f"  after crop       : {len(keep)} nodes")

    if len(keep) < 20:
        print("  Too few nodes survived — widen the bounding box.")
        return 1

    edges = []
    for u in sorted(keep):
        for v, (dist, speed) in sorted(adj.get(u, {}).items()):
            if v in keep:
                edges.append({"u": u, "v": v, "km": round(dist, 5), "kph": round(speed, 1)})

    oneway_count = sum(1 for e in edges if not any(
        o["u"] == e["v"] and o["v"] == e["u"] for o in edges))

    doc = {
        "name": args.name,
        "label": args.label,
        "attribution": ATTRIBUTION,
        "source": "Overpass API",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bbox": {"south": args.south, "north": args.north,
                 "west": args.west, "east": args.east},
        "nodes": [{"id": n, "lat": round(coords[n][0], 7), "lon": round(coords[n][1], 7)}
                  for n in sorted(keep)],
        "edges": edges,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))

    size_kb = os.path.getsize(path) / 1024
    print(f"\nWrote {path}  ({size_kb:.0f} KB)")
    print(f"  {len(doc['nodes'])} nodes, {len(edges)} directed edges, "
          f"{oneway_count} of them one-way")
    print(f"  {ATTRIBUTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
