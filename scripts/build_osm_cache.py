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

# Overpass mirrors, tried in order. The main endpoint rate-limits by IP and
# returns 504s under load, which is exactly when a cache build is wanted, so a
# failure on one host falls through to the next rather than aborting the run.
# The Overpass fetch and graph building live in app.core.overpass_network so the
# API and this script cannot drift apart — the backend serves boundaries drawn
# on the map through exactly this code path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.overpass_network import (          # noqa: E402
    ATTRIBUTION, OVERPASS_MIRRORS, ROAD_SPEEDS,
    build_graph, crop, fetch_overpass, haversine_km, is_oneway,
    largest_scc, simplify, speed_for,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache an OSM road network to JSON.")
    ap.add_argument("--name", default="delhi_central", help="cache file name (no extension)")
    ap.add_argument("--label", default="Central New Delhi", help="human-readable label")
    ap.add_argument("--south", type=float, default=28.615)
    ap.add_argument("--north", type=float, default=28.655)
    ap.add_argument("--west", type=float, default=77.195)
    ap.add_argument("--east", type=float, default=77.245)
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument("--center-lat", type=float, default=28.6315,
                    help="anchor the crop here (default: Connaught Place)")
    ap.add_argument("--center-lon", type=float, default=77.2167)
    ap.add_argument("--out-dir", default="data/networks")
    ap.add_argument("--overpass-url", default=None,
                    help="use only this Overpass endpoint instead of the built-in mirrors")
    args = ap.parse_args()

    print(f"Fetching {args.label} "
          f"({args.south},{args.west}) -> ({args.north},{args.east}) …")
    try:
        mirrors = [args.overpass_url] if args.overpass_url else None
        payload = fetch_overpass(args.south, args.west, args.north, args.east,
                                 mirrors=mirrors)
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

    center = (args.center_lat, args.center_lon) if args.center_lat is not None else None
    keep = crop(adj, keep, args.max_nodes, coords=coords, center=center)
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

    lat_vals = [coords[n][0] for n in keep]
    lon_vals = [coords[n][1] for n in keep]
    doc["extent"] = {"south": round(min(lat_vals), 6), "north": round(max(lat_vals), 6),
                     "west": round(min(lon_vals), 6), "east": round(max(lon_vals), 6)}
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
