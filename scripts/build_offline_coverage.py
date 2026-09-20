"""
build_offline_coverage.py
-------------------------
Downloads a whole city's drivable road graph once and writes it to
`data/osm_offline/<name>.json`, so that *drawing a boundary* works with no
internet at all.

How this differs from build_osm_cache.py
========================================
`build_osm_cache.py` produces a ready-to-route network of a few hundred
junctions — one fixed area, already cropped, served by
`POST /api/network/from_cache`.

This produces the opposite: the *whole* city, uncropped, which is never routed
directly. It is the source the backend clips against when someone drags a box
(see `app/core/offline_osm.py`). Bareilly's committed cache covers only the
central 2.7 x 1.8 km, so a box drawn anywhere else in the city had nothing local
to answer it and fell through to the public Overpass API — 88 seconds, in front
of judges, for a feature whose entire pitch is that you can draw anywhere.

The file is deliberately compact rather than readable: node ids are dropped for
array indices and coordinates are rounded to ~0.1 m, which is roughly a third
the size of the same graph in the cache format. It ships in the repo and is read
by the deployed backend, so the size is paid on every cold start.

Usage
=====
    python scripts/build_offline_coverage.py --name bareilly \\
        --label "Bareilly City, Uttar Pradesh" \\
        --south 28.32 --north 28.42 --west 79.38 --east 79.48

Data (c) OpenStreetMap contributors, ODbL. Attribution is written into the file
and must be displayed wherever the map is shown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.overpass_network import (          # noqa: E402
    ATTRIBUTION, build_graph, fetch_overpass, largest_scc, simplify,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache a whole city's road graph for offline boundary drawing.")
    ap.add_argument("--name", default="bareilly")
    ap.add_argument("--label", default="Bareilly City, Uttar Pradesh")
    ap.add_argument("--south", type=float, default=28.32)
    ap.add_argument("--north", type=float, default=28.42)
    ap.add_argument("--west", type=float, default=79.38)
    ap.add_argument("--east", type=float, default=79.48)
    ap.add_argument("--out-dir", default="data/osm_offline")
    ap.add_argument("--overpass-url", default=None,
                    help="use only this Overpass endpoint instead of the built-in mirrors")
    args = ap.parse_args()

    km_ns = (args.north - args.south) * 111
    km_ew = (args.east - args.west) * 97.6
    print(f"Fetching {args.label} — {km_ns:.1f} x {km_ew:.1f} km "
          f"({args.south},{args.west}) -> ({args.north},{args.east}) …")

    try:
        mirrors = [args.overpass_url] if args.overpass_url else None
        payload = fetch_overpass(args.south, args.west, args.north, args.east,
                                 mirrors=mirrors)
    except Exception as exc:
        print(f"  Overpass request failed: {type(exc).__name__}: {exc}")
        return 1

    coords, adj = build_graph(payload)
    print(f"  raw graph         : {len(coords)} nodes")

    coords, adj = simplify(coords, adj)
    print(f"  after simplify    : {len(coords)} nodes")

    keep = largest_scc(adj, set(coords))
    print(f"  strongly connected: {len(keep)} nodes")

    if len(keep) < 200:
        print("  Too few nodes survived — widen the bounding box.")
        return 1

    # Node ids become array indices. OSM ids are 10+ digits and appear once per
    # node and twice per edge, so dropping them is most of the size saving.
    order = sorted(keep)
    index = {osm_id: i for i, osm_id in enumerate(order)}
    nodes = [[round(coords[n][0], 6), round(coords[n][1], 6)] for n in order]

    edges = []
    for u in order:
        for v, (dist, speed) in sorted(adj.get(u, {}).items()):
            if v in index:
                edges.append([index[u], index[v], round(dist, 5), round(speed, 1)])

    doc = {
        "name": args.name,
        "label": args.label,
        "attribution": ATTRIBUTION,
        "source": "Overpass API",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "compact-v1",
        "bbox": {"south": args.south, "north": args.north,
                 "west": args.west, "east": args.east},
        "extent": {
            "south": min(n[0] for n in nodes), "north": max(n[0] for n in nodes),
            "west": min(n[1] for n in nodes), "east": max(n[1] for n in nodes),
        },
        "nodes": nodes,
        "edges": edges,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\nWrote {path}  ({size_mb:.2f} MB)")
    print(f"  {len(nodes)} nodes, {len(edges)} directed edges")
    print(f"  {ATTRIBUTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
