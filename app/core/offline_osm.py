"""
offline_osm.py
--------------
Serves a hand-drawn bounding box from a city road graph committed to the repo,
so that drawing a boundary needs no internet at all.

Why this exists
===============
`from_cache` serves a few fixed districts instantly, and `from_osm` serves
anything at all by asking the public Overpass API. The gap between them is the
demo's headline feature: the judge drags a box wherever they like, and that box
by definition was never cached, so it went to Overpass — measured at 44s, 169s
and a plain 502 within one hour, and 88s on the deployed backend while a judge
watched.

A local Overpass instance (see docker-compose.overpass.yml) closes that gap, but
only for a backend running on the same machine. The deployed backend cannot
reach it, so the committed coverage file is what makes the deployed demo
independent of the network.

Inside a covered city this is not a fallback or an approximation: it is the same
OpenStreetMap geometry, including one-way restrictions, that Overpass would have
returned — just clipped locally instead of over the wire.

Boxes outside every covered city still go to Overpass. That is deliberate: the
alternative is quietly returning a half-empty graph for a box that straddles the
edge, and a slow honest answer beats a fast misleading one.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from app.core.graph_model import TrafficNetwork
from app.core.overpass_network import ATTRIBUTION, crop, largest_scc

# data/osm_offline/ relative to the repository root (this file is app/core/…)
COVERAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "osm_offline",
)

# A box this far inside the coverage edge is still served offline. Roughly 100 m:
# the extent is the outermost *junction*, not the edge of the imported area, so
# insisting on strict containment rejects boxes drawn right up to the city edge
# that the data covers perfectly well.
EDGE_TOLERANCE_DEG = 0.001


@lru_cache(maxsize=4)
def _read_coverage(name: str) -> dict:
    safe = os.path.basename(name)
    path = os.path.join(COVERAGE_DIR, f"{safe}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def available_coverage() -> List[Dict]:
    """Every offline city on disk, as {name, label, extent, nodes}."""
    out: List[Dict] = []
    if not os.path.isdir(COVERAGE_DIR):
        return out
    for fname in sorted(os.listdir(COVERAGE_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            doc = _read_coverage(fname[:-5])
        except Exception:
            continue
        out.append({
            "name": doc.get("name", fname[:-5]),
            "label": doc.get("label", fname[:-5]),
            "extent": doc.get("extent", {}),
            "nodes": len(doc.get("nodes", [])),
            "attribution": doc.get("attribution", ATTRIBUTION),
        })
    return out


def coverage_for_bbox(north: float, south: float,
                      east: float, west: float) -> Optional[str]:
    """Name of a covered city that contains this box, or None."""
    for cov in available_coverage():
        ext = cov.get("extent") or {}
        if not ext:
            continue
        t = EDGE_TOLERANCE_DEG
        if (south >= ext["south"] - t and north <= ext["north"] + t
                and west >= ext["west"] - t and east <= ext["east"] + t):
            return cov["name"]
    return None


def _thin_to_budget(adj, nodes, max_nodes):
    """
    Drop the lowest road classes until the network fits the node budget.

    Speed stands in for road class (see ROAD_SPEEDS): service lanes and
    residential streets go first, arterials last. Thresholds are tried from the
    lowest up and the first one that fits wins, so the result keeps as much
    detail as the budget allows while still reaching the edges of the box.

    Returns the original graph untouched if no threshold both fits the budget
    and leaves something routable behind.
    """
    speeds = sorted({kph for outs in adj.values() for (_, kph) in outs.values()})
    for threshold in speeds[1:]:            # the lowest keeps everything
        sub: Dict[int, Dict[int, Tuple[float, float]]] = {}
        for u, outs in adj.items():
            kept = {v: d for v, d in outs.items() if d[1] >= threshold}
            if kept:
                sub[u] = kept
        touched = (set(sub) | {v for outs in sub.values() for v in outs}) & nodes
        scc = largest_scc(sub, touched)
        if len(scc) < 10:
            break                            # any further cut is emptier still
        if len(scc) <= max_nodes:
            return scc, sub
    return nodes, adj


def load_offline_network(
    north: float, south: float, east: float, west: float,
    max_nodes: int = 400,
    congestion_seed: Optional[int] = 42,
    congestion_low: float = 1.0,
    congestion_high: float = 3.0,
) -> Tuple[TrafficNetwork, dict]:
    """
    Clip a drawn box out of a committed city graph.

    Raises LookupError when no covered city contains the box, and ValueError
    when the box is covered but holds too little road to route on — the same
    distinction `from_osm` already makes, so the caller's error handling and the
    messages the user sees do not change.
    """
    name = coverage_for_bbox(north, south, east, west)
    if name is None:
        raise LookupError("No offline coverage for this bounding box.")

    doc = _read_coverage(name)
    raw_nodes = doc["nodes"]          # [[lat, lon], …] indexed by node id

    inside = {
        i for i, (lat, lon) in enumerate(raw_nodes)
        if south <= lat <= north and west <= lon <= east
    }
    if len(inside) < 10:
        raise ValueError(
            "That box has almost no drivable road in it. Draw a larger one, "
            "or one over streets rather than open ground."
        )

    adj: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for u, v, km, kph in doc["edges"]:
        if u in inside and v in inside:
            adj.setdefault(u, {})[v] = (km, kph)

    keep = largest_scc(adj, inside)
    if len(keep) < 10:
        raise ValueError(
            "The roads in that box do not join up into a network you can route "
            "on. Draw a box that covers a connected set of streets."
        )

    # Fitting the box into max_nodes by walking outwards from the middle keeps
    # a dense blob and leaves the rest of the drawn area blank — measured on a
    # 7 x 6 km box, the result covered about a quarter of it in each direction,
    # so most of what the user drew came back empty. Dropping the smallest road
    # classes first instead keeps a skeleton that still spans the whole box,
    # which is both what drawing that box asked for and the roads a van would
    # actually use.
    if len(keep) > max_nodes:
        keep, adj = _thin_to_budget(adj, keep, max_nodes)

    # Only if even the arterials alone overshoot — a very large box — does the
    # centre-out walk come back, because at that point something has to give
    # and staying connected matters more than spanning.
    if len(keep) > max_nodes:
        coords = {i: (raw_nodes[i][0], raw_nodes[i][1]) for i in keep}
        center = ((north + south) / 2, (east + west) / 2)
        keep = crop(adj, keep, max_nodes, coords=coords, center=center)

    net = TrafficNetwork()
    for i in sorted(keep):
        lat, lon = raw_nodes[i]
        net.add_node(i, x=lon, y=lat)      # x=lon, y=lat, as the map expects
    for u in sorted(keep):
        for v, (km, kph) in adj.get(u, {}).items():
            if v in keep:
                net.add_edge(u, v, distance=km, base_speed_kmph=kph,
                             bidirectional=False)

    net.randomize_congestion(seed=congestion_seed, low=congestion_low,
                             high=congestion_high)

    meta = {
        "attribution": doc.get("attribution", ATTRIBUTION),
        "area_label": doc.get("label", name),
        "source": "offline",
        "coverage": name,
    }
    return net, meta
