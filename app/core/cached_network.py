"""
cached_network.py
-----------------
Loads a real road network that was downloaded from OpenStreetMap ahead of time
and committed to `data/networks/`.

Why not just download it on demand
==================================
Fetching from OpenStreetMap at request time proved unreliable from the deployed
host. Measured against the live backend, the same bounding box returned 426
nodes in 44s, 316 nodes in 169s, and plain failure (502) within the same hour —
OSM's public endpoints rate-limit by IP, and on shared hosting another tenant's
traffic counts against you. A live demo cannot rest on that.

The network is therefore fetched once (see `scripts/build_osm_cache.py`),
committed, and loaded from disk here in milliseconds. The roads are real and so
are the one-way restrictions; only the download moved offline.

Congestion is still randomised per run, exactly as it is for the synthetic
networks, so the traffic conditions are not frozen along with the geometry.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

from app.core.graph_model import TrafficNetwork

# data/networks/ relative to the repository root (this file is app/core/…)
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "networks",
)


class CachedNetworkNotFound(FileNotFoundError):
    """Raised when the requested cache file isn't present on disk."""


def available_networks() -> List[Dict[str, str]]:
    """Every cached network on disk, as {name, label, nodes, attribution}."""
    out: List[Dict[str, str]] = []
    if not os.path.isdir(CACHE_DIR):
        return out
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            doc = _read_cache(fname[:-5])
        except Exception:
            continue
        out.append({
            "name": doc.get("name", fname[:-5]),
            "label": doc.get("label", fname[:-5]),
            "nodes": len(doc.get("nodes", [])),
            "attribution": doc.get("attribution", ""),
        })
    return out


@lru_cache(maxsize=8)
def _read_cache(name: str) -> dict:
    """Parsed cache document. Cached in memory — the file never changes at runtime."""
    safe = os.path.basename(name)           # never let a name escape CACHE_DIR
    path = os.path.join(CACHE_DIR, f"{safe}.json")
    if not os.path.isfile(path):
        raise CachedNetworkNotFound(
            f"No cached network named '{safe}'. Available: "
            f"{[n['name'] for n in available_networks()] or 'none'}. "
            f"Build one with: python scripts/build_osm_cache.py --name {safe}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_cached_network(
    name: str = "delhi_central",
    congestion_seed: Optional[int] = 42,
    congestion_low: float = 1.0,
    congestion_high: float = 3.0,
) -> TrafficNetwork:
    """
    Build a TrafficNetwork from a cached OSM download.

    Each stored edge is added in its own direction only: a two-way street was
    written to the cache as both directions, while a one-way street has just
    the one, so real turn restrictions survive into the routing engine.
    """
    doc = _read_cache(name)

    net = TrafficNetwork()
    for node in doc["nodes"]:
        # x=lon, y=lat — matches the convention the OSM loader and the map use
        net.add_node(node["id"], x=node["lon"], y=node["lat"])

    known = {n["id"] for n in doc["nodes"]}
    for edge in doc["edges"]:
        u, v = edge["u"], edge["v"]
        if u not in known or v not in known:
            continue
        net.add_edge(u, v, distance=edge["km"], base_speed_kmph=edge["kph"],
                     bidirectional=False)

    net.randomize_congestion(seed=congestion_seed, low=congestion_low, high=congestion_high)
    return net


def cache_metadata(name: str = "delhi_central") -> dict:
    """Label, attribution and fetch date — for display alongside the map."""
    doc = _read_cache(name)
    return {
        "name": doc.get("name", name),
        "label": doc.get("label", name),
        "attribution": doc.get("attribution", ""),
        "fetched_utc": doc.get("fetched_utc", ""),
        "bbox": doc.get("bbox", {}),
        "num_nodes": len(doc.get("nodes", [])),
        "num_edges": len(doc.get("edges", [])),
    }
