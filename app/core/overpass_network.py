"""
overpass_network.py
-------------------
Loads a real road network from OpenStreetMap by talking to the Overpass API
directly, with no osmnx and no geocoding step.

Why this exists
===============
`osm_network.py` goes through osmnx, and osmnx failed outright on the deployed
backend: every bounding box, rural or urban, came back as

    Failed to fetch OSM network: cannot access local variable 'response'
    where it is not associated with a value

That is osmnx's own downloader referencing a variable it never assigned after
its HTTP call fell over, so the real error is lost and the whole draw-a-boundary
feature is dead in production. Talking to Overpass over plain urllib removes
osmnx, geopandas and the geocoder from the request path, keeps the actual HTTP
error, and lets a busy mirror fall through to the next one.

This is the same code path that built the cached networks in data/networks/,
which is the strongest argument for it: it is known to work against Overpass
from an ordinary machine.

Where the query goes
====================
By default, the three public mirrors listed below. They are shared and
rate-limited by IP, so for a live demo set OVERPASS_URL to a local instance
(see docker-compose.overpass.yml and DEPLOYMENT.md) and the public endpoints
become a fallback rather than the critical path.

Data (c) OpenStreetMap contributors, ODbL. Attribution must be displayed
wherever the map is shown.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from app.core.graph_model import TrafficNetwork


OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# A local Overpass instance is preferred over the public mirrors whenever one is
# configured (see docker-compose.overpass.yml). The public endpoints are shared
# and rate-limited by IP: measured answers range from 2s to 25s and include
# outright 502s minutes apart, which is not something a live demo should rest
# on. A local instance answers the same query in well under a second and cannot
# be throttled by another tenant's traffic.
#
# Set OVERPASS_URL=http://localhost:12345/api/interpreter to switch. Leaving it
# unset keeps the previous behaviour exactly, so this is opt-in.
DEFAULT_LOCAL_TIMEOUT = 20
OVERPASS_LOADING_MESSAGE = (
    "Map data is still loading in Overpass. Please try again in a moment."
)


def local_overpass_url() -> Optional[str]:
    """
    The configured local Overpass endpoint, or None when unset.

    Read at call time rather than at import: load_dotenv() runs from
    app.core.store, so reading this at import time would make the value depend
    on module import order — the kind of bug that only shows up once, on the
    machine that matters.
    """
    return (os.getenv("OVERPASS_URL") or "").strip() or None


def public_fallback_enabled() -> bool:
    """
    Whether a failing local instance may fall through to the public mirrors.

    Defaults to on, so forgetting to start Docker costs latency rather than the
    whole draw-a-boundary feature. Turn it off to catch a misconfigured local
    instance during a rehearsal, when a silent fallback would hide the problem
    until the day it matters.
    """
    return (os.getenv("OVERPASS_PUBLIC_FALLBACK") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def local_timeout() -> int:
    """
    Seconds to wait on the local instance before giving up on it.

    A healthy local instance answers a city-sized box in under a second, so the
    default is generous. It exists to bound the damage when the container is up
    but wedged: the fallback race costs another 30s on top, and the point of
    this whole exercise was to not spend a minute staring at a map.
    """
    raw = (os.getenv("OVERPASS_LOCAL_TIMEOUT") or "").strip()
    if not raw:
        return DEFAULT_LOCAL_TIMEOUT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LOCAL_TIMEOUT
    return value if value > 0 else DEFAULT_LOCAL_TIMEOUT


def _status_url(interpreter_url: str) -> str:
    """Map .../api/interpreter to .../api/status for local readiness checks."""
    base = interpreter_url.split("?", 1)[0].rstrip("/")
    if base.endswith("/interpreter"):
        return base[:-len("/interpreter")] + "/status"
    return base + "/status"


def local_overpass_ready() -> Tuple[bool, Optional[str]]:
    """
    Readiness of the configured local Overpass instance.

    Returns (ready, detail). When no local instance is configured we report ready
    because Overpass requests are not gated on local-import state in that mode.
    """
    local = local_overpass_url()
    if not local:
        return True, None

    try:
        status_url = _status_url(local)
        req = urllib.request.Request(
            status_url,
            headers={"User-Agent": "SIH26137-overpass-health/1.0"},
        )
        with urllib.request.urlopen(req, timeout=min(local_timeout(), 5)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return False, OVERPASS_LOADING_MESSAGE

    text = raw.lower()
    if "slot available now" in text or "slots available now" in text:
        return True, None
    if "currently running queries" in text and "rate limit" in text:
        return True, None
    return False, OVERPASS_LOADING_MESSAGE


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


def build_query(south: float, west: float, north: float, east: float,
                timeout: int = 120) -> str:
    """The Overpass QL for every drivable way (and its nodes) in the box."""
    classes = "|".join(ROAD_SPEEDS)
    return (
        f"[out:json][timeout:{timeout}];"
        f'way["highway"~"^({classes})$"]({south},{west},{north},{east});'
        f"(._;>;);out body;"
    )


def ask_endpoint(url: str, body: bytes, socket_timeout: int):
    """POST the query to one endpoint. Returns (url, payload, seconds taken)."""
    req = urllib.request.Request(
        url, data=body,
        headers={"User-Agent": "SIH26137-cache-builder/1.0 (academic project)"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=socket_timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return url, payload, time.time() - started


def fetch_local_overpass(url: str, body: bytes) -> Optional[dict]:
    """
    Ask the local Overpass instance, returning None if it cannot answer.

    An empty result counts as "cannot answer" here, which is the opposite of how
    the public mirrors are read. A local instance is built from a regional
    .osm.pbf extract, so a box outside that region returns zero elements and is
    indistinguishable from open farmland. Falling through to the public mirrors
    tells the two apart, instead of confidently reporting "no drivable roads" for
    Mumbai on a Delhi-only extract — which is the exact failure a judge would
    find by drawing somewhere unplanned.
    """
    try:
        _, payload, took = ask_endpoint(url, body, local_timeout())
    except Exception as exc:
        print(f"  local Overpass ({url}): {type(exc).__name__}: {exc}")
        return None

    if not payload.get("elements"):
        print(f"  local Overpass answered in {took:.1f}s but returned nothing — "
              f"the box is probably outside the imported extract")
        return None

    print(f"  local Overpass answered in {took:.1f}s "
          f"({len(payload['elements'])} elements)")
    return payload


def fetch_overpass(south: float, west: float, north: float, east: float,
                   timeout: int = 120, mirrors: List[str] = None,
                   attempts: int = 3) -> dict:
    """
    Download drivable ways (and their nodes) inside the bounding box.

    A local Overpass instance is tried first when OVERPASS_URL is set, and the
    public mirrors are only touched if it fails to produce anything. With the
    variable unset the behaviour is unchanged from before.

    The public mirrors are raced, not tried in turn. Measured on a 22x33 km box,
    the sequential version cost 35s: overpass-api.de spent 10s before returning a
    504, and only then did the next mirror start, taking a further 24.5s. The
    parsing and graph building either side of it total 0.07s, so essentially all
    of the wait was one slow endpoint being asked politely one at a time.

    Racing them costs three cheap requests and returns as soon as any endpoint
    answers, so a dead or busy mirror costs nothing rather than dominating.
    Losing requests are left to finish into a daemon thread and ignored;
    Overpass has no cancel, and the alternative is making the user wait for a
    reply nobody will read.
    """
    body = build_query(south, west, north, east, timeout).encode("utf-8")

    # An explicit `mirrors` argument is a deliberate override — the cache
    # builder's --overpass-url flag, and the tests — so it bypasses the
    # env-configured local instance rather than silently ignoring the caller.
    if mirrors is None:
        local = local_overpass_url()
        if local:
            payload = fetch_local_overpass(local, body)
            if payload is not None:
                return payload
            if not public_fallback_enabled():
                raise RuntimeError(
                    f"The local Overpass instance at {local} did not answer, and "
                    "OVERPASS_PUBLIC_FALLBACK is off. Start it with "
                    "`docker compose -f docker-compose.overpass.yml up -d`, or "
                    "unset OVERPASS_PUBLIC_FALLBACK to allow the public mirrors."
                )
            print("  falling back to the public Overpass mirrors")

    mirrors = mirrors or OVERPASS_MIRRORS
    # Observed answers from a working mirror land between 2s and 25s. A round is
    # only over once every mirror has settled, so this ceiling is what a single
    # hung connection costs before the retry — 60s here turned a 2.4s answer
    # into a 68s wait. Short enough to fail over quickly, long enough to keep
    # the slow-but-real replies.
    socket_timeout = min(timeout, 30)
    errors = []
    empty_replies = []

    for round_no in range(attempts):
        # Deliberately not a `with` block: its __exit__ joins every worker, so
        # returning the winner inside one would wait for the losers anyway.
        pool = ThreadPoolExecutor(max_workers=len(mirrors))
        try:
            futures = {pool.submit(ask_endpoint, url, body, socket_timeout): url
                       for url in mirrors}
            try:
                for fut in as_completed(futures, timeout=socket_timeout + 5):
                    url = futures[fut]
                    try:
                        url, payload, took = fut.result()
                    except Exception as exc:
                        errors.append(exc)
                        print(f"  {url.split('/')[2]}: {type(exc).__name__}: {exc}")
                        continue

                    # First to answer is not the same as first to answer with
                    # anything. One mirror returns an empty result in under a
                    # second for boxes that plainly have roads in them, and a
                    # plain race hands the win to exactly that kind of endpoint.
                    # Empty replies are kept aside and only used if every mirror
                    # agrees the box is empty, which is a real answer.
                    if not payload.get("elements"):
                        print(f"  {url.split('/')[2]}: answered in {took:.1f}s but returned nothing")
                        empty_replies.append(payload)
                        continue

                    print(f"  {url.split('/')[2]} answered first in {took:.1f}s "
                          f"({len(payload.get('elements', []))} elements)")
                    return payload
            except TimeoutError:
                errors.append(RuntimeError("every Overpass mirror timed out"))
        finally:
            # Hand back control immediately; the losing requests run themselves
            # out in the background and their results are dropped.
            pool.shutdown(wait=False, cancel_futures=True)

        if round_no + 1 < attempts:
            # Mirrors that refuse do so quickly; there is little to gain by
            # pausing long before asking again.
            wait = 2 * (round_no + 1)
            print(f"  every mirror refused — waiting {wait}s before retrying")
            time.sleep(wait)

    # Every mirror that managed to reply said the box holds no roads. That is an
    # answer about the box, not a failure to fetch, and the caller words it as
    # such.
    if empty_replies:
        return empty_replies[0]

    raise (errors[-1] if errors else RuntimeError("no Overpass mirror configured"))


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


def crop(adj, nodes, max_nodes, coords=None, center=None):
    """
    Breadth-first neighbourhood of at most max_nodes, kept strongly connected.

    `center` anchors the crop on a landmark (lat, lon). Without it the walk
    starts from an arbitrary node and drifts to whichever corner of the bounding
    box that happens to sit in — the first build of this cache ended up in the
    north-west and did not even contain Connaught Place, despite being labelled
    "Central New Delhi". Anchoring keeps the result recognisable and repeatable.
    """
    if len(nodes) <= max_nodes:
        return nodes
    if center and coords:
        clat, clon = center
        start = min(nodes, key=lambda n: haversine_km(coords[n][0], coords[n][1], clat, clon))
    else:
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


# Keyed on the bounding box rounded to ~10 m and the parameters that change the
# result. Bounded so a long-running server cannot grow one road network at a
# time into a memory leak.
_NETWORK_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_LIMIT = 24


def _cache_key(north, south, east, west, max_nodes, congestion_seed):
    return (round(north, 4), round(south, 4), round(east, 4), round(west, 4),
            int(max_nodes), int(congestion_seed))


def load_overpass_network(
    north: float,
    south: float,
    east: float,
    west: float,
    max_nodes: int = 400,
    congestion_seed: int = 42,
    congestion_low: float = 1.0,
    congestion_high: float = 2.5,
    timeout: int = 120,
) -> Tuple[TrafficNetwork, dict]:
    """
    Fetch the drivable roads inside a bounding box and build a TrafficNetwork.

    Returns (network, meta). Each edge is added in its own direction only, so
    one-way streets stay one-way through to the routing engine.

    Raises ValueError when the box holds too little road to route on, which is
    a real answer for a box dragged over farmland or water — the caller should
    say so rather than reporting it as a download failure.
    """
    if north <= south or east <= west:
        raise ValueError("The boundary is empty — drag a box rather than clicking.")

    key = _cache_key(north, south, east, west, max_nodes, congestion_seed)
    cached = _NETWORK_CACHE.get(key)
    if cached is not None:
        _NETWORK_CACHE.move_to_end(key)
        print("  served from cache")
        # A fresh TrafficNetwork each time: callers mutate theirs (congestion,
        # incidents), and handing out the same object would let one request's
        # road closure leak into the next.
        return _network_from_parts(cached), dict(cached["meta"])

    payload = fetch_overpass(south, west, north, east, timeout=timeout)

    coords, adj = build_graph(payload)
    if not coords:
        raise ValueError("No drivable roads were found inside that boundary.")

    coords, adj = simplify(coords, adj)
    keep = largest_scc(adj, set(coords))

    center = ((north + south) / 2, (east + west) / 2)
    keep = crop(adj, keep, max_nodes, coords=coords, center=center)

    if len(keep) < 5:
        raise ValueError(
            f"Only {len(keep)} connected junctions were found inside that boundary — "
            "drag a larger box, or one over a denser street network."
        )

    parts = {
        "nodes": [(nid, coords[nid][1], coords[nid][0]) for nid in keep],   # (id, lon, lat)
        "edges": [(u, v, d, kph)
                  for u in keep
                  for v, (d, kph) in adj.get(u, {}).items() if v in keep],
        "congestion": (congestion_seed, congestion_low, congestion_high),
    }
    parts["meta"] = {
        "attribution": ATTRIBUTION,
        "bbox": {"north": north, "south": south, "east": east, "west": west},
        "num_nodes": len(parts["nodes"]),
        "num_edges": len(parts["edges"]),
    }

    _NETWORK_CACHE[key] = parts
    while len(_NETWORK_CACHE) > _CACHE_LIMIT:
        _NETWORK_CACHE.popitem(last=False)

    return _network_from_parts(parts), dict(parts["meta"])


def _network_from_parts(parts: dict) -> TrafficNetwork:
    """Rebuilds a TrafficNetwork from the cached primitives."""
    seed, low, high = parts["congestion"]
    net = TrafficNetwork()
    for nid, lon, lat in parts["nodes"]:
        # x=lon, y=lat — the convention the map and the rest of the app use.
        net.add_node(nid, x=lon, y=lat)
    for u, v, dist_km, speed_kph in parts["edges"]:
        net.add_edge(u, v, distance=dist_km, base_speed_kmph=speed_kph,
                     bidirectional=False)
    net.randomize_congestion(seed=seed, low=low, high=high)
    return net
