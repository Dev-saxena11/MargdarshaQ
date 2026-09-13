"""
osm_network.py
----------------
Loads a REAL city road network from OpenStreetMap (via the osmnx library)
and converts it into our TrafficNetwork format, so the exact same QPSO / VRP
/ benchmarking code that runs on synthetic graphs also runs on real streets.

Requires `osmnx` (see requirements.txt) and an internet connection at
runtime -- this module is not exercised in the build sandbox (no network
access there), so test it first thing after `pip install -r requirements.txt`
on your own machine.

osmnx API note: osmnx's function names/signatures have shifted across major
versions (1.x vs 2.x). This module targets osmnx>=1.6,<2.0 (pinned in
requirements.txt) and includes small compatibility fallbacks where the API
is known to differ, but if you upgrade osmnx and something breaks here,
check the osmnx changelog for `graph_from_place` / `graph_from_bbox` /
undirected-graph conversion first -- those are the three calls used below.
"""

from __future__ import annotations
from typing import Optional, Tuple

from app.core.graph_model import TrafficNetwork


def _to_undirected(G):
    """Handle osmnx/networkx API differences across versions."""
    try:
        import osmnx as ox
        return ox.utils_graph.get_undirected(G)  # osmnx 1.x
    except AttributeError:
        try:
            import osmnx as ox
            return ox.convert.to_undirected(G)    # osmnx 2.x
        except AttributeError:
            import networkx as nx
            return nx.Graph(G)                    # last-resort fallback


def load_osm_network(
    place: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,  # (north, south, east, west)
    network_type: str = "drive",
    simplify: bool = True,
    max_nodes: Optional[int] = 2000,
    congestion_seed: int = 1,
    congestion_low: float = 1.0,
    congestion_high: float = 2.5,
) -> TrafficNetwork:
    """
    Fetch a real road network and convert it to our TrafficNetwork.

    place: a geocodable place name, e.g. "Connaught Place, New Delhi, India"
    bbox: alternative to `place` -- (north, south, east, west) lat/lon bounds
    max_nodes: if the fetched graph is larger than this, keep only the
               largest connected component and warn (large OSM graphs make
               VRP's O(n_customers^2) Dijkstra precompute and QPSO's
               per-evaluation route decode noticeably slower -- fine for a
               few thousand nodes, but keep instances modest for interactive
               use).

    Node coordinates are stored as (x=longitude, y=latitude) -- i.e. GEOGRAPHIC
    coordinates, not planar ones like the synthetic generator uses. Callers
    (e.g. the API layer) should track this so the frontend knows to render
    on a real map (Leaflet) rather than a plain 2D canvas.
    """
    try:
        import osmnx as ox
    except ImportError as e:
        raise ImportError(
            "osmnx is required for OSM network loading. Install it with "
            "`pip install osmnx` (already in requirements.txt -- did you "
            "run `pip install -r requirements.txt`?)."
        ) from e

    if place:
        G = ox.graph_from_place(place, network_type=network_type, simplify=simplify)
    elif bbox:
        north, south, east, west = bbox
        G = ox.graph_from_bbox(north, south, east, west, network_type=network_type, simplify=simplify)
    else:
        raise ValueError("Provide either `place` (a geocodable name) or `bbox` (north, south, east, west).")

    # add real-world speed limits (imputed where missing) and travel times
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception as e:
        print(f"Warning: ox.add_edge_speeds failed ({e}), falling back to default speeds.")

    # CRITICAL FIX: Ensure the graph is STRONGLY connected! 
    # Otherwise, random customers might be placed in dead-ends or unreachable
    # components (like gated communities), which causes the routing engine
    # to return an empty path, leading to a blank map on the frontend!
    import networkx as nx

    # 1. Ensure the base graph is strongly connected
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()

    # 2. Crop to max_nodes while preserving strong connectivity
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        # Simple BFS outward to collect a local neighborhood
        start_node = list(G.nodes())[0]
        keep_nodes = {start_node}
        for u, v in nx.bfs_edges(G, start_node):
            keep_nodes.add(v)
            if len(keep_nodes) >= max_nodes:
                break
        
        G = G.subgraph(list(keep_nodes)).copy()
        
        # After cropping, the edges might be severed. We MUST run SCC again
        # so the resulting subgraph has no unreachable nodes!
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_scc).copy()

    # 3. Build the backend TrafficNetwork using the strongly connected DiGraph
    net = TrafficNetwork()
    for node_id, data in G.nodes(data=True):
        net.add_node(node_id, x=data["x"], y=data["y"])  # x=lon, y=lat

    for u, v, data in G.edges(data=True):
        # osmnx MultiGraph edges can carry 'length' in meters and 'speed_kph'
        length_m = data.get("length", 100.0)
        speed_kph = data.get("speed_kph", 40.0)
        if isinstance(speed_kph, list):  # osmnx sometimes returns a list for multi-edges
            speed_kph = speed_kph[0]
        distance_km = length_m / 1000.0
        
        # Since we are using the directed G, we preserve real-world one-way streets!
        net.add_edge(u, v, distance=distance_km, base_speed_kmph=speed_kph)

    net.randomize_congestion(seed=congestion_seed, low=congestion_low, high=congestion_high)

    return net


if __name__ == "__main__":
    # Manual smoke test -- requires internet + osmnx installed. Not run in CI/sandbox.
    net = load_osm_network(place="New Delhi, India", max_nodes=500)
    print(f"Loaded OSM network: {net.num_nodes()} nodes, {net.graph.number_of_edges()} edges")
