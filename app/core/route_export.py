"""
route_export.py
----------------
Write a solved plan out in formats a fleet operator's own tools can read:
GeoJSON for anything map-shaped, and GPX for a driver's navigation device.

Why this exists
===============
A plan that only exists inside this dashboard cannot be given to a driver. The
optimiser could produce a route for a real city on a real road network and
there was no way to get it out -- which is the difference between something
worth demonstrating and something worth piloting.

Coordinates
===========
Nodes carry x = longitude and y = latitude (see the OSM loaders, which write
them that way, and the dashboard, which reads them back as [y, x] for Leaflet's
lat/lng order).

GeoJSON wants [longitude, latitude]; GPX wants lat="" lon="" attributes. Both
are produced from the same nodes here, which is the whole reason this is one
module: the orders disagree, the mistake is invisible on a world map until
something lands in the wrong hemisphere, and having written it once there is
only one place for it to be wrong.

Synthetic networks
==================
A synthetic city's coordinates are grid units, not degrees. Exporting those as
GeoJSON would produce a file that loads happily and places the depot off the
coast of West Africa, so `is_geo=False` is refused rather than exported. The
caller knows which kind of network it solved; the exporter should not guess.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Colours cycle per vehicle so a multi-van plan stays readable when every route
# is dropped onto one map. Matches the dashboard's own route palette.
ROUTE_COLOURS = [
    "#0284C7", "#7C3AED", "#059669", "#D97706",
    "#DB2777", "#2563EB", "#DC2626",
]


class NotGeoreferenced(ValueError):
    """Raised when asked to export a synthetic network as real-world geometry."""


def _coords_for(node_ids: Sequence[int],
                node_xy: Dict[int, Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(lon, lat) for each node that has coordinates, in order."""
    out = []
    for nid in node_ids:
        xy = node_xy.get(nid)
        if xy is not None:
            out.append(xy)
    return out


def _require_geo(is_geo: bool) -> None:
    if not is_geo:
        raise NotGeoreferenced(
            "This network is synthetic: its coordinates are grid units, not "
            "degrees. Exporting them as GeoJSON or GPX would produce a file that "
            "opens fine and puts the depot in the wrong hemisphere. Solve on a "
            "real road network (a drawn boundary or a shipped district) first."
        )


def routes_to_geojson(
    routes: Iterable[Dict[str, Any]],
    node_xy: Dict[int, Tuple[float, float]],
    depot: Optional[int] = None,
    is_geo: bool = True,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    A FeatureCollection with one LineString per vehicle, plus Point features for
    the depot and every stop.

    `routes` are dicts as the API returns them: vehicle_id, customer_sequence,
    load, and full_path (the road-network nodes including intermediate
    junctions). full_path is preferred for the line geometry, because the
    customer sequence alone would draw straight lines between stops and imply
    the van flew.
    """
    _require_geo(is_geo)
    features: List[Dict[str, Any]] = []

    for i, route in enumerate(routes):
        stops = list(route.get("customer_sequence") or [])
        if not stops:
            continue                      # a parked van is not a route
        path = list(route.get("full_path") or []) or stops
        line = _coords_for(path, node_xy)
        if len(line) >= 2:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[lon, lat] for lon, lat in line]},
                "properties": {
                    "vehicle_id": route.get("vehicle_id", i),
                    "stops": len(stops),
                    "load": route.get("load"),
                    "stroke": ROUTE_COLOURS[i % len(ROUTE_COLOURS)],
                    "stroke-width": 4,
                    # `full_path` is the driven road; without it the line is the
                    # stop order and the distances along it are not real.
                    "geometry_source": "road_network" if route.get("full_path") else "stop_order",
                },
            })

        for order, node in enumerate(stops, start=1):
            xy = node_xy.get(node)
            if xy is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [xy[0], xy[1]]},
                "properties": {"vehicle_id": route.get("vehicle_id", i),
                               "stop_order": order, "node_id": node},
            })

    if depot is not None and depot in node_xy:
        lon, lat = node_xy[depot]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"role": "depot", "node_id": depot},
        })

    collection: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if properties:
        collection["properties"] = properties
    return collection


def routes_to_gpx(
    routes: Iterable[Dict[str, Any]],
    node_xy: Dict[int, Tuple[float, float]],
    depot: Optional[int] = None,
    is_geo: bool = True,
    creator: str = "MargdarshaQ",
) -> str:
    """
    One GPX <trk> per vehicle, with the stops as <wpt> waypoints.

    Track segments carry the driven road geometry; waypoints carry the stops in
    service order, which is what a navigation device lists. A device that only
    understands one of the two still gets something usable.
    """
    _require_geo(is_geo)

    gpx = ET.Element("gpx", {
        "version": "1.1",
        "creator": creator,
        "xmlns": "http://www.topografix.com/GPX/1/1",
    })
    metadata = ET.SubElement(gpx, "metadata")
    ET.SubElement(metadata, "name").text = "Optimised delivery plan"
    ET.SubElement(metadata, "time").text = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    if depot is not None and depot in node_xy:
        lon, lat = node_xy[depot]
        wpt = ET.SubElement(gpx, "wpt", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
        ET.SubElement(wpt, "name").text = "Depot"
        ET.SubElement(wpt, "type").text = "depot"

    for i, route in enumerate(routes):
        stops = list(route.get("customer_sequence") or [])
        if not stops:
            continue
        vehicle = route.get("vehicle_id", i)

        for order, node in enumerate(stops, start=1):
            xy = node_xy.get(node)
            if xy is None:
                continue
            wpt = ET.SubElement(gpx, "wpt", {"lat": f"{xy[1]:.7f}", "lon": f"{xy[0]:.7f}"})
            ET.SubElement(wpt, "name").text = f"Van {vehicle} stop {order}"
            ET.SubElement(wpt, "type").text = "stop"

        path = list(route.get("full_path") or []) or stops
        line = _coords_for(path, node_xy)
        if len(line) < 2:
            continue
        trk = ET.SubElement(gpx, "trk")
        ET.SubElement(trk, "name").text = f"Van {vehicle}"
        seg = ET.SubElement(trk, "trkseg")
        for lon, lat in line:
            ET.SubElement(seg, "trkpt", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(gpx, encoding="unicode")


def geojson_to_text(collection: Dict[str, Any]) -> str:
    return json.dumps(collection, indent=2)
