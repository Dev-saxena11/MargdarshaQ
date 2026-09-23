"""
tests/test_route_export.py
---------------------------
Writing a solved plan out as GeoJSON and GPX.

The failure worth guarding against here is not a crash. Longitude and latitude
are both plausible-looking numbers, GeoJSON orders them [lon, lat] and GPX
orders them lat="" lon="", and a file with them swapped parses perfectly,
renders perfectly, and puts a Bareilly delivery round in the South Atlantic.
Nothing downstream complains. So the tests assert where the coordinates
actually land, not merely that a file came out.

The second guard is against exporting a synthetic network. Its coordinates are
grid units, so the same silent failure applies with no way to detect it from
the file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from app.core.route_export import (
    NotGeoreferenced,
    geojson_to_text,
    routes_to_geojson,
    routes_to_gpx,
)

GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}

# Somewhere in Bareilly: x is longitude, y is latitude, as the OSM loaders
# write them and as the dashboard reads them back.
NODE_XY = {
    0: (79.4304, 28.3670),        # depot
    1: (79.4310, 28.3680),
    2: (79.4320, 28.3690),
    3: (79.4330, 28.3700),
    10: (79.4315, 28.3685),       # an intermediate junction, not a stop
    11: (79.4325, 28.3695),
}

ROUTES = [
    {"vehicle_id": 0, "customer_sequence": [1, 2], "load": 30.0,
     "full_path": [0, 10, 1, 11, 2, 0]},
    {"vehicle_id": 1, "customer_sequence": [3], "load": 15.0, "full_path": [0, 3, 0]},
    {"vehicle_id": 2, "customer_sequence": [], "load": 0.0, "full_path": []},
]


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def test_one_line_per_loaded_vehicle():
    gj = routes_to_geojson(ROUTES, NODE_XY, depot=0)
    lines = [f for f in gj["features"] if f["geometry"]["type"] == "LineString"]
    assert len(lines) == 2, "a parked van is not a route"


def test_coordinates_are_longitude_then_latitude():
    """
    GeoJSON's order. Swapped, the file still parses and still renders -- it
    just draws the round in the wrong hemisphere.
    """
    gj = routes_to_geojson(ROUTES, NODE_XY, depot=0)
    line = next(f for f in gj["features"] if f["geometry"]["type"] == "LineString")
    for lon, lat in line["geometry"]["coordinates"]:
        assert 79.0 < lon < 80.0, f"{lon} is not a Bareilly longitude"
        assert 28.0 < lat < 29.0, f"{lat} is not a Bareilly latitude"


def test_the_line_follows_the_road_not_the_stop_order():
    """
    full_path carries the intermediate junctions. Falling back to the stop
    sequence would draw straight lines between stops and imply the van flew.
    """
    gj = routes_to_geojson(ROUTES, NODE_XY, depot=0)
    line = next(f for f in gj["features"] if f["geometry"]["type"] == "LineString")
    assert len(line["geometry"]["coordinates"]) == 6
    assert line["properties"]["geometry_source"] == "road_network"


def test_a_route_without_road_geometry_says_so():
    """
    Straight lines between stops are still worth exporting, but the distances
    along them are not real and the file should not imply otherwise.
    """
    gj = routes_to_geojson(
        [{"vehicle_id": 0, "customer_sequence": [1, 2, 3], "load": 1.0}], NODE_XY, depot=0
    )
    line = next(f for f in gj["features"] if f["geometry"]["type"] == "LineString")
    assert line["properties"]["geometry_source"] == "stop_order"


def test_every_stop_and_the_depot_become_points():
    gj = routes_to_geojson(ROUTES, NODE_XY, depot=0)
    points = [f for f in gj["features"] if f["geometry"]["type"] == "Point"]
    assert len(points) == 4                       # three stops plus the depot
    assert any(p["properties"].get("role") == "depot" for p in points)


def test_stops_carry_their_service_order():
    gj = routes_to_geojson(ROUTES, NODE_XY, depot=0)
    first_van = [f["properties"] for f in gj["features"]
                 if f["geometry"]["type"] == "Point"
                 and f["properties"].get("vehicle_id") == 0]
    assert [p["stop_order"] for p in first_van] == [1, 2]


def test_each_vehicle_gets_its_own_colour():
    gj = routes_to_geojson(ROUTES, NODE_XY, depot=0)
    strokes = [f["properties"]["stroke"] for f in gj["features"]
               if f["geometry"]["type"] == "LineString"]
    assert len(set(strokes)) == len(strokes)


def test_the_collection_serialises_to_valid_json():
    import json
    parsed = json.loads(geojson_to_text(routes_to_geojson(ROUTES, NODE_XY, depot=0)))
    assert parsed["type"] == "FeatureCollection"


def test_a_node_with_no_coordinates_is_skipped_rather_than_crashing():
    routes = [{"vehicle_id": 0, "customer_sequence": [1, 999], "load": 1.0,
               "full_path": [0, 1, 999, 0]}]
    gj = routes_to_geojson(routes, NODE_XY, depot=0)
    points = [f for f in gj["features"] if f["geometry"]["type"] == "Point"]
    assert [p["properties"]["node_id"] for p in points if "node_id" in p["properties"]] == [1, 0]


# ---------------------------------------------------------------------------
# GPX
# ---------------------------------------------------------------------------

def test_gpx_is_well_formed_and_correctly_namespaced():
    root = ET.fromstring(routes_to_gpx(ROUTES, NODE_XY, depot=0))
    assert root.tag.endswith("gpx")
    assert root.get("version") == "1.1"


def test_gpx_has_one_track_per_loaded_vehicle():
    root = ET.fromstring(routes_to_gpx(ROUTES, NODE_XY, depot=0))
    assert len(root.findall("g:trk", GPX_NS)) == 2


def test_gpx_attributes_are_latitude_and_longitude_the_right_way_round():
    """GPX names them, so a swap here is unambiguous -- and just as invisible."""
    root = ET.fromstring(routes_to_gpx(ROUTES, NODE_XY, depot=0))
    for point in root.findall(".//g:trkpt", GPX_NS) + root.findall("g:wpt", GPX_NS):
        assert 28.0 < float(point.get("lat")) < 29.0
        assert 79.0 < float(point.get("lon")) < 80.0


def test_gpx_waypoints_cover_the_depot_and_every_stop():
    root = ET.fromstring(routes_to_gpx(ROUTES, NODE_XY, depot=0))
    names = [w.find("g:name", GPX_NS).text for w in root.findall("g:wpt", GPX_NS)]
    assert names[0] == "Depot"
    assert len(names) == 4


# ---------------------------------------------------------------------------
# Synthetic networks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exporter", [routes_to_geojson, routes_to_gpx])
def test_a_synthetic_network_is_refused_rather_than_exported(exporter):
    """
    Grid units are not degrees. The file would open fine and be wrong, which is
    worse than no file at all.
    """
    with pytest.raises(NotGeoreferenced, match="synthetic"):
        exporter(ROUTES, NODE_XY, depot=0, is_geo=False)
