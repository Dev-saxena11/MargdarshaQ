"""
store.py
---------
Simple in-memory store for generated networks and VRP instances, keyed by
UUID. Fine for a hackathon prototype / single-process demo. For a real
deployment, swap this for Redis or a database -- the store interface
(get/put) is intentionally minimal so that swap is a small change.
"""

from __future__ import annotations
import uuid
from typing import Dict, Tuple

from app.core.graph_model import TrafficNetwork
from app.core.vrp_problem import VRPProblem

_networks: Dict[str, TrafficNetwork] = {}
_network_is_geo: Dict[str, bool] = {}
_vrp_problems: Dict[str, Tuple[str, VRPProblem]] = {}  # vrp_id -> (network_id, problem)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def put_network(net: TrafficNetwork, is_geo: bool = False) -> str:
    network_id = new_id()
    _networks[network_id] = net
    _network_is_geo[network_id] = is_geo
    return network_id


def get_network(network_id: str) -> TrafficNetwork:
    if network_id not in _networks:
        raise KeyError(f"network_id '{network_id}' not found")
    return _networks[network_id]


def is_geo_network(network_id: str) -> bool:
    return _network_is_geo.get(network_id, False)


def put_vrp(network_id: str, problem: VRPProblem) -> str:
    vrp_id = new_id()
    _vrp_problems[vrp_id] = (network_id, problem)
    return vrp_id


def get_vrp(vrp_id: str) -> VRPProblem:
    if vrp_id not in _vrp_problems:
        raise KeyError(f"vrp_id '{vrp_id}' not found")
    return _vrp_problems[vrp_id][1]


def get_vrp_network_id(vrp_id: str) -> str:
    if vrp_id not in _vrp_problems:
        raise KeyError(f"vrp_id '{vrp_id}' not found")
    return _vrp_problems[vrp_id][0]
