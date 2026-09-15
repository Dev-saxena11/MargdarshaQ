"""
store.py
---------
Persistent store for generated networks and VRP instances using PostgreSQL.
Uses psycopg2 and pickle to store serialized objects in BYTEA columns.
"""

from __future__ import annotations
import os
import uuid
import pickle
import psycopg2
from psycopg2.extensions import connection
import logging
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into os.environ

from app.core.graph_model import TrafficNetwork

from app.core.vrp_problem import VRPProblem

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Keep the in-memory fallback in case DATABASE_URL is not set (e.g. for CI tests)
_networks: Dict[str, TrafficNetwork] = {}
_network_is_geo: Dict[str, bool] = {}
_vrp_problems: Dict[str, Tuple[str, VRPProblem]] = {}

def get_connection() -> Optional[connection]:
    if not DATABASE_URL:
        return None
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def init_db():
    conn = get_connection()
    if not conn:
        logger.warning("No DATABASE_URL set, falling back to in-memory store.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS networks (
                    id TEXT PRIMARY KEY,
                    is_geo BOOLEAN,
                    data BYTEA
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vrp_problems (
                    id TEXT PRIMARY KEY,
                    network_id TEXT,
                    data BYTEA
                )
            """)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize tables: {e}")
    finally:
        conn.close()

# Run table creation on import
init_db()

def new_id() -> str:
    return uuid.uuid4().hex[:12]

def put_network(net: TrafficNetwork, is_geo: bool = False) -> str:
    network_id = new_id()
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO networks (id, is_geo, data) VALUES (%s, %s, %s)",
                    (network_id, is_geo, psycopg2.Binary(pickle.dumps(net)))
                )
            conn.commit()
        finally:
            conn.close()
    else:
        _networks[network_id] = net
        _network_is_geo[network_id] = is_geo
    return network_id

def get_network(network_id: str) -> TrafficNetwork:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM networks WHERE id = %s", (network_id,))
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"network_id '{network_id}' not found")
                return pickle.loads(row[0])
        finally:
            conn.close()
    else:
        if network_id not in _networks:
            raise KeyError(f"network_id '{network_id}' not found")
        return _networks[network_id]

def is_geo_network(network_id: str) -> bool:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT is_geo FROM networks WHERE id = %s", (network_id,))
                row = cur.fetchone()
                if not row:
                    return False
                return row[0]
        finally:
            conn.close()
    else:
        return _network_is_geo.get(network_id, False)

def put_vrp(network_id: str, problem: VRPProblem) -> str:
    vrp_id = new_id()
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO vrp_problems (id, network_id, data) VALUES (%s, %s, %s)",
                    (vrp_id, network_id, psycopg2.Binary(pickle.dumps(problem)))
                )
            conn.commit()
        finally:
            conn.close()
    else:
        _vrp_problems[vrp_id] = (network_id, problem)
    return vrp_id

def get_vrp(vrp_id: str) -> VRPProblem:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM vrp_problems WHERE id = %s", (vrp_id,))
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"vrp_id '{vrp_id}' not found")
                return pickle.loads(row[0])
        finally:
            conn.close()
    else:
        if vrp_id not in _vrp_problems:
            raise KeyError(f"vrp_id '{vrp_id}' not found")
        return _vrp_problems[vrp_id][1]

def get_vrp_network_id(vrp_id: str) -> str:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT network_id FROM vrp_problems WHERE id = %s", (vrp_id,))
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"vrp_id '{vrp_id}' not found")
                return row[0]
        finally:
            conn.close()
    else:
        if vrp_id not in _vrp_problems:
            raise KeyError(f"vrp_id '{vrp_id}' not found")
        return _vrp_problems[vrp_id][0]
