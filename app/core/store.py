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
from threading import Lock

load_dotenv()  # Load variables from .env into os.environ

from app.core.graph_model import TrafficNetwork
from app.core.vrp_problem import VRPProblem

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Keep the in-memory fallback in case DATABASE_URL is not set (e.g. for CI tests)
_networks: Dict[Tuple[str, str], TrafficNetwork] = {}
_network_is_geo: Dict[Tuple[str, str], bool] = {}
_vrp_problems: Dict[Tuple[str, str], Tuple[str, VRPProblem]] = {}

_store_lock = Lock()
MAX_MEMORY_ITEMS = 50

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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    full_name TEXT,
                    company_name TEXT,
                    role TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS networks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    is_geo BOOLEAN,
                    data BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vrp_problems (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    network_id TEXT,
                    data BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Patch existing tables just in case they were created before this update
            cur.execute("ALTER TABLE networks ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            cur.execute("ALTER TABLE vrp_problems ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            cur.execute("ALTER TABLE networks ADD COLUMN IF NOT EXISTS user_id TEXT")
            cur.execute("ALTER TABLE vrp_problems ADD COLUMN IF NOT EXISTS user_id TEXT")
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize tables: {e}")
    finally:
        conn.close()

# Run table creation on import
init_db()

def create_user(email: str, password_hash: str, full_name: str, company_name: str, role: str) -> str:
    user_id = new_id()
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (id, email, password_hash, full_name, company_name, role) VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, email, password_hash, full_name, company_name, role)
                )
            conn.commit()
        finally:
            conn.close()
    return user_id

def get_user_by_email(email: str) -> Optional[dict]:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, email, password_hash, full_name, company_name, role FROM users WHERE email = %s", (email,))
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "email": row[1],
                        "password_hash": row[2],
                        "full_name": row[3],
                        "company_name": row[4],
                        "role": row[5]
                    }
        finally:
            conn.close()
    return None

def get_user_by_id(user_id: str) -> Optional[dict]:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, email, password_hash, full_name, company_name, role FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "email": row[1],
                        "password_hash": row[2],
                        "full_name": row[3],
                        "company_name": row[4],
                        "role": row[5]
                    }
        finally:
            conn.close()
    return None

def get_user_stats(user_id: str) -> dict:
    conn = get_connection()
    stats = {"networks": 0, "vrps": 0}
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM networks WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    stats["networks"] = row[0]
                
                cur.execute("SELECT COUNT(*) FROM vrp_problems WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    stats["vrps"] = row[0]
        finally:
            conn.close()
    return stats

def cleanup_old_db_entries():
    conn = get_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM networks WHERE created_at < NOW() - INTERVAL '24 hours'")
            cur.execute("DELETE FROM vrp_problems WHERE created_at < NOW() - INTERVAL '24 hours'")
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to cleanup database: {e}")
    finally:
        conn.close()

def new_id() -> str:
    return uuid.uuid4().hex[:12]

def put_network(user_id: str, net: TrafficNetwork, is_geo: bool = False) -> str:
    network_id = new_id()
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO networks (id, user_id, is_geo, data) VALUES (%s, %s, %s, %s)",
                    (network_id, user_id, is_geo, psycopg2.Binary(pickle.dumps(net)))
                )
            conn.commit()
        finally:
            conn.close()
        # Fire and forget background cleanup
        cleanup_old_db_entries()
    else:
        with _store_lock:
            _networks[(user_id, network_id)] = net
            _network_is_geo[(user_id, network_id)] = is_geo
            if len(_networks) > MAX_MEMORY_ITEMS:
                oldest = next(iter(_networks))
                del _networks[oldest]
                del _network_is_geo[oldest]
    return network_id

def get_network(user_id: str, network_id: str) -> TrafficNetwork:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM networks WHERE id = %s AND user_id = %s", (network_id, user_id))
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"network_id '{network_id}' not found")
                return pickle.loads(row[0])
        finally:
            conn.close()
    else:
        if (user_id, network_id) not in _networks:
            raise KeyError(f"network_id '{network_id}' not found")
        return _networks[(user_id, network_id)]

def is_geo_network(user_id: str, network_id: str) -> bool:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT is_geo FROM networks WHERE id = %s AND user_id = %s", (network_id, user_id))
                row = cur.fetchone()
                if not row:
                    return False
                return row[0]
        finally:
            conn.close()
    else:
        return _network_is_geo.get((user_id, network_id), False)

def put_vrp(user_id: str, network_id: str, problem: VRPProblem) -> str:
    vrp_id = new_id()
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO vrp_problems (id, user_id, network_id, data) VALUES (%s, %s, %s, %s)",
                    (vrp_id, user_id, network_id, psycopg2.Binary(pickle.dumps(problem)))
                )
            conn.commit()
        finally:
            conn.close()
        # Fire and forget background cleanup
        cleanup_old_db_entries()
    else:
        with _store_lock:
            _vrp_problems[(user_id, vrp_id)] = (network_id, problem)
            if len(_vrp_problems) > MAX_MEMORY_ITEMS:
                oldest = next(iter(_vrp_problems))
                del _vrp_problems[oldest]
    return vrp_id

def get_vrp(user_id: str, vrp_id: str) -> VRPProblem:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM vrp_problems WHERE id = %s AND user_id = %s", (vrp_id, user_id))
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"vrp_id '{vrp_id}' not found")
                return pickle.loads(row[0])
        finally:
            conn.close()
    else:
        if (user_id, vrp_id) not in _vrp_problems:
            raise KeyError(f"vrp_id '{vrp_id}' not found")
        return _vrp_problems[(user_id, vrp_id)][1]

def get_vrp_network_id(user_id: str, vrp_id: str) -> str:
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT network_id FROM vrp_problems WHERE id = %s AND user_id = %s", (vrp_id, user_id))
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"vrp_id '{vrp_id}' not found")
                return row[0]
        finally:
            conn.close()
    else:
        if (user_id, vrp_id) not in _vrp_problems:
            raise KeyError(f"vrp_id '{vrp_id}' not found")
        return _vrp_problems[(user_id, vrp_id)][0]

def load_stress_test_cache() -> dict | None:
    """Loads data/stress_test_cache.json if present, else None."""
    import os, json
    path = os.path.join("data", "stress_test_cache.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
