"""
schemas.py
-----------
Pydantic request/response models for the SIH26137 FastAPI backend.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

class NetworkGenerateRequest(BaseModel):
    n_nodes: int = Field(30, ge=5, le=500, description="Number of intersections/nodes")
    connectivity: float = Field(0.15, ge=0.05, le=0.5, description="k-nearest-neighbor connectivity fraction")
    seed: int = Field(42, description="Random seed for reproducibility")
    grid_size: float = Field(100.0, description="Size of the 2D plane nodes are scattered in")


class NodeOut(BaseModel):
    id: int
    x: float
    y: float


class EdgeOut(BaseModel):
    u: int
    v: int
    distance: float
    base_time: float
    congestion_factor: float


class NetworkResponse(BaseModel):
    network_id: str
    num_nodes: int
    num_edges: int
    is_geo: bool = Field(False, description="True if node x/y are real longitude/latitude (OSM); False for synthetic planar coords")
    nodes: List[NodeOut]
    edges: List[EdgeOut]


class OSMNetworkRequest(BaseModel):
    place: Optional[str] = Field(None, description="Geocodable place name, e.g. 'Connaught Place, New Delhi, India'")
    north: Optional[float] = Field(None, description="Bounding box north latitude (alternative to `place`)")
    south: Optional[float] = None
    east: Optional[float] = None
    west: Optional[float] = None
    network_type: str = Field("drive", description="osmnx network type: drive, walk, bike, etc.")
    max_nodes: int = Field(2000, ge=10, le=20000)
    seed: int = Field(1, description="Seed for simulated congestion randomization")


# ---------------------------------------------------------------------------
# VRP instance generation
# ---------------------------------------------------------------------------

class VRPGenerateRequest(BaseModel):
    network_id: str
    n_customers: int = Field(15, ge=1, le=300)
    depot: int = Field(0, description="Node id to use as the depot")
    vehicle_capacity: float = Field(100.0, gt=0)
    n_vehicles: Optional[int] = Field(None, description="If omitted, auto-computed from total demand")
    demand_min: float = Field(5.0)
    demand_max: float = Field(20.0)
    horizon: float = Field(480.0, description="Operating time horizon in minutes")
    window_length_min: float = Field(60.0)
    window_length_max: float = Field(180.0)
    service_time: float = Field(10.0)
    seed: int = Field(1)


class CustomerOut(BaseModel):
    node_id: int
    demand: float
    ready_time: float
    due_time: float
    service_time: float


class VRPInstanceResponse(BaseModel):
    vrp_id: str
    network_id: str
    depot: int
    n_vehicles: int
    vehicle_capacity: float
    total_demand: float
    customers: List[CustomerOut]


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

AlgorithmName = Literal["qpso", "ga", "sa", "standard_pso", "greedy"]


class VRPSolveRequest(BaseModel):
    vrp_id: str
    algorithm: AlgorithmName = "qpso"
    n_particles: int = Field(50, ge=5, le=500, description="Swarm/population size (QPSO, GA, standard_pso)")
    max_iter: int = Field(150, ge=1, le=5000)
    seed: int = Field(1)
    use_local_search: bool = Field(True, description="QPSO only: hybridize with 2-opt/or-opt local search")


class RouteOut(BaseModel):
    vehicle_id: int
    customer_sequence: List[int]
    load: float
    full_path: List[int] = Field(default_factory=list, description="Full road-network node sequence (depot -> ... -> depot) including intermediate intersections, for accurate map rendering")
    congestion_delay_min: float = Field(0.0, description="Minutes lost specifically due to traffic congestion factor > 1.0")
    avg_congestion: float = Field(1.0, description="Average congestion factor encountered along route edges")


class VRPSolveResponse(BaseModel):
    algorithm: str
    routes: List[RouteOut]
    total_distance: float
    total_time: float
    capacity_violation: float
    time_window_violation: float
    feasible: bool
    fitness: float
    runtime_ms: float
    n_evaluations: int
    convergence_curve: List[float]
    congestion_delay_min: float = Field(0.0, description="Total fleet minutes lost to traffic congestion")
    avg_congestion: float = Field(1.0, description="Fleet-wide average congestion factor across traversed road segments")
    base_time_min: float = Field(0.0, description="Fleet travel time in free-flow conditions with no congestion")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    vrp_id: str
    max_iter: int = Field(150, ge=1, le=5000)
    seed: int = Field(1)
    algorithms: Optional[List[AlgorithmName]] = Field(
        None, description="Subset of algorithms to run; defaults to all"
    )


class BenchmarkAlgoResult(BaseModel):
    algorithm: str
    fitness: float
    distance: float
    time: float
    feasible: bool
    runtime_ms: float
    n_evaluations: int
    convergence_curve: List[float]
    congestion_delay_min: float = Field(0.0, description="Total fleet minutes lost to traffic congestion")
    avg_congestion: float = Field(1.0, description="Fleet-wide average congestion factor")


class BenchmarkResponse(BaseModel):
    vrp_id: str
    results: List[BenchmarkAlgoResult]


# ---------------------------------------------------------------------------
# Before / After Comparison
# ---------------------------------------------------------------------------

class VRPCompareRequest(BaseModel):
    vrp_id: str
    baseline_algo: AlgorithmName = "greedy"
    optimized_algo: AlgorithmName = "qpso"
    n_particles: int = Field(50, ge=5, le=500)
    max_iter: int = Field(150, ge=1, le=5000)
    seed: int = Field(1)
    use_local_search: bool = Field(True)


class VRPCompareResponse(BaseModel):
    vrp_id: str
    baseline: VRPSolveResponse
    optimized: VRPSolveResponse
    time_saved_pct: float = Field(..., description="Percentage of travel time saved by optimized route vs baseline")
    congestion_avoided_pct: float = Field(..., description="Percentage of traffic congestion delay avoided by optimized route")
    distance_saved_pct: float = Field(..., description="Percentage of total distance saved")
    time_saved_min: float = Field(..., description="Total minutes saved")
    delay_saved_min: float = Field(..., description="Total congestion delay minutes avoided")
    distance_saved_km: float = Field(..., description="Total kilometers saved")


# ---------------------------------------------------------------------------
# Dynamic Traffic Simulation & Re-optimization
# ---------------------------------------------------------------------------

class TrafficIncidentRequest(BaseModel):
    network_id: str
    u: int
    v: int
    factor: float = Field(3.5, ge=1.0, le=10.0, description="Congestion multiplier (e.g. 3.5 = 350% travel time)")
    start_time: float = Field(0.0, ge=0.0)
    duration_min: Optional[float] = Field(None, description="Incident duration in minutes (None = permanent)")


class TrafficIncidentResponse(BaseModel):
    network_id: str
    u: int
    v: int
    factor: float
    message: str


class DynamicSolveRequest(BaseModel):
    vrp_id: str
    incident_u: Optional[int] = Field(None, description="Node u of congested edge (optional, auto-selected if omitted)")
    incident_v: Optional[int] = Field(None, description="Node v of congested edge")
    incident_factor: float = Field(3.5, ge=1.0, le=10.0)
    trigger_time_min: float = Field(60.0, ge=0.0, description="Simulation time (minutes) when traffic incident occurs mid-route")
    algorithm: AlgorithmName = "qpso"
    n_particles: int = Field(50, ge=5, le=500)
    max_iter: int = Field(150, ge=1, le=5000)
    seed: int = Field(1)


class DynamicSolveResponse(BaseModel):
    vrp_id: str
    trigger_time_min: float
    incident: Optional[Dict[str, Any]]
    initial_solution: VRPSolveResponse
    static_affected_solution: VRPSolveResponse
    dynamic_rerouted_solution: VRPSolveResponse
    time_saved_min: float
    time_saved_pct: float
    delay_avoided_min: float
    delay_avoided_pct: float
    tw_violations_avoided: float
    served_customer_ids: List[int]
    unserved_customer_ids: List[int]


