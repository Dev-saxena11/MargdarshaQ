"""
schemas.py
-----------
Pydantic request/response models for the SIH26137 FastAPI backend.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Dict, Any


# ---------------------------------------------------------------------------
# Authentication & User Profile
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    email: str
    password: str = Field(min_length=6)
    full_name: str
    company_name: str
    role: str

class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserProfileStats(BaseModel):
    full_name: str
    company_name: str
    role: str
    email: str
    operational_zones_mapped: int
    route_plans_dispatched: int

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
    # Present for real-world networks so the UI can name the place on screen and
    # carry OpenStreetMap's required attribution. None for synthetic networks.
    area_label: Optional[str] = Field(None, description="Human-readable area, e.g. 'Connaught Place, New Delhi'")
    attribution: Optional[str] = Field(None, description="Data attribution to display with the map")
    # Thinning is invisible on the map -- what comes back is a perfectly good
    # network of main roads, and nothing about it says the residential streets
    # were dropped to fit the node budget. Reported so the UI can say so.
    nodes_available: Optional[int] = Field(None, description="Junctions the drawn area actually contains")
    thinned: bool = Field(False, description="True when smaller roads were dropped to fit max_nodes")


class OSMNetworkRequest(BaseModel):
    place: Optional[str] = Field(None, description="Geocodable place name, e.g. 'Connaught Place, New Delhi, India'")
    north: Optional[float] = Field(None, description="Bounding box north latitude (alternative to `place`)")
    south: Optional[float] = None
    east: Optional[float] = None
    west: Optional[float] = None
    network_type: str = Field("drive", description="osmnx network type: drive, walk, bike, etc.")
    max_nodes: int = Field(2000, ge=10, le=20000)
    seed: int = Field(1, description="Seed for simulated congestion randomization")


class CachedNetworkRequest(BaseModel):
    """
    Load a real OSM road network that was downloaded ahead of time and shipped
    in the repository, instead of fetching from OpenStreetMap at request time.
    """
    name: str = Field("delhi_central", description="Cached network name (see data/networks/)")
    seed: int = Field(42, description="Seed for simulated congestion randomization")


# ---------------------------------------------------------------------------
# VRP instance generation
# ---------------------------------------------------------------------------

class VRPGenerateRequest(BaseModel):
    network_id: str
    n_customers: int = Field(15, ge=1, le=300)
    depot: int = Field(0, description="Node id to use as the depot")
    vehicle_capacity: float = Field(100.0, gt=0)
    n_vehicles: Optional[int] = Field(None, description="If omitted, auto-computed from total demand")
    require_all_vehicles: bool = Field(
        False,
        description="Treat n_vehicles as a fleet that must all be sent out rather than a "
                    "ceiling. Off by default: parking a van the plan does not need is "
                    "normally the better answer.",
    )
    demand_min: float = Field(5.0)
    demand_max: float = Field(20.0)
    horizon: float = Field(480.0, description="Operating time horizon in minutes")
    window_length_min: float = Field(60.0)
    window_length_max: float = Field(180.0)
    service_time: float = Field(10.0)
    seed: int = Field(1)
    time_dependent: bool = Field(False, description="Price each leg by the time of day the vehicle departs, so routes account for rush hour")
    bucket_minutes: float = Field(30.0, ge=5.0, le=120.0, description="Width of each time bucket when time_dependent is on")
    customer_nodes: Optional[List[int]] = Field(
        None,
        description="Explicit stop node ids, in place of random sampling. Used by the "
                    "map-driven builder where the engineer clicks the stops on the network; "
                    "when given, n_customers is ignored.",
    )

    @model_validator(mode="after")
    def validate_vrp_params(self) -> "VRPGenerateRequest":
        if self.demand_min > self.demand_max:
            raise ValueError("demand_min cannot be greater than demand_max.")
        if self.demand_max > self.vehicle_capacity:
            raise ValueError(f"vehicle_capacity ({self.vehicle_capacity}) cannot be smaller than demand_max ({self.demand_max}).")
        if self.window_length_min > self.window_length_max:
            raise ValueError("window_length_min cannot be greater than window_length_max.")
        if self.window_length_max > self.horizon:
            raise ValueError(f"window_length_max ({self.window_length_max}) cannot be greater than the horizon ({self.horizon}).")
        return self


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
    # One run per algorithm shows which algorithm won that run, which is not the
    # same claim as which algorithm is better -- a metaheuristic's result moves
    # with its seed. Repeating the run is what turns a single figure into a
    # spread you can show somebody.
    trials: int = Field(
        1, ge=1, le=25,
        description="Runs per algorithm, each with a different seed. 1 (the "
                    "default) reproduces the previous single-run behaviour "
                    "exactly, including the seed used."
    )
    success_threshold_pct: float = Field(
        5.0, ge=0.0, le=100.0,
        description="A trial counts as a success when its fitness is within "
                    "this percentage of the best fitness found by any "
                    "algorithm in any trial of this run."
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

    # Repeated-run statistics. With trials=1 these hold the single run, so a
    # caller that ignores them sees exactly what it saw before: every scalar
    # field above describes the BEST trial, not an average, so "the result"
    # still means the same thing.
    trials: int = Field(1, description="How many runs this algorithm was given")
    fitness_samples: List[float] = Field(
        default_factory=list,
        description="Fitness of each trial, in seed order. The raw sample, not "
                    "a summary, so the client can draw a box plot without the "
                    "server having to guess which quartiles it wants."
    )
    runtime_samples_ms: List[float] = Field(
        default_factory=list, description="Wall-clock time of each trial"
    )
    time_samples: List[float] = Field(
        default_factory=list,
        description="Fleet time of each trial. With distance_samples this gives "
                    "a cloud of points per algorithm rather than one, which is "
                    "what makes a time-versus-distance trade-off plot readable."
    )
    distance_samples: List[float] = Field(
        default_factory=list, description="Distance driven in each trial"
    )
    feasible_rate: float = Field(
        1.0, description="Fraction of trials that satisfied every hard constraint"
    )
    success_rate: float = Field(
        1.0, description="Fraction of trials landing within "
                         "success_threshold_pct of the best fitness seen"
    )


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


class RoadClosureRequest(BaseModel):
    network_id: str
    u: int
    v: int
    reopen: bool = Field(False, description="Lift a closure instead of applying one")
    both_directions: bool = Field(True, description="Close the return direction too, where one exists")
    # Optional: when given, the closure is checked against this instance's stops
    # and depot, so a road that strands a customer is reported rather than
    # quietly turned into a large routing penalty.
    vrp_id: Optional[str] = Field(None, description="Check reachability for this instance's stops")


class RoadClosureResponse(BaseModel):
    network_id: str
    u: int
    v: int
    closed: bool
    edges_changed: int
    closed_roads: List[List[int]] = Field(default_factory=list)
    stranded_customers: List[int] = Field(default_factory=list,
        description="Stops no longer reachable from the depot. Non-empty means this closure has no valid plan.")
    message: str


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


# In-Dashboard AI Assistant (Issue #32)
# ---------------------------------------------------------------------------

class AssistantContext(BaseModel):
    scenario_name: Optional[str] = Field(None, description="Active demo or custom scenario name")
    num_nodes: Optional[int] = Field(None, description="Network intersection count")
    num_customers: Optional[int] = Field(None, description="Customer order count")
    num_vehicles: Optional[int] = Field(None, description="Active vehicle fleet size")
    depot: Optional[int] = Field(None, description="Depot node id")
    baseline_algo: Optional[str] = Field(None, description="Baseline algorithm name (e.g. Greedy Nearest-Neighbor)")
    optimized_algo: Optional[str] = Field(None, description="Optimized algorithm name (e.g. QPSO)")
    time_saved_pct: Optional[float] = Field(None, description="Percentage time saved")
    time_saved_min: Optional[float] = Field(None, description="Minutes saved")
    delay_saved_pct: Optional[float] = Field(None, description="Congestion delay percentage avoided")
    delay_saved_min: Optional[float] = Field(None, description="Congestion delay minutes avoided")
    dist_saved_pct: Optional[float] = Field(None, description="Distance percentage saved")
    dist_saved_km: Optional[float] = Field(None, description="Kilometers saved")
    baseline_time: Optional[float] = None
    optimized_time: Optional[float] = None
    baseline_dist: Optional[float] = None
    optimized_dist: Optional[float] = None
    baseline_delay: Optional[float] = None
    optimized_delay: Optional[float] = None
    baseline_late: Optional[float] = None
    optimized_late: Optional[float] = None
    baseline_feasible: Optional[bool] = None
    optimized_feasible: Optional[bool] = None
    avg_congestion_baseline: Optional[float] = None
    avg_congestion_optimized: Optional[float] = None
    benchmark_ranks: Optional[List[Dict[str, Any]]] = None
    active_view: Optional[str] = Field(None, description="Active UI view: 'judge' or 'control'")
    slot_filling_active: bool = Field(False, description="True if a multi-turn parameter collection is in progress")
    collected_params: Dict[str, Any] = Field(default_factory=dict, description="VRP configuration parameters collected so far")
    current_prompt: Optional[str] = Field(None, description="The specific parameter currently being asked for")
    # What the user has actually clicked on the map. The assistant asks for the
    # depot and the stops by handing over to the map rather than by asking for
    # node ids, which nobody outside this codebase knows.
    selected_depot: Optional[int] = Field(None, description="Depot node the user clicked on the map")
    selected_stops: List[int] = Field(default_factory=list, description="Stop nodes the user clicked on the map")
    # The network those clicks belong to. A boundary clipped out of the offline
    # map numbers its nodes from zero, while the full cached city carries OSM
    # ids, so the two id spaces do not overlap at all. Planning has to run
    # against the network the user actually picked on, not a fresh load of it.
    active_network_id: Optional[str] = Field(None, description="network_id currently loaded in the control room")


class AssistantChatRequest(BaseModel):
    message: str = Field("", description="User query or selected prompt text")
    chip: Optional[str] = Field(None, description="Preset chip identifier if triggered by quick prompt")
    context: Optional[AssistantContext] = Field(None, description="Current session state and metrics snapshot")


class AssistantChatResponse(BaseModel):
    reply: str = Field(..., description="Formatted markdown explanation for presentation")
    suggested_chips: List[str] = Field(default_factory=list, description="Follow-up quick action prompts")
    metrics_summary: Optional[Dict[str, Any]] = Field(None, description="Key extracted quantitative metrics for UI badges")
    context: Optional[AssistantContext] = Field(None, description="Returned context to allow state tracking across turns")
    map_action: Optional[Dict[str, Any]] = Field(None, description="Data payload to auto-sync the frontend map")


# ---------------------------------------------------------------------------
# Project-Grounded Chatbot / RAG (Issue #33)
# ---------------------------------------------------------------------------

class ChatSource(BaseModel):
    document: str = Field(..., description="Path or name of the source documentation file")
    section: str = Field(..., description="Section or heading title within the document")
    snippet: str = Field(..., description="Exemplar text excerpt from the passage")
    relevance_score: float = Field(..., description="BM25 relevance score")


class ChatRequest(BaseModel):
    message: Optional[str] = Field(None, description="User question or query text")
    query: Optional[str] = Field(None, description="Alternative alias for message")
    context: Optional[AssistantContext] = Field(None, description="Active session state and metrics snapshot")
    top_k: int = Field(3, ge=1, le=10, description="Number of retrieved context passages")

    def get_query(self) -> str:
        return (self.message or self.query or "").strip()


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Grounded markdown response")
    sources: List[ChatSource] = Field(default_factory=list, description="Retrieved documentation sources and citations")
    suggested_chips: List[str] = Field(default_factory=list, description="Suggested follow-up judge queries")


class RAGStatusResponse(BaseModel):
    total_chunks: int
    indexed_files: List[str]
    status: str

