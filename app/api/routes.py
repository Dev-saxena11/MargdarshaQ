"""
routes.py
----------
FastAPI endpoints for SIH26137: quantum-inspired traffic route optimization.

Workflow:
    1. POST /api/network/generate   -> synthetic city road network
    2. POST /api/vrp/generate       -> VRP instance (customers/demands/time windows) on that network
    3. POST /api/vrp/solve          -> solve with one algorithm (qpso / ga / sa / standard_pso / greedy)
    4. POST /api/benchmark/run      -> solve with ALL algorithms and return a side-by-side comparison
"""

from __future__ import annotations
import time
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core.graph_model import generate_synthetic_city_graph
from app.core.osm_network import load_osm_network
from app.core.vrp_problem import generate_synthetic_vrp, VRPProblem, evaluate_solution
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.classical_baselines_vrp import (
    run_ga_vrp, run_sa_vrp, run_standard_pso_vrp, run_greedy_nn_vrp
)
from app.core.dynamic_vrp import simulate_dynamic_reroute
from app.core import store
from app.core.assistant import assistant_engine
from app.models.schemas import (
    NetworkGenerateRequest, NetworkResponse, NodeOut, EdgeOut, OSMNetworkRequest,
    VRPGenerateRequest, VRPInstanceResponse, CustomerOut,
    VRPSolveRequest, VRPSolveResponse, RouteOut,
    BenchmarkRequest, BenchmarkResponse, BenchmarkAlgoResult,
    VRPCompareRequest, VRPCompareResponse,
    TrafficIncidentRequest, TrafficIncidentResponse, DynamicSolveRequest, DynamicSolveResponse,
    AssistantChatRequest, AssistantChatResponse,
)

router = APIRouter(prefix="/api")


def _network_payload(net):
    """
    Node and edge lists for the API response.

    The network is a DiGraph, so an ordinary two-way road is two opposing edges.
    Drawing both would paint every street twice and report double the edge count,
    so each physical road is emitted once (see TrafficNetwork.road_pairs).
    """
    nodes = [NodeOut(id=n, x=d["x"], y=d["y"]) for n, d in net.graph.nodes(data=True)]
    edges = []
    for u, v in net.road_pairs():
        d = net.graph[u][v]
        edges.append(EdgeOut(u=u, v=v, distance=d["distance"], base_time=d["base_time"],
                             congestion_factor=d["congestion_factor"]))
    return nodes, edges



# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

@router.post("/network/generate", response_model=NetworkResponse)
def generate_network(req: NetworkGenerateRequest):
    net = generate_synthetic_city_graph(
        n_nodes=req.n_nodes, connectivity=req.connectivity,
        seed=req.seed, grid_size=req.grid_size,
    )
    network_id = store.put_network(net, is_geo=False)

    nodes, edges = _network_payload(net)

    return NetworkResponse(
        network_id=network_id, num_nodes=net.num_nodes(),
        num_edges=len(edges), is_geo=False, nodes=nodes, edges=edges,
    )


@router.post("/network/from_osm", response_model=NetworkResponse)
def generate_network_from_osm(req: OSMNetworkRequest):
    bbox = None
    if req.north is not None and req.south is not None and req.east is not None and req.west is not None:
        bbox = (req.north, req.south, req.east, req.west)
    elif not req.place:
        raise HTTPException(status_code=400, detail="Provide either `place` or all four bbox bounds (north/south/east/west).")

    try:
        net = load_osm_network(
            place=req.place, bbox=bbox, network_type=req.network_type,
            max_nodes=req.max_nodes, congestion_seed=req.seed,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch OSM network: {e}")

    network_id = store.put_network(net, is_geo=True)

    nodes, edges = _network_payload(net)

    return NetworkResponse(
        network_id=network_id, num_nodes=net.num_nodes(),
        num_edges=len(edges), is_geo=True, nodes=nodes, edges=edges,
    )


@router.get("/network/{network_id}", response_model=NetworkResponse)
def get_network(network_id: str):
    try:
        net = store.get_network(network_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    is_geo = store.is_geo_network(network_id)
    nodes, edges = _network_payload(net)
    return NetworkResponse(
        network_id=network_id, num_nodes=net.num_nodes(),
        num_edges=len(edges), is_geo=is_geo, nodes=nodes, edges=edges,
    )


# ---------------------------------------------------------------------------
# VRP instance generation
# ---------------------------------------------------------------------------

@router.post("/vrp/generate", response_model=VRPInstanceResponse)
def generate_vrp(req: VRPGenerateRequest):
    try:
        net = store.get_network(req.network_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        problem = generate_synthetic_vrp(
            net, n_customers=req.n_customers, depot=req.depot,
            vehicle_capacity=req.vehicle_capacity, n_vehicles=req.n_vehicles,
            demand_range=(req.demand_min, req.demand_max),
            horizon=req.horizon,
            window_length_range=(req.window_length_min, req.window_length_max),
            service_time=req.service_time, seed=req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vrp_id = store.put_vrp(req.network_id, problem)

    customers = [
        CustomerOut(node_id=c.node_id, demand=c.demand, ready_time=c.ready_time,
                    due_time=c.due_time, service_time=c.service_time)
        for c in problem.customers
    ]

    return VRPInstanceResponse(
        vrp_id=vrp_id, network_id=req.network_id, depot=problem.depot,
        n_vehicles=problem.n_vehicles, vehicle_capacity=problem.vehicle_capacity,
        total_demand=problem.total_demand(), customers=customers,
    )


@router.get("/vrp/{vrp_id}", response_model=VRPInstanceResponse)
def get_vrp_instance(vrp_id: str):
    try:
        problem = store.get_vrp(vrp_id)
        network_id = store.get_vrp_network_id(vrp_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    customers = [
        CustomerOut(node_id=c.node_id, demand=c.demand, ready_time=c.ready_time,
                    due_time=c.due_time, service_time=c.service_time)
        for c in problem.customers
    ]
    return VRPInstanceResponse(
        vrp_id=vrp_id, network_id=network_id, depot=problem.depot,
        n_vehicles=problem.n_vehicles, vehicle_capacity=problem.vehicle_capacity,
        total_demand=problem.total_demand(), customers=customers,
    )


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _routes_with_loads(problem: VRPProblem, routes):
    demand_lookup = {c.node_id: c.demand for c in problem.customers}
    G = problem.net.graph
    out = []
    for i, route in enumerate(routes):
        if not route:
            continue
        load = sum(demand_lookup[n] for n in route)

        # Build the full road-network path: depot -> c1 -> c2 -> ... -> depot,
        # stitching together each leg's precomputed shortest path so the
        # frontend can draw the actual roads driven, not straight lines.
        full_path: list = []
        stops = [problem.depot] + route + [problem.depot]
        for a, b in zip(stops[:-1], stops[1:]):
            leg = problem.path_between(a, b)
            if full_path and leg and full_path[-1] == leg[0]:
                full_path.extend(leg[1:])
            else:
                full_path.extend(leg)

        # Calculate route-level congestion metrics along full_path
        route_base_time = 0.0
        route_travel_time = 0.0
        cong_factors = []
        for u, v in zip(full_path[:-1], full_path[1:]):
            if G.has_edge(u, v):
                edge_data = G[u][v]
                b_time = edge_data.get("base_time", 0.0)
                c_fac = edge_data.get("congestion_factor", 1.0)
                route_base_time += b_time
                route_travel_time += b_time * c_fac
                cong_factors.append(c_fac)

        route_delay = max(0.0, route_travel_time - route_base_time)
        avg_cong = float(np.mean(cong_factors)) if cong_factors else 1.0

        out.append(RouteOut(
            vehicle_id=i,
            customer_sequence=route,
            load=load,
            full_path=full_path,
            congestion_delay_min=round(route_delay, 2),
            avg_congestion=round(avg_cong, 2),
        ))
    return out


def _build_solve_response(problem: VRPProblem, sol, curve, n_eval, name, runtime_ms):
    routes_out = _routes_with_loads(problem, sol.routes)
    G = problem.net.graph

    total_base_time = 0.0
    total_effective_time = 0.0
    all_cong = []

    for r in routes_out:
        for u, v in zip(r.full_path[:-1], r.full_path[1:]):
            if G.has_edge(u, v):
                b_time = G[u][v].get("base_time", 0.0)
                c_fac = G[u][v].get("congestion_factor", 1.0)
                total_base_time += b_time
                total_effective_time += b_time * c_fac
                all_cong.append(c_fac)

    total_delay = max(0.0, total_effective_time - total_base_time)
    avg_cong = float(np.mean(all_cong)) if all_cong else 1.0

    return VRPSolveResponse(
        algorithm=name,
        routes=routes_out,
        total_distance=round(sol.total_distance, 2),
        total_time=round(sol.total_time, 2),
        capacity_violation=round(sol.capacity_violation, 2),
        time_window_violation=round(sol.time_window_violation, 2),
        feasible=sol.feasible,
        fitness=round(sol.fitness, 2),
        runtime_ms=round(runtime_ms, 2),
        n_evaluations=n_eval,
        convergence_curve=curve,
        congestion_delay_min=round(total_delay, 2),
        avg_congestion=round(avg_cong, 2),
        base_time_min=round(total_base_time, 2),
    )


def _solve_one(problem: VRPProblem, algorithm: str, n_particles: int,
                max_iter: int, seed: int, use_local_search: bool):
    """Runs one algorithm and returns a normalized (solution, runtime_ms, n_eval, curve, algo_name)."""
    t0 = time.perf_counter()

    if algorithm == "qpso":
        opt = QPSOVRPOptimizer(problem, n_particles=n_particles, max_iter=max_iter,
                                seed=seed, use_local_search=use_local_search)
        result = opt.optimize()
        sol, curve, n_eval, name = result.best_solution, result.convergence_curve, result.n_evaluations, "QPSO (Quantum-Inspired PSO)"

    elif algorithm == "ga":
        r = run_ga_vrp(problem, pop_size=n_particles, max_iter=max_iter, seed=seed)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm

    elif algorithm == "sa":
        r = run_sa_vrp(problem, max_iter=max_iter * 20, seed=seed)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm

    elif algorithm == "standard_pso":
        r = run_standard_pso_vrp(problem, n_particles=n_particles, max_iter=max_iter, seed=seed)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm

    elif algorithm == "greedy":
        r = run_greedy_nn_vrp(problem)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm

    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm '{algorithm}'")

    runtime_ms = (time.perf_counter() - t0) * 1000
    return sol, curve, n_eval, name, runtime_ms


@router.post("/vrp/solve", response_model=VRPSolveResponse)
def solve_vrp(req: VRPSolveRequest):
    try:
        problem = store.get_vrp(req.vrp_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    sol, curve, n_eval, name, runtime_ms = _solve_one(
        problem, req.algorithm, req.n_particles, req.max_iter, req.seed, req.use_local_search
    )

    if sol is None:
        raise HTTPException(status_code=500, detail="Solver failed to produce a solution")

    return _build_solve_response(problem, sol, curve, n_eval, name, runtime_ms)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

ALL_ALGORITHMS = ["greedy", "qpso", "ga", "sa", "standard_pso"]


@router.post("/benchmark/run", response_model=BenchmarkResponse)
def run_benchmark(req: BenchmarkRequest):
    try:
        problem = store.get_vrp(req.vrp_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    algorithms = req.algorithms or ALL_ALGORITHMS
    results = []

    for algo in algorithms:
        sol, curve, n_eval, name, runtime_ms = _solve_one(
            problem, algo, n_particles=50, max_iter=req.max_iter,
            seed=req.seed, use_local_search=True,
        )
        if sol is None:
            continue

        resp = _build_solve_response(problem, sol, curve, n_eval, name, runtime_ms)
        results.append(BenchmarkAlgoResult(
            algorithm=name, fitness=resp.fitness, distance=resp.total_distance,
            time=resp.total_time, feasible=resp.feasible, runtime_ms=resp.runtime_ms,
            n_evaluations=n_eval, convergence_curve=curve,
            congestion_delay_min=resp.congestion_delay_min,
            avg_congestion=resp.avg_congestion,
        ))

    return BenchmarkResponse(vrp_id=req.vrp_id, results=results)


# ---------------------------------------------------------------------------
# Before / After Comparison
# ---------------------------------------------------------------------------

@router.post("/vrp/compare", response_model=VRPCompareResponse)
def compare_vrp(req: VRPCompareRequest):
    try:
        problem = store.get_vrp(req.vrp_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Solve baseline
    base_sol, base_curve, base_neval, base_name, base_rt = _solve_one(
        problem, req.baseline_algo, n_particles=req.n_particles,
        max_iter=req.max_iter, seed=req.seed, use_local_search=False
    )
    if base_sol is None:
        raise HTTPException(status_code=500, detail="Baseline solver failed")
    baseline_resp = _build_solve_response(problem, base_sol, base_curve, base_neval, base_name, base_rt)

    # Solve optimized
    opt_sol, opt_curve, opt_neval, opt_name, opt_rt = _solve_one(
        problem, req.optimized_algo, n_particles=req.n_particles,
        max_iter=req.max_iter, seed=req.seed, use_local_search=req.use_local_search
    )
    if opt_sol is None:
        raise HTTPException(status_code=500, detail="Optimized solver failed")
    optimized_resp = _build_solve_response(problem, opt_sol, opt_curve, opt_neval, opt_name, opt_rt)

    # Compute deltas
    base_time = baseline_resp.total_time
    opt_time = optimized_resp.total_time
    time_saved_min = max(0.0, base_time - opt_time)
    time_saved_pct = (time_saved_min / base_time * 100.0) if base_time > 0 else 0.0

    base_dist = baseline_resp.total_distance
    opt_dist = optimized_resp.total_distance
    dist_saved_km = max(0.0, base_dist - opt_dist)
    dist_saved_pct = (dist_saved_km / base_dist * 100.0) if base_dist > 0 else 0.0

    base_delay = baseline_resp.congestion_delay_min
    opt_delay = optimized_resp.congestion_delay_min
    delay_saved_min = max(0.0, base_delay - opt_delay)
    delay_saved_pct = (delay_saved_min / base_delay * 100.0) if base_delay > 0 else 0.0

    return VRPCompareResponse(
        vrp_id=req.vrp_id,
        baseline=baseline_resp,
        optimized=optimized_resp,
        time_saved_pct=round(time_saved_pct, 1),
        congestion_avoided_pct=round(delay_saved_pct, 1),
        distance_saved_pct=round(dist_saved_pct, 1),
        time_saved_min=round(time_saved_min, 1),
        delay_saved_min=round(delay_saved_min, 1),
        distance_saved_km=round(dist_saved_km, 1),
    )


# ---------------------------------------------------------------------------
# Dynamic Traffic & Mid-Route Re-Optimization
# ---------------------------------------------------------------------------

@router.post("/traffic/incident", response_model=TrafficIncidentResponse)
def create_traffic_incident(req: TrafficIncidentRequest):
    try:
        net = store.get_network(req.network_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not net.graph.has_edge(req.u, req.v):
        raise HTTPException(status_code=400, detail=f"Edge ({req.u}, {req.v}) does not exist in network '{req.network_id}'")

    net.apply_incident(req.u, req.v, req.factor, start_time=req.start_time, duration_min=req.duration_min)

    return TrafficIncidentResponse(
        network_id=req.network_id,
        u=req.u, v=req.v, factor=req.factor,
        message=f"Traffic incident applied to edge ({req.u}, {req.v}) with {req.factor:.1f}x congestion multiplier"
    )


@router.post("/vrp/solve-dynamic", response_model=DynamicSolveResponse)
def solve_dynamic_vrp(req: DynamicSolveRequest):
    try:
        problem = store.get_vrp(req.vrp_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        res = simulate_dynamic_reroute(
            problem=problem,
            incident_u=req.incident_u,
            incident_v=req.incident_v,
            incident_factor=req.incident_factor,
            trigger_time_min=req.trigger_time_min,
            algorithm=req.algorithm,
            n_particles=req.n_particles,
            max_iter=req.max_iter,
            seed=req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid dynamic simulation parameters: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Dynamic VRP simulation failed: {e}")

    # The simulation runs on an isolated copy so the stored instance keeps its
    # original traffic. Congestion-aware metrics and full_path reconstruction
    # must therefore come from that copy, not from `problem`.
    sim_problem = res.simulated_problem or problem

    init_resp = _build_solve_response(problem, res.initial_solution, [], 0, "Initial Plan (t=0)", 0.0)
    static_resp = _build_solve_response(sim_problem, res.static_solution, [], 0, "Static Execution (Blind to Traffic)", 0.0)
    dynamic_resp = _build_solve_response(sim_problem, res.dynamic_solution, [], 0, f"Dynamic QPSO (Re-routed at t={req.trigger_time_min:.0f}m)", 0.0)

    return DynamicSolveResponse(
        vrp_id=req.vrp_id,
        trigger_time_min=res.trigger_time_min,
        incident=res.incident,
        initial_solution=init_resp,
        static_affected_solution=static_resp,
        dynamic_rerouted_solution=dynamic_resp,
        time_saved_min=res.time_saved_min,
        time_saved_pct=res.time_saved_pct,
        delay_avoided_min=res.delay_avoided_min,
        delay_avoided_pct=res.delay_avoided_pct,
        tw_violations_avoided=res.tw_violations_avoided,
        served_customer_ids=res.served_customer_ids,
        unserved_customer_ids=res.unserved_customer_ids,
    )


# In-Dashboard AI Assistant (Issue #32)
# ---------------------------------------------------------------------------

@router.post("/assistant/chat", response_model=AssistantChatResponse)
def assistant_chat(req: AssistantChatRequest):
    """Answers judge/user questions about current solve results, QPSO, and map."""
    return assistant_engine.chat(req)

