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
from app.core import store
from app.models.schemas import (
    NetworkGenerateRequest, NetworkResponse, NodeOut, EdgeOut, OSMNetworkRequest,
    VRPGenerateRequest, VRPInstanceResponse, CustomerOut,
    VRPSolveRequest, VRPSolveResponse, RouteOut,
    BenchmarkRequest, BenchmarkResponse, BenchmarkAlgoResult,
)

router = APIRouter(prefix="/api")


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

    nodes = [NodeOut(id=n, x=d["x"], y=d["y"]) for n, d in net.graph.nodes(data=True)]
    edges = [
        EdgeOut(u=u, v=v, distance=d["distance"], base_time=d["base_time"],
                congestion_factor=d["congestion_factor"])
        for u, v, d in net.graph.edges(data=True)
    ]

    return NetworkResponse(
        network_id=network_id, num_nodes=net.num_nodes(),
        num_edges=net.graph.number_of_edges(), is_geo=False, nodes=nodes, edges=edges,
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

    nodes = [NodeOut(id=n, x=d["x"], y=d["y"]) for n, d in net.graph.nodes(data=True)]
    edges = [
        EdgeOut(u=u, v=v, distance=d["distance"], base_time=d["base_time"],
                congestion_factor=d["congestion_factor"])
        for u, v, d in net.graph.edges(data=True)
    ]

    return NetworkResponse(
        network_id=network_id, num_nodes=net.num_nodes(),
        num_edges=net.graph.number_of_edges(), is_geo=True, nodes=nodes, edges=edges,
    )


@router.get("/network/{network_id}", response_model=NetworkResponse)
def get_network(network_id: str):
    try:
        net = store.get_network(network_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    is_geo = store.is_geo_network(network_id)
    nodes = [NodeOut(id=n, x=d["x"], y=d["y"]) for n, d in net.graph.nodes(data=True)]
    edges = [
        EdgeOut(u=u, v=v, distance=d["distance"], base_time=d["base_time"],
                congestion_factor=d["congestion_factor"])
        for u, v, d in net.graph.edges(data=True)
    ]
    return NetworkResponse(
        network_id=network_id, num_nodes=net.num_nodes(),
        num_edges=net.graph.number_of_edges(), is_geo=is_geo, nodes=nodes, edges=edges,
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

        out.append(RouteOut(vehicle_id=i, customer_sequence=route, load=load, full_path=full_path))
    return out


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

    return VRPSolveResponse(
        algorithm=name,
        routes=_routes_with_loads(problem, sol.routes),
        total_distance=sol.total_distance,
        total_time=sol.total_time,
        capacity_violation=sol.capacity_violation,
        time_window_violation=sol.time_window_violation,
        feasible=sol.feasible,
        fitness=sol.fitness,
        runtime_ms=runtime_ms,
        n_evaluations=n_eval,
        convergence_curve=curve,
    )


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
        results.append(BenchmarkAlgoResult(
            algorithm=name, fitness=sol.fitness, distance=sol.total_distance,
            time=sol.total_time, feasible=sol.feasible, runtime_ms=runtime_ms,
            n_evaluations=n_eval, convergence_curve=curve,
        ))

    return BenchmarkResponse(vrp_id=req.vrp_id, results=results)
