"""
vrp_problem.py
---------------
Capacitated Vehicle Routing Problem with Time Windows (CVRPTW), built on top
of the TrafficNetwork road graph.

Why this instead of plain shortest-path:
    The problem statement explicitly targets "large-scale Vehicle Routing
    Problems (VRP)" -- an NP-hard combinatorial problem. Single-vehicle
    shortest-path is solved trivially (and exactly) by Dijkstra, so it can't
    demonstrate any advantage for a metaheuristic. CVRPTW, with multiple
    vehicles, capacity limits, and time windows, is genuinely NP-hard and is
    where quantum-inspired metaheuristics are meant to show their value.

Encoding (random-key style, shared by QPSO / GA / SA / standard PSO so
comparisons stay fair):
    chromosome = vector of length n_customers, each value in [0, K)
        vehicle_id       = floor(value)             -> which vehicle serves this customer
        priority_in_route = value - floor(value)      -> ordering within that vehicle's route

Decoding:
    1. Group customers by vehicle_id.
    2. Within each group, sort by priority_in_route ascending -> visiting order.
    3. Each vehicle's route: depot -> customers in order -> depot.
    4. Walk the route, accumulating distance/time, checking capacity and time
       windows; violations are captured as penalty terms (soft constraints),
       which lets the search explore through infeasible region and converge
       toward feasible optima -- standard practice for VRP metaheuristics.
"""

from __future__ import annotations
import math
import random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from app.core.graph_model import TrafficNetwork


@dataclass
class Customer:
    node_id: int
    demand: float
    ready_time: float     # earliest service can start (minutes)
    due_time: float       # latest service can start (minutes)
    service_time: float   # time spent servicing this customer (minutes)


@dataclass
class VRPProblem:
    net: TrafficNetwork
    depot: int
    customers: List[Customer]
    vehicle_capacity: float
    n_vehicles: int

    # Optional per-vehicle capacities, indexed by vehicle. Used for mid-route
    # re-planning, where each vehicle has already consumed part of its load and
    # so has less room left than a fresh vehicle. None (the default) means every
    # vehicle has the uniform `vehicle_capacity`.
    vehicle_capacities: Optional[List[float]] = None

    # Treat n_vehicles as a fleet that must all be sent out, rather than as a
    # ceiling. Off by default, because leaving a van parked is usually the
    # better plan and pretending otherwise would quietly make every published
    # benchmark figure worse. See the idle-vehicle penalty in evaluate_solution.
    require_all_vehicles: bool = False

    # Time-dependent travel times. When False (the default) the problem behaves
    # exactly as it always has: one travel-time matrix, no clock, so a road
    # costs the same at 09:00 and at 14:00. When True, a matrix is precomputed
    # per time bucket and the cost of a leg depends on when the vehicle leaves,
    # which is what makes rush hour something the optimiser routes around rather
    # than a label on a chart.
    #
    # Off by default on purpose: switching it on changes every travel time, and
    # therefore every previously published benchmark figure. It is opted into
    # explicitly so results are never silently redefined.
    time_dependent: bool = False
    bucket_minutes: float = 30.0
    horizon_minutes: float = 720.0

    # How the objective trades fleet time against distance driven. These live on
    # the problem rather than on each solver because they are a property of the
    # question being asked, not of the method used to answer it -- and because
    # every solver here reaches the objective through evaluate_solution, so one
    # place is enough for all of them to agree.
    #
    # The defaults are this project's own: a municipal fleet cares mostly about
    # finishing the round on time. A published benchmark may score something
    # else -- Solomon's CVRPTW set is judged on total distance alone -- and an
    # instance loaded from one sets these accordingly, so the optimiser is
    # answering the question the benchmark actually asks rather than being
    # marked against a different one.
    objective_w_time: float = 0.6
    objective_w_distance: float = 0.4

    # precomputed after __post_init__
    time_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    dist_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    path_matrix: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    # bucket index -> travel-time matrix for departures inside that bucket
    time_matrices: Dict[int, Dict[Tuple[int, int], float]] = field(default_factory=dict)
    node_index: Dict[int, int] = field(default_factory=dict)  # node_id -> position (0=depot,1..n=customers)
    all_nodes: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.all_nodes = [self.depot] + [c.node_id for c in self.customers]
        self.node_index = {n: i for i, n in enumerate(self.all_nodes)}
        self._precompute_matrices()
        if self.time_dependent:
            self._precompute_time_buckets()

    # ---- time-dependent travel times ----------------------------------

    def _bucket_count(self) -> int:
        return max(1, int(math.ceil(self.horizon_minutes / self.bucket_minutes)))

    def bucket_for(self, depart_at: float) -> int:
        """
        Which time bucket a departure falls in.

        Departures past the horizon are clamped to the last bucket rather than
        wrapping: a vehicle running late is still in evening traffic, not back
        in the morning.
        """
        if depart_at <= 0:
            return 0
        return min(int(depart_at // self.bucket_minutes), self._bucket_count() - 1)

    def _precompute_time_buckets(self):
        """
        One travel-time matrix per bucket, each priced at that bucket's midpoint.

        Bucketing rather than recomputing per departure keeps this tractable:
        the alternative is a Dijkstra per (source, departure time) pair, which
        the solver would pay for on every fitness evaluation.
        """
        G = self.net.graph
        for b in range(self._bucket_count()):
            mid = (b + 0.5) * self.bucket_minutes
            matrix: Dict[Tuple[int, int], float] = {}
            for src in self.all_nodes:
                times, _ = nx.single_source_dijkstra(
                    G, src,
                    weight=lambda u, v, d, _t=mid: self.net.travel_time(u, v, current_time=_t),
                )
                for dst in self.all_nodes:
                    matrix[(src, dst)] = 0.0 if dst == src else times.get(dst, float("inf"))
            self.time_matrices[b] = matrix

    def _precompute_matrices(self, current_time: Optional[float] = None):
        """
        For every node of interest (depot + customers), run Dijkstra to get
        shortest travel TIME to every other node of interest based on current network travel times.
        """
        G = self.net.graph

        for s in self.all_nodes:
            times, paths = nx.single_source_dijkstra(
                G, s, weight=lambda u, v, d: self.net.travel_time(u, v, current_time=current_time)
            )
            for t in self.all_nodes:
                if t == s:
                    self.time_matrix[(s, t)] = 0.0
                    self.dist_matrix[(s, t)] = 0.0
                    continue
                if t not in times:
                    # unreachable -- large penalty distance/time
                    self.time_matrix[(s, t)] = float("inf")
                    self.dist_matrix[(s, t)] = float("inf")
                    self.path_matrix[(s, t)] = []
                    continue
                self.time_matrix[(s, t)] = times[t]
                path = paths[t]
                dist = sum(G[u][v]["distance"] for u, v in zip(path[:-1], path[1:]))
                self.dist_matrix[(s, t)] = dist
                self.path_matrix[(s, t)] = path

    def recompute_matrices(self, current_time: Optional[float] = None):
        """Recompute time, distance, and path matrices after a network congestion update or traffic incident."""
        self._precompute_matrices(current_time=current_time)

    def path_between(self, a: int, b: int) -> List[int]:
        """Full sequence of road-network nodes (intermediate intersections
        included) from a to b along the shortest-time path. Used for
        visualization -- draws the real route rather than a straight line."""
        if a == b:
            return [a]
        return self.path_matrix.get((a, b), [a, b])

    def travel_time(self, a: int, b: int, depart_at: Optional[float] = None) -> float:
        """
        Travel time from a to b, optionally for a vehicle leaving at `depart_at`
        minutes into the operating day.

        With time-dependence off, or no departure time supplied, this is the
        static matrix lookup it always was.
        """
        if self.time_dependent and depart_at is not None and self.time_matrices:
            return self.time_matrices[self.bucket_for(depart_at)][(a, b)]
        return self.time_matrix[(a, b)]

    def travel_distance(self, a: int, b: int) -> float:
        return self.dist_matrix[(a, b)]

    def total_demand(self) -> float:
        return sum(c.demand for c in self.customers)

    def capacity_for(self, vehicle_index: int) -> float:
        """
        Capacity available to one vehicle. Falls back to the uniform
        `vehicle_capacity` unless per-vehicle capacities were supplied
        (see `vehicle_capacities`).
        """
        if self.vehicle_capacities is None:
            return self.vehicle_capacity
        if 0 <= vehicle_index < len(self.vehicle_capacities):
            return self.vehicle_capacities[vehicle_index]
        return self.vehicle_capacity


# ---------------------------------------------------------------------------
# Synthetic VRP instance generator
# ---------------------------------------------------------------------------

def generate_synthetic_vrp(
    net: TrafficNetwork,
    n_customers: int = 15,
    depot: int = 0,
    vehicle_capacity: float = 100.0,
    n_vehicles: Optional[int] = None,
    demand_range: Tuple[float, float] = (5, 20),
    horizon: float = 480.0,          # e.g. 8-hour operating window (minutes)
    window_length_range: Tuple[float, float] = (60, 180),
    service_time: float = 10.0,
    seed: int = 1,
    time_dependent: bool = False,
    bucket_minutes: float = 30.0,
    customer_nodes: Optional[List[int]] = None,
    require_all_vehicles: bool = False,
) -> VRPProblem:
    """
    Build a VRP instance on `net`.

    Customers are sampled at random unless `customer_nodes` names them, which is
    what the map-driven builder does: the engineer clicks the stops they want on
    the real road network, so the instance has to honour that exact set rather
    than draw its own. Demands and time windows are still generated from `seed`,
    keeping a hand-picked instance as reproducible as a sampled one.
    """
    rng = random.Random(seed)
    all_graph_nodes = list(net.graph.nodes())
    if depot not in all_graph_nodes:
        depot = all_graph_nodes[0]
    candidates = [n for n in all_graph_nodes if n != depot]

    if customer_nodes is not None:
        # De-duplicate while holding the click order, so the instance matches
        # what the engineer selected on screen.
        seen = set()
        chosen = []
        for nid in customer_nodes:
            if nid == depot or nid in seen:
                continue
            if nid not in net.graph:
                raise ValueError(f"Node {nid} is not in this road network.")
            seen.add(nid)
            chosen.append(nid)
        if not chosen:
            raise ValueError("Select at least one stop that is not the depot.")
    else:
        if n_customers > len(candidates):
            raise ValueError(f"Requested {n_customers} customers but graph only has "
                              f"{len(candidates)} non-depot nodes.")
        chosen = rng.sample(candidates, n_customers)

    customers = []
    for node_id in chosen:
        demand = rng.uniform(*demand_range)
        ready = rng.uniform(0, horizon * 0.6)
        window_len = rng.uniform(*window_length_range)
        due = min(ready + window_len, horizon)
        customers.append(Customer(
            node_id=node_id, demand=demand,
            ready_time=ready, due_time=due, service_time=service_time,
        ))

    total_demand = sum(c.demand for c in customers)
    if n_vehicles is None:
        # enough vehicles to cover total demand with ~30% slack, min 2
        n_vehicles = max(2, int(np.ceil(total_demand / vehicle_capacity * 1.3)))

    return VRPProblem(
        net=net, depot=depot, customers=customers,
        vehicle_capacity=vehicle_capacity, n_vehicles=n_vehicles,
        time_dependent=time_dependent,
        bucket_minutes=bucket_minutes,
        horizon_minutes=horizon,
        require_all_vehicles=require_all_vehicles,
    )


# ---------------------------------------------------------------------------
# Shared decode + fitness evaluation (used by QPSO, GA, SA, standard PSO)
# ---------------------------------------------------------------------------

@dataclass
class VRPSolution:
    routes: List[List[int]]          # each route = list of customer node_ids in visit order (no depot)
    total_distance: float
    total_time: float
    capacity_violation: float
    time_window_violation: float
    fitness: float
    feasible: bool


def decode_chromosome(problem: VRPProblem, chromosome: np.ndarray) -> List[List[int]]:
    """chromosome: array of length n_customers, values in [0, n_vehicles)."""
    n_vehicles = problem.n_vehicles
    vehicle_ids = np.clip(np.floor(chromosome).astype(int), 0, n_vehicles - 1)
    priorities = chromosome - np.floor(chromosome)

    routes: List[List[Tuple[int, float]]] = [[] for _ in range(n_vehicles)]
    for idx, customer in enumerate(problem.customers):
        v = vehicle_ids[idx]
        routes[v].append((customer.node_id, priorities[idx]))

    # sort each route by priority ascending -> visiting order
    ordered_routes = []
    for route in routes:
        route.sort(key=lambda t: t[1])
        ordered_routes.append([node_id for node_id, _ in route])

    return ordered_routes


def evaluate_solution(
    problem: VRPProblem,
    routes: List[List[int]],
    capacity_penalty_weight: float = 50.0,
    time_window_penalty_weight: float = 10.0,
    w_time: Optional[float] = None,
    w_distance: Optional[float] = None,
    idle_vehicle_penalty_weight: float = 200.0,
) -> VRPSolution:
    # Unset means "whatever this instance is scored on" (see VRPProblem), which
    # is how a benchmark instance gets judged on its own objective without every
    # solver having to be told about it. An explicit argument still wins.
    if w_time is None:
        w_time = getattr(problem, "objective_w_time", 0.6)
    if w_distance is None:
        w_distance = getattr(problem, "objective_w_distance", 0.4)

    customer_lookup = {c.node_id: c for c in problem.customers}
    total_distance = 0.0
    total_time = 0.0
    capacity_violation = 0.0
    time_window_violation = 0.0

    for v_idx, route in enumerate(routes):
        if not route:
            continue

        # capacity check (per-vehicle, so partially-loaded vehicles mid-route
        # are held to the capacity they actually have left)
        route_capacity = problem.capacity_for(v_idx)
        route_demand = sum(customer_lookup[n].demand for n in route)
        if route_demand > route_capacity:
            capacity_violation += (route_demand - route_capacity)

        # walk the route: depot -> c1 -> c2 -> ... -> depot
        current_node = problem.depot
        current_time = 0.0
        for node_id in route:
            # Priced at the moment the vehicle actually leaves, so a leg driven
            # through rush hour costs more than the same leg at midday.
            travel_t = problem.travel_time(current_node, node_id, depart_at=current_time)
            travel_d = problem.travel_distance(current_node, node_id)

            if not np.isfinite(travel_t):
                # unreachable node -- heavy penalty, skip further accumulation for this edge
                time_window_violation += 1000.0
                current_node = node_id
                continue

            arrival = current_time + travel_t
            cust = customer_lookup[node_id]

            if arrival < cust.ready_time:
                wait = cust.ready_time - arrival
                start_service = cust.ready_time
            else:
                wait = 0.0
                start_service = arrival

            if start_service > cust.due_time:
                time_window_violation += (start_service - cust.due_time)

            total_distance += travel_d
            total_time += travel_t + wait

            current_time = start_service + cust.service_time
            current_node = node_id

        # return to depot
        back_t = problem.travel_time(current_node, problem.depot, depart_at=current_time)
        back_d = problem.travel_distance(current_node, problem.depot)
        if np.isfinite(back_t):
            total_distance += back_d
            total_time += back_t

    # Spreading the same stops over more vans costs time, so left alone the
    # optimiser parks any van it does not need — asking for five and being
    # shown three is the correct answer to "how many do I need". When the
    # fleet size is a given rather than a ceiling (a depot with five drivers
    # rostered, who are paid either way), an idle van is the thing to avoid,
    # and this makes leaving one parked the expensive option instead.
    idle_vehicle_penalty = 0.0
    if getattr(problem, "require_all_vehicles", False):
        idle = sum(1 for route in routes if not route)
        idle_vehicle_penalty = idle * idle_vehicle_penalty_weight

    fitness = (
        w_time * total_time
        + w_distance * total_distance
        + capacity_penalty_weight * capacity_violation
        + time_window_penalty_weight * time_window_violation
        + idle_vehicle_penalty
    )

    feasible = (capacity_violation < 1e-6) and (time_window_violation < 1e-6)

    return VRPSolution(
        routes=routes,
        total_distance=total_distance,
        total_time=total_time,
        capacity_violation=capacity_violation,
        time_window_violation=time_window_violation,
        fitness=fitness,
        feasible=feasible,
    )


def evaluate_chromosome(problem: VRPProblem, chromosome: np.ndarray, **kwargs) -> VRPSolution:
    routes = decode_chromosome(problem, chromosome)
    return evaluate_solution(problem, routes, **kwargs)


if __name__ == "__main__":
    from app.core.graph_model import generate_synthetic_city_graph

    net = generate_synthetic_city_graph(n_nodes=40, seed=7)
    vrp = generate_synthetic_vrp(net, n_customers=12, depot=0, vehicle_capacity=80, seed=3)

    print(f"Customers: {len(vrp.customers)}, Vehicles available: {vrp.n_vehicles}, "
          f"Total demand: {vrp.total_demand():.1f}, Capacity/vehicle: {vrp.vehicle_capacity}")

    rng = np.random.default_rng(0)
    chromosome = rng.uniform(0, vrp.n_vehicles, size=len(vrp.customers))
    sol = evaluate_chromosome(vrp, chromosome)
    print("\nSample random solution:")
    for i, route in enumerate(sol.routes):
        if route:
            print(f"  Vehicle {i}: depot -> {' -> '.join(map(str, route))} -> depot")
    print(f"Total distance: {sol.total_distance:.2f} km, Total time: {sol.total_time:.2f} min")
    print(f"Capacity violation: {sol.capacity_violation:.2f}, TW violation: {sol.time_window_violation:.2f}")
    print(f"Fitness: {sol.fitness:.2f}, Feasible: {sol.feasible}")
