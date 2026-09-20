"""
build_benchmark_matrix.py
-------------------------
Runs all five solvers against every shipped city scenario once and writes the
results to `data/benchmarks/algorithm_matrix.json`, which the Executive Overview
reads.

Why precompute
==============
The matrix is 5 algorithms x 5 cities = 25 solves. Run live that is a minute or
more of a judge watching a spinner, on a free-tier host that sleeps between
requests, to produce numbers that do not change between runs anyway — every
instance is built from a fixed seed, so the answer is the same today as it was
yesterday. Running it here and committing the result makes the page instant and
keeps it working with no internet, for the same reason the road networks in
`data/osm_offline/` are committed.

This is a claim about our own solver's performance, so it has to be
reproducible: the config block is written into the output, and re-running this
script on an unchanged codebase reproduces the numbers exactly. Re-run it when
an algorithm changes, or the figures on the page become a historical artifact.

Usage
=====
    python scripts/build_benchmark_matrix.py
    python scripts/build_benchmark_matrix.py --max-iter 50      # quick check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.cached_network import available_networks, load_cached_network   # noqa: E402
from app.core.classical_baselines_vrp import (                                # noqa: E402
    run_ga_vrp, run_greedy_nn_vrp, run_sa_vrp, run_standard_pso_vrp,
)
from app.core.qpso_vrp import QPSOVRPOptimizer                                # noqa: E402
from app.core.vrp_problem import generate_synthetic_vrp                       # noqa: E402

# How many stops each scenario gets. Matches the curated areas the Executive
# Overview already lists, so the matrix describes the same instances a visitor
# can load and solve themselves rather than a private set of easier ones.
SCENARIO_STOPS = {
    "delhi_central": 14,
    "mumbai_bkc": 14,
    "bengaluru_koramangala": 16,
    "hyderabad_hitec": 14,
    "bareilly": 14,
}

ALGORITHMS = ["greedy", "qpso", "ga", "sa", "standard_pso"]


def solve_one(problem, algorithm: str, n_particles: int, max_iter: int, seed: int):
    """Mirrors the API's _solve_one so the committed numbers match a live run."""
    t0 = time.perf_counter()
    if algorithm == "qpso":
        r = QPSOVRPOptimizer(problem, n_particles=n_particles, max_iter=max_iter,
                             seed=seed, use_local_search=True).optimize()
        sol, curve, n_eval, name = (r.best_solution, r.convergence_curve,
                                    r.n_evaluations, "QPSO (Quantum-Inspired PSO)")
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
        raise ValueError(f"Unknown algorithm '{algorithm}'")
    return sol, curve, n_eval, name, (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark all solvers on every shipped city.")
    ap.add_argument("--n-particles", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--congestion-seed", type=int, default=42)
    ap.add_argument("--capacity", type=float, default=80.0)
    ap.add_argument("--horizon", type=float, default=480.0)
    ap.add_argument("--out", default="data/benchmarks/algorithm_matrix.json")
    args = ap.parse_args()

    catalogue = {n["name"]: n for n in available_networks()}
    names = [n for n in SCENARIO_STOPS if n in catalogue]
    if not names:
        print("No cached networks found in data/networks/.")
        return 1

    scenarios = []
    for name in names:
        label = catalogue[name]["label"]
        stops = SCENARIO_STOPS[name]
        print(f"\n{label}  ({catalogue[name]['nodes']} junctions, {stops} stops)")

        net = load_cached_network(name, congestion_seed=args.congestion_seed)
        problem = generate_synthetic_vrp(
            net, n_customers=stops, depot=0,
            vehicle_capacity=args.capacity,
            demand_range=(5.0, 20.0), horizon=args.horizon,
            window_length_range=(60.0, 180.0), service_time=10.0, seed=args.seed,
        )

        results = []
        for algo in ALGORITHMS:
            sol, curve, n_eval, algo_label, runtime_ms = solve_one(
                problem, algo, args.n_particles, args.max_iter, args.seed)
            if sol is None:
                continue
            vans_used = sum(1 for r in sol.routes if r)
            results.append({
                "algorithm": algo,
                "label": algo_label,
                "fitness": round(sol.fitness, 4),
                "distance_km": round(sol.total_distance, 3),
                "time_min": round(sol.total_time, 2),
                "feasible": bool(sol.feasible),
                "runtime_ms": round(runtime_ms, 1),
                "n_evaluations": int(n_eval),
                "vans_used": vans_used,
                "convergence": [round(float(c), 4) for c in (curve or [])],
            })
            print(f"  {algo_label:34s} fitness {sol.fitness:10.2f}  "
                  f"{sol.total_distance:7.2f} km  {sol.total_time:8.1f} min  "
                  f"{runtime_ms:8.0f} ms  {'feasible' if sol.feasible else 'INFEASIBLE'}")

        # Feasible first, then fitness. Fitness already carries a penalty for
        # missing a delivery window, but the penalty is a weight rather than a
        # wall, so a plan that saves enough driving can come out "best" while
        # still turning up late — which is not a plan anyone can run. Ranking
        # on fitness alone would put that at the top of the table.
        best = min(results, key=lambda r: (not r["feasible"], r["fitness"]))["algorithm"] \
            if results else None
        feasible_count = sum(1 for r in results if r["feasible"])
        scenarios.append({
            "key": name,
            "label": label,
            "junctions": catalogue[name]["nodes"],
            "stops": stops,
            "vehicles": problem.n_vehicles,
            "best_algorithm": best,
            "feasible_count": feasible_count,
            "results": results,
        })

    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_particles": args.n_particles, "max_iter": args.max_iter,
            "seed": args.seed, "congestion_seed": args.congestion_seed,
            "vehicle_capacity": args.capacity, "horizon_min": args.horizon,
            "demand_range": [5.0, 20.0], "window_length_range": [60.0, 180.0],
            "service_time_min": 10.0,
        },
        "algorithms": ALGORITHMS,
        "scenarios": scenarios,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({size_kb:.0f} KB)")
    print(f"  {len(scenarios)} scenarios x {len(ALGORITHMS)} algorithms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
