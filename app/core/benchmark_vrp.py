"""
benchmark_vrp.py
------------------
Systematic benchmarking for the CVRPTW formulation: QPSO vs GA vs SA vs
standard PSO vs a greedy construction heuristic. Produces:
    - comparison table (fitness, feasibility, runtime, evaluations)
    - convergence curve overlay
    - scalability test across increasing numbers of customers
    - statistical robustness check (multiple seeds per algorithm)
"""

from __future__ import annotations
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp, VRPProblem
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.classical_baselines_vrp import (
    run_ga_vrp, run_sa_vrp, run_standard_pso_vrp, run_greedy_nn_vrp
)
from app.core.exact_vrp import (
    EXACT_MAX_CUSTOMERS, ExactSolverTooLarge, optimality_gap, solve_vrp_exact,
)


def run_full_vrp_benchmark(problem: VRPProblem, max_iter: int = 200, seed: int = 1,
                           include_exact: bool = True,
                           exact_max_customers: int = EXACT_MAX_CUSTOMERS) -> Dict[str, Any]:
    """
    Every algorithm on one instance.

    On instances small enough to solve exactly, the proven optimum is included
    as a baseline. Without it the table only ranks heuristics against each
    other, which cannot answer how close to optimal the winner actually got.
    Larger instances omit the row: the exact solver is exponential, so running
    it there would hang rather than inform.
    """
    results = {}

    if include_exact and len(problem.customers) <= exact_max_customers:
        try:
            results["Exact"] = solve_vrp_exact(problem, max_customers=exact_max_customers)
        except ExactSolverTooLarge:
            pass

    results["Greedy NN"] = run_greedy_nn_vrp(problem)

    t0 = time.perf_counter()
    qpso = QPSOVRPOptimizer(problem, n_particles=50, max_iter=max_iter, seed=seed)
    qres = qpso.optimize()
    results["QPSO"] = {
        "algorithm": "QPSO (Quantum-Inspired PSO)",
        "best_solution": qres.best_solution,
        "best_fitness": qres.best_fitness,
        "runtime_sec": time.perf_counter() - t0,
        "n_evaluations": qres.n_evaluations,
        "convergence_curve": qres.convergence_curve,
    }

    results["GA"] = run_ga_vrp(problem, pop_size=50, max_iter=max_iter, seed=seed)
    results["SA"] = run_sa_vrp(problem, max_iter=max_iter * 20, seed=seed)
    results["Standard PSO"] = run_standard_pso_vrp(problem, n_particles=50, max_iter=max_iter, seed=seed)

    return results


def print_vrp_comparison_table(results: Dict[str, Any]):
    # The gap column only means anything when the optimum is known, so it
    # appears only on instances the exact solver could finish. A blank column
    # would read as "no gap", which is the opposite of "not measured".
    optimum = None
    if "Exact" in results:
        exact = results["Exact"]
        optimum = exact["best_fitness"] if isinstance(exact, dict) else exact.best_fitness

    gap_header = f" {'Gap vs opt':>11}" if optimum is not None else ""
    print(f"\n{'Algorithm':<28} {'Fitness':>10} {'Distance':>10} {'Time':>10} {'Feasible':>9} {'Runtime(ms)':>13} {'Evals':>8}{gap_header}")
    print("-" * (95 + len(gap_header)))
    for key, r in results.items():
        if isinstance(r, dict):
            algo, fit, sol, rt, ev = r["algorithm"], r["best_fitness"], r["best_solution"], r["runtime_sec"] * 1000, r["n_evaluations"]
        else:
            algo, fit, sol, rt, ev = r.algorithm, r.best_fitness, r.best_solution, r.runtime_sec * 1000, r.n_evaluations
        dist = sol.total_distance if sol else float("nan")
        tme = sol.total_time if sol else float("nan")
        feas = sol.feasible if sol else False
        gap = ""
        if optimum is not None:
            measured = "--" if key == "Exact" else f"{optimality_gap(fit, optimum):+.2f}%"
            gap = f" {measured:>11}"
        print(f"{algo:<28} {fit:>10.2f} {dist:>10.2f} {tme:>10.2f} {str(feas):>9} {rt:>13.2f} {ev:>8d}{gap}")


def plot_vrp_convergence(results: Dict[str, Any], save_path: str):
    plt.figure(figsize=(9, 6))
    for key in ["QPSO", "GA", "SA", "Standard PSO"]:
        r = results[key]
        curve = r["convergence_curve"] if isinstance(r, dict) else r.convergence_curve
        algo_name = r["algorithm"] if isinstance(r, dict) else r.algorithm
        x = np.linspace(0, 1, len(curve))
        plt.plot(x, curve, label=algo_name, linewidth=2)

    # greedy is a single-point reference line
    greedy_fit = results["Greedy NN"].best_fitness
    plt.axhline(greedy_fit, color="gray", linestyle="--", alpha=0.6, label="Greedy NN (no search)")

    plt.xlabel("Normalized iteration progress")
    plt.ylabel("Best fitness (distance + time + violation penalties)")
    plt.title("CVRPTW Convergence: QPSO vs Classical Metaheuristics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Convergence plot saved to: {save_path}")


def _scaled_budget(d: int, base_particles: int = 50, base_iter: int = 120,
                    particles_per_dim: float = 3.0, iter_per_dim: float = 3.0) -> tuple:
    """
    Scale swarm size and iteration count with problem dimensionality (d = n_customers)
    so every algorithm gets a search budget proportional to how hard the problem is,
    rather than a fixed budget that starves larger instances. This is standard
    practice when benchmarking metaheuristics across problem sizes -- comparing
    algorithms under a FIXED budget as dimensionality grows unfairly penalizes
    whichever algorithm needs more evaluations to cover the larger space.
    """
    n_particles = max(base_particles, int(particles_per_dim * d))
    max_iter = max(base_iter, int(iter_per_dim * d))
    return n_particles, max_iter


def run_vrp_scalability_test(
    customer_counts: List[int] = [10, 20, 40, 60],
    seed: int = 1,
    scale_budget: bool = True,
) -> Dict[int, Dict[str, Any]]:
    scalability = {}
    net = generate_synthetic_city_graph(n_nodes=150, seed=seed)  # big enough graph for largest customer count

    for n_cust in customer_counts:
        vrp = generate_synthetic_vrp(net, n_customers=n_cust, depot=0, vehicle_capacity=80, seed=seed)
        row = {}

        if scale_budget:
            n_particles, max_iter = _scaled_budget(n_cust)
        else:
            n_particles, max_iter = 50, 120

        t0 = time.perf_counter()
        qpso = QPSOVRPOptimizer(vrp, n_particles=n_particles, max_iter=max_iter, seed=seed)
        qres = qpso.optimize()
        row["QPSO"] = {"fitness": qres.best_fitness, "runtime": time.perf_counter() - t0,
                        "feasible": qres.best_solution.feasible if qres.best_solution else False}

        t0 = time.perf_counter()
        ga = run_ga_vrp(vrp, pop_size=n_particles, max_iter=max_iter, seed=seed)
        row["GA"] = {"fitness": ga.best_fitness, "runtime": time.perf_counter() - t0,
                      "feasible": ga.best_solution.feasible if ga.best_solution else False}

        t0 = time.perf_counter()
        pso = run_standard_pso_vrp(vrp, n_particles=n_particles, max_iter=max_iter, seed=seed)
        row["Standard PSO"] = {"fitness": pso.best_fitness, "runtime": time.perf_counter() - t0,
                                "feasible": pso.best_solution.feasible if pso.best_solution else False}

        scalability[n_cust] = row
        print(f"n_customers={n_cust:4d} (particles={n_particles}, iters={max_iter}) | " +
              " | ".join(f"{k}: fit={v['fitness']:.1f} t={v['runtime']*1000:.0f}ms feas={v['feasible']}"
                         for k, v in row.items()))

    return scalability


def plot_vrp_scalability(scalability: Dict[int, Dict[str, Any]], save_path: str):
    sizes = sorted(scalability.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for algo in ["QPSO", "GA", "Standard PSO"]:
        runtimes = [scalability[n][algo]["runtime"] * 1000 for n in sizes]
        fits = [scalability[n][algo]["fitness"] for n in sizes]
        ax1.plot(sizes, runtimes, marker="o", label=algo)
        ax2.plot(sizes, fits, marker="o", label=algo)

    ax1.set_xlabel("Number of customers"); ax1.set_ylabel("Runtime (ms)")
    ax1.set_title("VRP Scalability: Runtime vs Problem Size")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Number of customers"); ax2.set_ylabel("Best fitness found")
    ax2.set_title("VRP Scalability: Solution Quality vs Problem Size")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scalability plot saved to: {save_path}")


def run_robustness_check(problem: VRPProblem, n_seeds: int = 5, max_iter: int = 150) -> Dict[str, Dict[str, float]]:
    """Run each algorithm across multiple random seeds to report mean +/- std
    (single-seed comparisons can be misleading due to randomness)."""
    algo_fns = {
        "QPSO": lambda seed: QPSOVRPOptimizer(problem, n_particles=50, max_iter=max_iter, seed=seed).optimize().best_fitness,
        "GA": lambda seed: run_ga_vrp(problem, max_iter=max_iter, seed=seed).best_fitness,
        "SA": lambda seed: run_sa_vrp(problem, max_iter=max_iter * 20, seed=seed).best_fitness,
        "Standard PSO": lambda seed: run_standard_pso_vrp(problem, max_iter=max_iter, seed=seed).best_fitness,
    }

    summary = {}
    for name, fn in algo_fns.items():
        fits = [fn(seed) for seed in range(n_seeds)]
        summary[name] = {"mean": float(np.mean(fits)), "std": float(np.std(fits)),
                          "best": float(np.min(fits)), "worst": float(np.max(fits))}
        print(f"{name:15s} | mean={summary[name]['mean']:.2f} +/- {summary[name]['std']:.2f} "
              f"| best={summary[name]['best']:.2f} | worst={summary[name]['worst']:.2f}")

    return summary


def run_dynamic_traffic_benchmark(vrp: VRPProblem, incident_factors: List[float] = [2.0, 3.5, 5.0]) -> Dict[float, Any]:
    """Benchmark Static vs Dynamic QPSO re-optimization across varying traffic incident severities."""
    from app.core.dynamic_vrp import simulate_dynamic_reroute

    print(f"\n{'Incident Severity':<18} {'Static Time':>12} {'Dynamic Time':>13} {'Time Saved':>12} {'Delay Avoided %':>17}")
    print("-" * 75)

    dyn_results = {}
    for factor in incident_factors:
        res = simulate_dynamic_reroute(
            problem=vrp, incident_factor=factor, trigger_time_min=60.0,
            algorithm="qpso", n_particles=50, max_iter=150, seed=1
        )
        dyn_results[factor] = res
        print(f"{factor:.1f}x Congestion {res.static_solution.total_time:>12.1f}m {res.dynamic_solution.total_time:>13.1f}m "
              f"{res.time_saved_min:>10.1f}m ({res.time_saved_pct:.1f}%) {res.delay_avoided_pct:>16.1f}%")

    return dyn_results


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print("=" * 95)
    print("VRP BENCHMARK: Single instance, all algorithms")
    print("=" * 95)
    net = generate_synthetic_city_graph(n_nodes=40, seed=7)
    vrp = generate_synthetic_vrp(net, n_customers=18, depot=0, vehicle_capacity=80, seed=3)
    results = run_full_vrp_benchmark(vrp, max_iter=200, seed=1)
    print_vrp_comparison_table(results)
    plot_vrp_convergence(results, save_path="data/vrp_convergence.png")

    print("\n" + "=" * 95)
    print("DYNAMIC TRAFFIC BENCHMARK (Static vs Dynamic QPSO Rerouting)")
    print("=" * 95)
    run_dynamic_traffic_benchmark(vrp, incident_factors=[2.0, 3.5, 5.0])

    print("\n" + "=" * 95)
    print("ROBUSTNESS CHECK (5 seeds per algorithm)")
    print("=" * 95)
    run_robustness_check(vrp, n_seeds=5, max_iter=150)

    print("\n" + "=" * 95)
    print("VRP SCALABILITY TEST (fixed budget -- for comparison)")
    print("=" * 95)
    scal_fixed = run_vrp_scalability_test(customer_counts=[10, 20, 40, 60], seed=1, scale_budget=False)
    plot_vrp_scalability(scal_fixed, save_path="data/vrp_scalability_fixed_budget.png")

    print("\n" + "=" * 95)
    print("VRP SCALABILITY TEST (scaled budget -- fair per-dimension search effort)")
    print("=" * 95)
    scal_scaled = run_vrp_scalability_test(customer_counts=[10, 20, 40, 60], seed=1, scale_budget=True)
    plot_vrp_scalability(scal_scaled, save_path="data/vrp_scalability_scaled_budget.png")

def run_real_city_demo_benchmark(place: str = "New Delhi, India", max_nodes: int = 1000):
    from app.core.osm_network import load_osm_network
    print("\n" + "=" * 95)
    print(f"REAL CITY DEMO BENCHMARK: {place}")
    print("=" * 95)
    try:
        net = load_osm_network(place=place, max_nodes=max_nodes)
        vrp = generate_synthetic_vrp(net, n_customers=16, depot=0, vehicle_capacity=80, seed=42)
        results = run_full_vrp_benchmark(vrp, max_iter=150, seed=1)
        print_vrp_comparison_table(results)
    except Exception as e:
        print(f"Could not run real city benchmark: {e}")

import multiprocessing
import traceback

def _worker_solve(problem, algorithm, seed, out_q):
    try:
        t0 = time.perf_counter()
        if algorithm == "qpso_local_search":
            opt = QPSOVRPOptimizer(problem, n_particles=50, max_iter=200, seed=seed, use_local_search=True)
            res = opt.optimize()
            sol = res.best_solution
            fit = res.best_fitness
        elif algorithm == "standard_pso":
            res = run_standard_pso_vrp(problem, n_particles=50, max_iter=200, seed=seed)
            sol = res.best_solution
            fit = res.best_fitness
        else:
            raise ValueError(f"Unknown algorithm {algorithm}")
            
        runtime = time.perf_counter() - t0
        
        # Serialize routes (just as a dict structure so it can be JSON dumped)
        routes_dump = []
        if sol and sol.routes:
            for route in sol.routes:
                routes_dump.append(route)
            
        out_q.put({
            "status": "success",
            "runtime_ms": runtime * 1000,
            "fitness": fit,
            "feasible": sol.feasible if sol else False,
            "routes": routes_dump
        })
    except Exception as e:
        traceback.print_exc()
        out_q.put({"status": "error", "error": str(e)})

def run_stress_test_at_scale(
    n_customers_list: list[int] = [100, 200],
    algorithms: list[str] = ["qpso_local_search", "standard_pso"],
    network_source: str = "real_city",
    network_id: str | None = None,
    time_budget_seconds: float = 300.0,
    n_seeds: int = 3,
) -> dict:
    import json
    from app.core.osm_network import load_osm_network
    
    if network_source == "real_city":
        net = load_osm_network(place="New Delhi, India", max_nodes=500)
        actual_network_id = network_id or "real_city_delhi"
    else:
        net = generate_synthetic_city_graph(n_nodes=300, seed=1)
        actual_network_id = "synthetic"
        
    results = {}
    
    for n_customers in n_customers_list:
        results[str(n_customers)] = {}
        problem = generate_synthetic_vrp(net, n_customers=n_customers, depot=0, vehicle_capacity=80)
        
        for algo in algorithms:
            print(f"Running {algo} at {n_customers} customers...")
            best_res = None
            
            for seed in range(n_seeds):
                ctx = multiprocessing.get_context('spawn')
                out_q = ctx.Queue()
                p = ctx.Process(target=_worker_solve, args=(problem, algo, seed, out_q))
                p.start()
                p.join(timeout=time_budget_seconds)
                
                if p.is_alive():
                    print(f"Timeout reached for {algo} at seed {seed}")
                    p.terminate()
                    p.join()
                    res = {"status": "timeout", "runtime_ms": time_budget_seconds * 1000, "fitness": None, "feasible": False, "routes": []}
                else:
                    if not out_q.empty():
                        res = out_q.get()
                    else:
                        res = {"status": "crash", "runtime_ms": time_budget_seconds * 1000, "fitness": None, "feasible": False, "routes": []}
                
                if res["status"] == "success":
                    if best_res is None or res["fitness"] < best_res["fitness"]:
                        best_res = res
                        
            if best_res is None:
                best_res = {"status": "timeout", "runtime_ms": time_budget_seconds * 1000, "fitness": None, "feasible": False, "routes": []}
                
            results[str(n_customers)][algo] = {
                "fitness": best_res.get("fitness"),
                "runtime_ms": best_res.get("runtime_ms"),
                "feasible": best_res.get("feasible"),
                "routes": best_res.get("routes")
            }
            
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "network_id": actual_network_id,
        "results": results
    }

if __name__ == "__main__":
    # Also run the real city benchmark at the end
    run_real_city_demo_benchmark()
