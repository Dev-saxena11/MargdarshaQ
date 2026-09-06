"""
benchmark.py
-------------
Systematic performance benchmarking: QPSO vs classical metaheuristics (GA, SA,
standard PSO) vs exact methods (Dijkstra, A*).

Produces:
    - A comparison table (cost, runtime, evaluations)
    - Convergence curves overlay (matplotlib) for QPSO vs GA vs SA vs standard PSO
    - Scalability test across increasing graph sizes

This directly satisfies the problem statement's requirement:
    "systematic performance benchmarking" + "demonstrate scalability"
"""

from __future__ import annotations
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from app.core.graph_model import generate_synthetic_city_graph, TrafficNetwork
from app.core.qpso import QPSORouteOptimizer
from app.core.classical_baselines import (
    run_dijkstra, run_astar, run_ga, run_sa, run_standard_pso
)


def run_full_benchmark(
    net: TrafficNetwork, source: int, destination: int,
    max_iter: int = 150, seed: int = 1,
) -> Dict[str, Any]:
    """Run QPSO + all baselines on the same network/source/destination and
    return a structured comparison (used for both console tables and plots)."""

    results = {}

    # Exact methods (reference only -- optimize pure time, not blended fitness)
    results["Dijkstra"] = run_dijkstra(net, source, destination)
    results["A*"] = run_astar(net, source, destination)

    # Metaheuristics -- all optimize the SAME blended fitness function
    t0 = time.perf_counter()
    qpso = QPSORouteOptimizer(net, source, destination, n_particles=40,
                               max_iter=max_iter, seed=seed)
    qpso_result = qpso.optimize()
    qpso_runtime = time.perf_counter() - t0
    results["QPSO"] = {
        "algorithm": "QPSO (Quantum-Inspired PSO)",
        "best_route": qpso_result.best_route,
        "best_cost": qpso_result.best_cost,
        "runtime_sec": qpso_runtime,
        "n_evaluations": qpso_result.n_evaluations,
        "convergence_curve": qpso_result.convergence_curve,
        "feasible": qpso_result.feasible,
    }

    results["GA"] = run_ga(net, source, destination, max_iter=max_iter, seed=seed)
    results["SA"] = run_sa(net, source, destination, max_iter=max_iter * 20, seed=seed)  # SA needs more iters (single-solution search)
    results["Standard PSO"] = run_standard_pso(net, source, destination, max_iter=max_iter, seed=seed)

    return results


def print_comparison_table(results: Dict[str, Any]):
    print(f"\n{'Algorithm':<28} {'Cost':>12} {'Runtime(ms)':>14} {'Evaluations':>14} {'Feasible':>10}")
    print("-" * 82)
    for key, r in results.items():
        if isinstance(r, dict):
            algo, cost, rt, ev, feas = (r["algorithm"], r["best_cost"],
                                        r["runtime_sec"] * 1000, r["n_evaluations"], r["feasible"])
        else:
            algo, cost, rt, ev, feas = (r.algorithm, r.best_cost,
                                        r.runtime_sec * 1000, r.n_evaluations, r.feasible)
        print(f"{algo:<28} {cost:>12.3f} {rt:>14.2f} {ev:>14d} {str(feas):>10}")


def plot_convergence(results: Dict[str, Any], save_path: str = "convergence_comparison.png"):
    """Overlay convergence curves for QPSO / GA / SA / Standard PSO."""
    plt.figure(figsize=(9, 6))

    for key in ["QPSO", "GA", "SA", "Standard PSO"]:
        r = results[key]
        curve = r["convergence_curve"] if isinstance(r, dict) else r.convergence_curve
        algo_name = r["algorithm"] if isinstance(r, dict) else r.algorithm
        # normalize x-axis to iteration fraction so different iter counts overlay fairly
        x = np.linspace(0, 1, len(curve))
        plt.plot(x, curve, label=algo_name, linewidth=2)

    plt.xlabel("Normalized iteration progress")
    plt.ylabel("Best fitness (blended cost: time + distance + congestion)")
    plt.title("Convergence Comparison: QPSO vs Classical Metaheuristics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConvergence plot saved to: {save_path}")


def run_scalability_test(
    node_sizes: List[int] = [10, 20, 40, 80, 150],
    max_iter: int = 100, seed: int = 1,
) -> Dict[int, Dict[str, Any]]:
    """
    Run QPSO vs GA vs Standard PSO across increasing graph sizes to
    demonstrate scalability for smart-city logistics (per problem objective 4).
    """
    scalability_results = {}

    for n_nodes in node_sizes:
        net = generate_synthetic_city_graph(n_nodes=n_nodes, seed=seed)
        source, destination = 0, n_nodes - 1

        row = {}
        t0 = time.perf_counter()
        qpso = QPSORouteOptimizer(net, source, destination, n_particles=40,
                                   max_iter=max_iter, seed=seed)
        qres = qpso.optimize()
        row["QPSO"] = {"cost": qres.best_cost, "runtime": time.perf_counter() - t0,
                        "feasible": qres.feasible}

        t0 = time.perf_counter()
        ga = run_ga(net, source, destination, max_iter=max_iter, seed=seed)
        row["GA"] = {"cost": ga.best_cost, "runtime": time.perf_counter() - t0,
                      "feasible": ga.feasible}

        t0 = time.perf_counter()
        pso = run_standard_pso(net, source, destination, max_iter=max_iter, seed=seed)
        row["Standard PSO"] = {"cost": pso.best_cost, "runtime": time.perf_counter() - t0,
                                "feasible": pso.feasible}

        scalability_results[n_nodes] = row
        print(f"n_nodes={n_nodes:4d} | " +
              " | ".join(f"{k}: cost={v['cost']:.2f} t={v['runtime']*1000:.0f}ms" for k, v in row.items()))

    return scalability_results


def plot_scalability(scalability_results: Dict[int, Dict[str, Any]], save_path: str = "scalability_comparison.png"):
    node_sizes = sorted(scalability_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for algo in ["QPSO", "GA", "Standard PSO"]:
        runtimes = [scalability_results[n][algo]["runtime"] * 1000 for n in node_sizes]
        costs = [scalability_results[n][algo]["cost"] for n in node_sizes]
        ax1.plot(node_sizes, runtimes, marker="o", label=algo)
        ax2.plot(node_sizes, costs, marker="o", label=algo)

    ax1.set_xlabel("Number of nodes (network size)")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_title("Scalability: Runtime vs Network Size")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Number of nodes (network size)")
    ax2.set_ylabel("Best cost found")
    ax2.set_title("Scalability: Solution Quality vs Network Size")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scalability plot saved to: {save_path}")


if __name__ == "__main__":
    print("=" * 82)
    print("BENCHMARK: Single network, all algorithms")
    print("=" * 82)
    net = generate_synthetic_city_graph(n_nodes=25, seed=7)
    results = run_full_benchmark(net, source=0, destination=20, max_iter=120, seed=1)
    print_comparison_table(results)
    plot_convergence(results, save_path="/home/claude/sih26137/data/convergence_comparison.png")

    print("\n" + "=" * 82)
    print("SCALABILITY TEST")
    print("=" * 82)
    scal = run_scalability_test(node_sizes=[10, 20, 40, 80], max_iter=80, seed=1)
    plot_scalability(scal, save_path="/home/claude/sih26137/data/scalability_comparison.png")
