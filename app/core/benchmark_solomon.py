"""
benchmark_solomon.py
---------------------
Run this project's solvers on Solomon's CVRPTW instances and report the gap to
the published best-known solutions.

Scoring, and why it is not the fitness number
=============================================
Solomon's objective is hierarchical: minimise the number of vehicles first,
then total distance. A solution using more vehicles does not beat one using
fewer however short its routes are. So the comparable figures are

    (vehicles used, total distance)      -- for a FEASIBLE solution

and not the solver's internal fitness, which folds penalties into a single
number. A run that breaks a time window can post a very low fitness and mean
nothing: the literature's figures are all feasible, so an infeasible run has no
gap to report and is marked as such rather than given a flattering percentage.

The gap is therefore reported only for feasible solutions, and alongside the
vehicle count rather than instead of it -- a result 2% short on distance while
using five more vehicles is worse, not better, and printing distance alone
would hide that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from app.core.classical_baselines_vrp import (
    VRPBenchmarkResult,
    run_ga_vrp,
    run_greedy_nn_vrp,
    run_sa_vrp,
    run_standard_pso_vrp,
)
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.solomon import available_instances, best_known, load_problem
from app.core.vrp_problem import VRPProblem, VRPSolution


@dataclass
class SolomonRun:
    """One algorithm on one instance, scored the way Solomon scores it."""

    instance: str
    algorithm: str
    feasible: bool
    vehicles_used: int
    distance: float
    runtime_sec: float
    best_known_vehicles: Optional[int] = None
    best_known_distance: Optional[float] = None

    @property
    def distance_gap_pct(self) -> Optional[float]:
        """
        How far above the best-known distance this landed, as a percentage.

        None when the run is infeasible or the instance has no recorded
        best-known value -- in both cases there is no comparison to make, and a
        number here would be read as one.
        """
        if not self.feasible or not self.best_known_distance:
            return None
        return 100.0 * (self.distance - self.best_known_distance) / self.best_known_distance

    @property
    def extra_vehicles(self) -> Optional[int]:
        if self.best_known_vehicles is None:
            return None
        return self.vehicles_used - self.best_known_vehicles


def _score(solution: Optional[VRPSolution]) -> tuple:
    """(feasible, vehicles used, total distance) for a solved instance."""
    if solution is None:
        return False, 0, float("inf")
    used = sum(1 for route in solution.routes if route)
    return solution.feasible, used, solution.total_distance


def _qpso(problem: VRPProblem, max_iter: int, seed: int) -> VRPBenchmarkResult:
    t0 = time.perf_counter()
    res = QPSOVRPOptimizer(problem, n_particles=50, max_iter=max_iter, seed=seed).optimize()
    return VRPBenchmarkResult(
        algorithm="QPSO (Quantum-Inspired PSO)",
        best_solution=res.best_solution,
        best_fitness=res.best_fitness,
        runtime_sec=time.perf_counter() - t0,
        n_evaluations=res.n_evaluations,
        convergence_curve=res.convergence_curve,
    )


SOLVERS: Dict[str, Callable[[VRPProblem, int, int], VRPBenchmarkResult]] = {
    "QPSO": _qpso,
    "GA": lambda p, it, s: run_ga_vrp(p, pop_size=50, max_iter=it, seed=s),
    "SA": lambda p, it, s: run_sa_vrp(p, max_iter=it * 20, seed=s),
    "Standard PSO": lambda p, it, s: run_standard_pso_vrp(p, n_particles=50, max_iter=it, seed=s),
    "Greedy NN": lambda p, it, s: run_greedy_nn_vrp(p),
}


def run_instance(name: str, algorithms: Optional[List[str]] = None,
                 max_iter: int = 200, seed: int = 1) -> List[SolomonRun]:
    """Every requested algorithm on one instance."""
    problem = load_problem(name)
    known = best_known(name)
    runs: List[SolomonRun] = []

    for algo in (algorithms or list(SOLVERS)):
        solver = SOLVERS[algo]
        result = solver(problem, max_iter, seed)
        feasible, used, distance = _score(result.best_solution)
        runs.append(SolomonRun(
            instance=name.upper(),
            algorithm=result.algorithm,
            feasible=feasible,
            vehicles_used=used,
            distance=distance,
            runtime_sec=result.runtime_sec,
            best_known_vehicles=known[0] if known else None,
            best_known_distance=known[1] if known else None,
        ))
    return runs


def run_suite(instances: Optional[List[str]] = None,
              algorithms: Optional[List[str]] = None,
              max_iter: int = 200, seed: int = 1) -> List[SolomonRun]:
    runs: List[SolomonRun] = []
    for name in (instances or available_instances()):
        runs.extend(run_instance(name, algorithms=algorithms, max_iter=max_iter, seed=seed))
    return runs


def print_table(runs: List[SolomonRun]) -> None:
    """One row per algorithm per instance, grouped by instance."""
    by_instance: Dict[str, List[SolomonRun]] = {}
    for run in runs:
        by_instance.setdefault(run.instance, []).append(run)

    header = (f"{'Algorithm':<28} {'Feasible':>9} {'Vans':>6} {'vs BK':>6} "
              f"{'Distance':>11} {'Best known':>11} {'Gap':>9} {'Runtime(s)':>11}")
    for instance, rows in by_instance.items():
        known = best_known(instance)
        known_text = f"{known[0]} vans / {known[1]:.2f}" if known else "not recorded"
        print(f"\n{instance}  (best known: {known_text})")
        print(header)
        print("-" * len(header))
        for r in rows:
            gap = "--" if r.distance_gap_pct is None else f"{r.distance_gap_pct:+.2f}%"
            extra = "--" if r.extra_vehicles is None else f"{r.extra_vehicles:+d}"
            bk = "--" if r.best_known_distance is None else f"{r.best_known_distance:11.2f}"
            print(f"{r.algorithm:<28} {str(r.feasible):>9} {r.vehicles_used:>6} {extra:>6} "
                  f"{r.distance:>11.2f} {bk:>11} {gap:>9} {r.runtime_sec:>11.2f}")

    infeasible = [r for r in runs if not r.feasible]
    if infeasible:
        print(f"\n{len(infeasible)} of {len(runs)} runs were infeasible and have no gap to "
              f"report. Published figures are all feasible, so an infeasible run is not a "
              f"result that happens to be worse -- it is not a result.")
