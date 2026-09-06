"""
qpso_vrp.py
------------
Quantum-inspired PSO (QPSO) applied to Capacitated VRP with Time Windows.

Same quantum-mechanical update rule as qpso.py (delta-potential-well model,
attraction toward a stochastic combination of pbest/gbest, drawn toward the
swarm's mean-best position) -- but now operating on VRP random-key
chromosomes (one gene per customer, value in [0, n_vehicles)) instead of
per-node routing priorities.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from app.core.vrp_problem import VRPProblem, evaluate_chromosome, VRPSolution
from app.core.local_search import local_search_refine, routes_to_chromosome


@dataclass
class VRPOptimizeResult:
    best_solution: VRPSolution
    best_fitness: float
    convergence_curve: List[float]
    n_evaluations: int


class QPSOVRPOptimizer:
    def __init__(
        self,
        problem: VRPProblem,
        n_particles: int = 50,
        max_iter: int = 200,
        alpha_start: float = 1.2,
        alpha_end: float = 0.35,
        seed: Optional[int] = None,
        capacity_penalty_weight: float = 50.0,
        time_window_penalty_weight: float = 10.0,
        max_jump_factor: Optional[float] = None,
        use_local_search: bool = True,
        local_search_interval: int = 15,
        local_search_passes: int = 2,
    ):
        self.problem = problem
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.rng = np.random.default_rng(seed)
        self.d = len(problem.customers)          # dimensionality = n_customers
        self.upper = problem.n_vehicles           # gene range [0, n_vehicles)
        self.cap_w = capacity_penalty_weight
        self.tw_w = time_window_penalty_weight
        # Cap on the ln(1/u) excursion term. Vanilla QPSO samples u ~ U(0,1) and
        # uses ln(1/u), which is unbounded as u -> 0. In low dimensions this rarely
        # matters, but in high-dimensional VRP chromosomes (one gene per customer),
        # the probability that AT LEAST ONE dimension draws a huge ln(1/u) value on
        # any given iteration grows quickly with dimensionality -- repeatedly
        # blowing up otherwise-good solutions right when the search should be
        # exploiting structure. Capping ln(1/u) at max_jump_factor keeps QPSO's
        # quantum-inspired stochastic jumps bounded, restoring stable convergence
        # at higher dimensions without changing the algorithm's core mechanism.
        # If not explicitly set, scale the cap DOWN as dimensionality grows: more
        # customers -> more genes that can independently draw a large jump ->
        # need a tighter per-dimension cap to keep the aggregate disruption bounded.
        # Empirically tuned against the CVRPTW benchmark (see benchmark_vrp.py).
        self.max_jump_factor = max_jump_factor if max_jump_factor is not None else max(0.3, 3.0 / (1 + self.d / 20.0))
        self.use_local_search = use_local_search
        self.local_search_interval = local_search_interval
        self.local_search_passes = local_search_passes
        self._n_eval = 0

    def _eval(self, chromosome: np.ndarray) -> VRPSolution:
        self._n_eval += 1
        return evaluate_chromosome(
            self.problem, chromosome,
            capacity_penalty_weight=self.cap_w,
            time_window_penalty_weight=self.tw_w,
        )

    def optimize(self, verbose: bool = False) -> VRPOptimizeResult:
        n, d, upper = self.n_particles, self.d, self.upper

        positions = self.rng.uniform(0, upper, size=(n, d))
        pbest = positions.copy()
        pbest_fit = np.full(n, np.inf)
        pbest_sol: List[Optional[VRPSolution]] = [None] * n

        gbest = None
        gbest_fit = np.inf
        gbest_sol: Optional[VRPSolution] = None

        convergence_curve = []

        for it in range(self.max_iter):
            fitnesses = np.zeros(n)
            for i in range(n):
                sol = self._eval(positions[i])
                fitnesses[i] = sol.fitness

                if sol.fitness < pbest_fit[i]:
                    pbest_fit[i] = sol.fitness
                    pbest[i] = positions[i].copy()
                    pbest_sol[i] = sol

                if sol.fitness < gbest_fit:
                    gbest_fit = sol.fitness
                    gbest = positions[i].copy()
                    gbest_sol = sol

            mbest = pbest.mean(axis=0)
            alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (it / max(1, self.max_iter - 1))

            phi = self.rng.uniform(0, 1, size=(n, d))
            p_attractor = phi * pbest + (1 - phi) * gbest

            u = self.rng.uniform(1e-6, 1.0, size=(n, d))
            sign = np.where(self.rng.uniform(0, 1, size=(n, d)) > 0.5, 1.0, -1.0)

            # cap the ln(1/u) excursion term (see __init__ docstring note) so a
            # single unlucky dimension can't blow up an otherwise-good solution
            jump = np.minimum(np.log(1.0 / u), self.max_jump_factor)

            positions = p_attractor + sign * alpha * np.abs(mbest - positions) * jump
            positions = np.clip(positions, 0.0, upper - 1e-9)

            convergence_curve.append(gbest_fit)

            # --- Memetic hybridization: periodically refine gbest with local search ---
            if self.use_local_search and gbest_sol is not None and \
               (it % self.local_search_interval == 0) and it > 0:
                refined_routes = local_search_refine(
                    self.problem, gbest_sol.routes, max_passes=self.local_search_passes
                )
                from app.core.vrp_problem import evaluate_solution
                refined_sol = evaluate_solution(self.problem, refined_routes)

                if refined_sol.fitness < gbest_fit:
                    gbest_fit = refined_sol.fitness
                    gbest_sol = refined_sol
                    gbest = routes_to_chromosome(self.problem, refined_routes)
                    gbest = np.clip(gbest, 0.0, upper - 1e-9)

                    # Lamarckian reinjection: replace the worst particle in the
                    # swarm with the refined solution so the population can
                    # build further on this improvement rather than losing it.
                    worst_idx = int(np.argmax(pbest_fit))
                    positions[worst_idx] = gbest.copy()
                    pbest[worst_idx] = gbest.copy()
                    pbest_fit[worst_idx] = gbest_fit
                    pbest_sol[worst_idx] = refined_sol

                    convergence_curve[-1] = gbest_fit  # reflect the refinement in this iteration's record

            if verbose and (it % 20 == 0 or it == self.max_iter - 1):
                print(f"Iter {it:4d} | gbest_fitness={gbest_fit:.3f} | feasible={gbest_sol.feasible if gbest_sol else None}")

        # Final polish: one more local-search refinement on the best solution found
        if self.use_local_search and gbest_sol is not None:
            from app.core.vrp_problem import evaluate_solution
            refined_routes = local_search_refine(
                self.problem, gbest_sol.routes, max_passes=self.local_search_passes + 1
            )
            refined_sol = evaluate_solution(self.problem, refined_routes)
            if refined_sol.fitness < gbest_fit:
                gbest_fit = refined_sol.fitness
                gbest_sol = refined_sol
                if convergence_curve:
                    convergence_curve[-1] = gbest_fit

        return VRPOptimizeResult(
            best_solution=gbest_sol,
            best_fitness=gbest_fit,
            convergence_curve=convergence_curve,
            n_evaluations=self._n_eval,
        )


if __name__ == "__main__":
    from app.core.graph_model import generate_synthetic_city_graph
    from app.core.vrp_problem import generate_synthetic_vrp

    net = generate_synthetic_city_graph(n_nodes=40, seed=7)
    vrp = generate_synthetic_vrp(net, n_customers=15, depot=0, vehicle_capacity=80, seed=3)

    opt = QPSOVRPOptimizer(vrp, n_particles=50, max_iter=150, seed=1)
    result = opt.optimize(verbose=True)

    print("\n--- Final VRP Solution ---")
    sol = result.best_solution
    for i, route in enumerate(sol.routes):
        if route:
            print(f"  Vehicle {i}: depot -> {' -> '.join(map(str, route))} -> depot")
    print(f"Total distance: {sol.total_distance:.2f} km | Total time: {sol.total_time:.2f} min")
    print(f"Capacity violation: {sol.capacity_violation:.2f} | TW violation: {sol.time_window_violation:.2f}")
    print(f"Feasible: {sol.feasible} | Fitness: {sol.fitness:.2f} | Evaluations: {result.n_evaluations}")
