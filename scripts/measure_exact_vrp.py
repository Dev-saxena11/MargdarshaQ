"""
measure_exact_vrp.py
---------------------
Runtime of the exact CVRPTW solver by instance size, which is the number that
decides where its guard belongs.

Stage 1 of the solver enumerates every ordering of every subset of customers,
so the work grows like e * n factorial: each additional customer costs roughly
ten times the last. The table in app/core/exact_vrp.py's docstring came from
this script.

Run with:  python scripts/measure_exact_vrp.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.exact_vrp import solve_vrp_exact
from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp

SIZES = (5, 6, 7, 8, 9, 10)


def main():
    print()
    print("Exact CVRPTW solver runtime by customer count")
    print("-" * 64)
    for n in SIZES:
        net = generate_synthetic_city_graph(n_nodes=max(20, n * 3), seed=7)
        problem = generate_synthetic_vrp(
            net, n_customers=n, depot=0, vehicle_capacity=200, seed=3
        )
        t0 = time.perf_counter()
        # max_customers is raised so the whole curve can be measured; the
        # library default stops at 10 on purpose.
        result = solve_vrp_exact(problem, max_customers=max(SIZES))
        elapsed = time.perf_counter() - t0
        print(
            f"  n={n:2d}  vehicles={problem.n_vehicles}  "
            f"{elapsed:7.2f}s   subsets={result.n_evaluations:>6,}   "
            f"optimum={result.best_fitness:.3f}"
        )
    print()


if __name__ == "__main__":
    main()
