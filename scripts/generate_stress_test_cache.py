"""
scripts/generate_stress_test_cache.py
--------------------------------------
Measure how the solvers actually scale, and cache the result for the dashboard.

The scalability chart in the control room used to be drawn from an assumed
complexity curve -- one measured runtime, multiplied by n^2 for QPSO and by n
for GA -- with axes labelled "Number of customers" and "Runtime (ms)" as though
something had been timed at each size. Nothing had. This script produces the
numbers that chart now plots, so "how did you measure this" has an answer.

Defaults are chosen so the answer is reproducible by whoever asks:

  - a synthetic 300-node city graph rather than a live OSM pull, so the sweep
    needs no Overpass instance, no internet and no API key, and gives the same
    graph on every machine;
  - sizes small enough to finish in minutes rather than the hour the previous
    100/200-customer real-city defaults could take;
  - a fixed seed range, so re-running reproduces the figures rather than
    producing new ones that quietly disagree with the committed cache.

Every default is overridable, and the settings used are written into the output
alongside the measurements so a chart can state exactly what was run.

    python scripts/generate_stress_test_cache.py
    python scripts/generate_stress_test_cache.py --sizes 20 40 60 --seeds 5
    python scripts/generate_stress_test_cache.py --network real_city   # needs OSM
"""

import argparse
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.benchmark_vrp import run_stress_test_at_scale

OUT_PATH = os.path.join("data", "stress_test_cache.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100],
                        help="Customer counts to measure (default: 20 40 60 80 100)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Runs per (size, algorithm); the best is kept (default: 3)")
    parser.add_argument("--budget", type=float, default=90.0,
                        help="Seconds before a single run is called a timeout (default: 90)")
    parser.add_argument("--network", choices=["synthetic", "real_city"], default="synthetic",
                        help="synthetic (default, reproducible offline) or a live OSM pull")
    parser.add_argument("--out", default=OUT_PATH, help=f"Output path (default: {OUT_PATH})")
    args = parser.parse_args()

    algorithms = ["qpso_local_search", "standard_pso"]

    print(f"Measuring scalability: sizes={args.sizes}, seeds={args.seeds}, "
          f"network={args.network}, budget={args.budget}s")
    print(f"{len(args.sizes) * len(algorithms) * args.seeds} runs in total.\n")

    results = run_stress_test_at_scale(
        n_customers_list=args.sizes,
        algorithms=algorithms,
        network_source=args.network,
        network_id=None,
        time_budget_seconds=args.budget,
        n_seeds=args.seeds,
    )

    # What was measured, recorded next to the measurements. A chart that cannot
    # say what produced its numbers is the problem this script exists to fix.
    results["measurement"] = {
        "sizes": args.sizes,
        "algorithms": algorithms,
        "seeds_per_point": args.seeds,
        "time_budget_seconds": args.budget,
        "network_source": args.network,
        "note": "Runtime and fitness are measured, not extrapolated. Each point "
                "is the best of `seeds_per_point` runs at that size.",
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWritten to {args.out}")
    for size in args.sizes:
        row = results["results"].get(str(size), {})
        cells = "  ".join(
            f"{algo.split('_')[0]}={(row.get(algo) or {}).get('runtime_ms') or float('nan'):.0f}ms"
            for algo in algorithms
        )
        print(f"  {size:>4} customers   {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
