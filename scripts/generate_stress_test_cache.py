import os
import sys
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.benchmark_vrp import run_stress_test_at_scale

def main():
    print("Starting scalability stress test cache generation...")
    # Using the time budget of 300 seconds
    results = run_stress_test_at_scale(
        n_customers_list=[100, 200],
        algorithms=["qpso_local_search", "standard_pso"],
        network_source="real_city",
        network_id="delhi_osm_locked",
        time_budget_seconds=300.0,
        n_seeds=3
    )
    
    os.makedirs("data", exist_ok=True)
    with open("data/stress_test_cache.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nStress test generated and saved to data/stress_test_cache.json")

if __name__ == "__main__":
    main()
