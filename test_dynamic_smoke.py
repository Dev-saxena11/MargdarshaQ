from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp
from app.core.dynamic_vrp import simulate_dynamic_reroute

print("1. Generating synthetic city network...")
net = generate_synthetic_city_graph(n_nodes=30, seed=42)

print("2. Generating VRP instance...")
vrp = generate_synthetic_vrp(net, n_customers=12, depot=0, vehicle_capacity=80, seed=1)

print("3. Running dynamic VRP simulation (traffic incident at t=60m)...")
res = simulate_dynamic_reroute(
    problem=vrp,
    incident_factor=4.0,
    trigger_time_min=60.0,
    algorithm="qpso",
    n_particles=30,
    max_iter=100,
    seed=1
)

print("\n--- DYNAMIC TRAFFIC SIMULATION RESULTS ---")
print(f"Incident: Edge ({res.incident['u']} -> {res.incident['v']}) with {res.incident['factor']}x congestion at t={res.trigger_time_min}m")
print(f"Served Customers by t={res.trigger_time_min}m: {res.served_customer_ids}")
print(f"Unserved Customers Re-routed: {res.unserved_customer_ids}")
print(f"Static Execution Total Time: {res.static_solution.total_time:.2f} min (TW Violations: {res.static_solution.time_window_violation:.2f})")
print(f"Dynamic Rerouted Total Time: {res.dynamic_solution.total_time:.2f} min (TW Violations: {res.dynamic_solution.time_window_violation:.2f})")
print(f"Time Saved: {res.time_saved_min:.2f} min ({res.time_saved_pct:.1f}%)")
print(f"Congestion Delay Avoided: {res.delay_avoided_min:.2f} min ({res.delay_avoided_pct:.1f}%)")
print(f"TW Penalties Avoided: {res.tw_violations_avoided:.2f}")
