# SIH26137 — Project Status, Loopholes & Roadmap

> **Read this before writing any code.** This file exists so every team member knows exactly what's done, what's broken/missing, and what to build next — without re-discovering it from scratch. Keep it updated as we progress.

**Problem Statement:** SIH26137 — Quantum-Inspired Intelligent Traffic Route Optimization in Transportation Systems Using Metaheuristic

**Current stage:** Early prototype (built in ~1 day as a proof-of-concept). Core algorithm works and has been benchmarked. Product/demo layer is largely missing or untested.

---

## 1. What's already built

### Algorithm core (the actual deliverable)
- **QPSO (Quantum-inspired PSO)** — `app/core/qpso.py` (single-vehicle shortest path) and `app/core/qpso_vrp.py` (full VRP version). Uses the delta-potential-well update rule (attraction toward pbest/gbest + mean-best position), operating on a random-key chromosome encoding (one gene per customer).
- **VRP formulation** — `app/core/vrp_problem.py`: CVRPTW (Capacitated VRP with Time Windows) — vehicle capacity + delivery time windows, not just plain shortest path.
- **2-opt / or-opt local search hybrid** — `app/core/local_search.py`, wired into QPSO as a memetic (Lamarckian) refinement step. This is what closes the gap between QPSO and standard PSO at scale — without it, vanilla QPSO was *losing* to standard PSO at 40–60 customers.
- **Classical baselines for comparison** — GA, SA, standard PSO, Greedy NN, Dijkstra/A* (`classical_baselines.py`, `classical_baselines_vrp.py`). Needed because the PS explicitly asks us to benchmark against classical metaheuristics, not just show our own number in isolation.
- **Known bug already found & fixed**: an early or-opt implementation mutated the routes list while iterating over stale indices, silently duplicating/dropping a customer. Found via the visualization dashboard, not the fitness numbers. Worth keeping as a note for the report — it's a good example of why we need visualization, not just metrics.
- **High-dimensional instability fix**: vanilla QPSO's `ln(1/u)` jump term is unbounded and misbehaves as the number of customers grows. We cap it (see docstring in `qpso_vrp.py`). Don't remove this without understanding why it's there.

### Infrastructure
- **FastAPI backend** (`app/api/routes.py`, `app/main.py`) — endpoints for network generation, VRP instance generation, solving with any one algorithm, and running the full benchmark across all algorithms.
- **Real-world map support** — `app/core/osm_network.py` loads an actual city road network from OpenStreetMap via `osmnx`, on top of the same network_id workflow as the synthetic generator.
- **Synthetic city generator** (`graph_model.py`) — k-nearest-neighbor road network, randomized congestion, for quick testing without internet/OSM.
- **Dashboard** (`frontend/dashboard.html`) — control-room style UI: generate network (synthetic or real), generate VRP, solve, run full benchmark, view on a Leaflet map (real) or SVG plot (synthetic) with a convergence chart.
- **Benchmark results already generated** (`data/*.png`) — convergence + scalability comparisons already show QPSO+local-search producing better solution quality than GA/standard PSO at larger problem sizes (e.g. ~60 customers), at the cost of higher runtime. This is our headline result — don't lose it, but re-verify it as the codebase changes.

---

## 2. Loopholes / known weak points

These are not hidden — flagging them upfront so nobody assumes something works just because it exists in the repo.

| # | Issue | Why it matters |
|---|---|---|
| 1 | **Dashboard is untested end-to-end.** Built without a live backend running against it in the original build environment. | This is our demo surface. If it breaks live in front of judges, that's the whole pitch gone. |
| 2 | **Traffic is not actually dynamic.** `congestion_factor` is randomized once at network generation and then stays fixed unless manually updated via `update_congestion()`. | The PS title is literally "traffic route optimization." Right now we don't simulate traffic *changing* and re-routing in response — a judge will ask about this directly. |
| 3 | **No real-world validation.** Everything tested so far is synthetic graphs or generic OSM pulls — no locked-in, real city case study with a believable delivery/logistics scenario. | Judges respond much better to "here's Bareilly/some real city with 25 delivery points" than "here's a random graph." |
| 4 | **"Quantum-inspired" advantage isn't narrated yet.** The benchmark data exists and looks good, but there's no write-up translating it into a clear "why does this beat classical metaheuristics" story with numbers. | Without this, the core selling point of the PS (quantum-inspired vs classical) doesn't land in the pitch. |
| 5 | **No fallback plan for live demo.** If OSM/internet fails on stage, there's currently no rehearsed backup. | Standard hackathon failure mode — OSM/API calls dying live. |
| 6 | **In-memory store only** (`app/core/store.py`) — no persistence. Restarting the API loses all generated networks/VRP instances. | Fine for now, but will bite us if we build a stateful demo flow. |
| 7 | **No automated tests** (`tests/` folder exists but is empty). | Any refactor from here on risks silently breaking the algorithm (see bug #already-found above — that class of bug is easy to reintroduce). |
| 8 | **Single-scenario benchmarking.** Robustness check exists (5-seed mean±std) but only on synthetic data — not yet run against the real-city case study we'll actually demo. | Numbers we show judges should come from the same scenario we demo, not a different synthetic one. |

---

## 3. Roadmap (priority order)

1. **Stabilize the dashboard.** Run the full API + dashboard flow locally end-to-end. Fix CORS/port issues, confirm network → VRP → solve → benchmark works without manual workarounds. *Nobody should build new features on top of this until it's confirmed working.*
2. **Make traffic dynamic.** Add time-varying or event-based congestion updates (e.g. congestion changes mid-route, optimizer reacts) so the "traffic optimization" claim is actually demonstrated, not just implied by a static random multiplier.
3. **Lock a real city case study.** Pick one real place via OSM, define a realistic scenario (N customers, depot, capacities, time windows), and make this the canonical demo scenario — all future benchmark numbers and the live demo should use this, not ad hoc synthetic graphs.
4. **Write the "why quantum-inspired wins" narrative.** Turn the existing scalability/convergence plots into a clear comparison story (including the honest runtime trade-off) for the report and pitch.
5. **Polish the frontend layer.** Add a clean, judge-facing summary view (before/after route, % savings, congestion avoided) on top of the existing control-room dashboard — it doesn't need to be replaced, just made presentable for a non-technical audience.
6. **Add a demo fallback.** Rehearsed synthetic-network backup + a recorded video run in case live OSM/network access fails on stage.
7. **(Once the above is stable) Add basic tests** around the local search and chromosome encoding so future contributions don't silently reintroduce correctness bugs.

---

## 4. Ground rules for contributing

- Don't touch the QPSO jump-cap or the local-search reinjection logic without reading the docstrings in `qpso_vrp.py` and `local_search.py` first — both encode fixes for real bugs we already hit once.
- Any change to the algorithm core should be re-benchmarked against `app/core/benchmark_vrp.py` before merging — we need to know if a "cleanup" quietly changes our headline numbers.
- Keep the real-city case study (once locked in step 3) as the reference scenario for any new benchmark numbers you generate.
