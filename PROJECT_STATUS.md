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

## 1a. Official PS deliverables — mapping against what's built

Source: official SIH26137 PDF (Egreen Quanta, AICTE SIH 2026 Quantum Technology Vertical). This is the authoritative deliverables list — everything else in this doc should trace back to it.

**Expected Solution (verbatim intent):** a complete software platform implementing a Quantum-Inspired Metaheuristic Optimization Algorithm for intelligent traffic routing, including graph-based network modelling, mathematical formulation, constraint handling, convergence analysis, and systematic performance benchmarking.

| # | PS Deliverable | Status | Gap vs. exact PS wording |
|---|---|---|---|
| 1 | **Graph-based Network Model** — weighted directed/undirected graph, nodes/edges, **"dynamic weight update mechanism"** | 🟡 Mostly done | Nodes/edges/weights all exist (`graph_model.py`, `osm_network.py`). `update_congestion()` exists but there's no automated *dynamic* update loop — PS explicitly names this as a required component, not optional polish. |
| 2 | **Mathematical Formulation** — complete optimization model of the VRP/traffic routing problem: objective function, capacity/time-window/flow constraints, decision variables | 🟡 Logic exists in code, not written up | Fitness function + penalties are implemented in `vrp_problem.py`, but PS wants a **formal written model** (objective function, constraints, decision variables as equations) — this doesn't exist as a document yet, only as code. |
| 3 | **Quantum-Inspired Algorithm Module** — QPSO or equivalent, particle/route encoding, quantum update rules | 🟢 Strongest deliverable | Fully implemented and matches PS wording almost exactly (`qpso_vrp.py` — delta-potential-well update, random-key encoding, memetic hybrid). |
| 4 | **Software Platform / Prototype** — UI/API, input of network data & traffic conditions, output of optimized routes, visualization on map/graph | 🟡 Built, untested | FastAPI + dashboard cover every listed component (input, output, map visualization). Untested end-to-end (see Loophole #1) — functionally complete on paper, not verified working. |
| 5 | **Demonstration** — algorithm description, implementation details, experimental results, **at least one realistic urban network (or synthetic large instance) showing near-optimal routes under varying traffic conditions** | 🔴 Weakest vs. PS wording | Two specific gaps: (a) no locked-in real urban case study yet — only ad hoc OSM pulls; (b) congestion is static per-run, not **"varying"** as PS explicitly requires. This deliverable's wording is the most specific in the whole PS and currently the least satisfied. |

**Additional requirement from the Objectives section:** benchmarking must be against **"conventional metaheuristics and exact methods"** — we have exact methods for shortest-path (Dijkstra/A*), but no exact method (e.g. brute-force/ILP on a small instance) for VRP itself. Worth adding a small-instance exact baseline (5–10 customers) purely to strengthen the "not just heuristic-vs-heuristic" comparison story.

---

## 2. Loopholes / known weak points

These are not hidden — flagging them upfront so nobody assumes something works just because it exists in the repo.

| # | Issue | Why it matters |
|---|---|---|
| 1 | **Dashboard is untested end-to-end.** Built without a live backend running against it in the original build environment. | This is our demo surface. If it breaks live in front of judges, that's the whole pitch gone. |
| 2 | **Traffic is not actually dynamic.** `congestion_factor` is randomized once at network generation and then stays fixed unless manually updated via `update_congestion()`. | PS deliverable #1 explicitly requires a **"dynamic weight update mechanism"**, and deliverable #5 explicitly requires demonstrating routes **"under varying traffic conditions."** This isn't just good-to-have — it's named twice in the official deliverables. |
| 3 | **No real-world validation.** Everything tested so far is synthetic graphs or generic OSM pulls — no locked-in, real city case study with a believable delivery/logistics scenario. | PS deliverable #5 explicitly asks for **"at least one realistic urban network... showing near-optimal routes."** Currently the least-satisfied deliverable in the whole PS. |
| 4 | **"Quantum-inspired" advantage isn't narrated yet.** The benchmark data exists and looks good, but there's no write-up translating it into a clear "why does this beat classical metaheuristics" story with numbers. | Without this, the core selling point of the PS (quantum-inspired vs classical) doesn't land in the pitch. |
| 4a | **No formal mathematical formulation document.** The objective function and constraints exist as code (`vrp_problem.py`) but not as a written model. | PS deliverable #2 explicitly asks for a "complete optimization model" — objective function, constraints, decision variables as equations, not just implementation. |
| 4b | **No exact-method baseline for VRP.** We have exact shortest-path methods (Dijkstra/A*), but no exact solver (even brute-force/ILP on a tiny instance) for VRP itself. | PS Objectives section explicitly says benchmarking should be against **"conventional metaheuristics and exact methods."** |
| 5 | **No fallback plan for live demo.** If OSM/internet fails on stage, there's currently no rehearsed backup. | Standard hackathon failure mode — OSM/API calls dying live. |
| 6 | **In-memory store only** (`app/core/store.py`) — no persistence. Restarting the API loses all generated networks/VRP instances. | Fine for now, but will bite us if we build a stateful demo flow. |
| 7 | **No automated tests** (`tests/` folder exists but is empty). | Any refactor from here on risks silently breaking the algorithm (see bug #already-found above — that class of bug is easy to reintroduce). |
| 8 | **Single-scenario benchmarking.** Robustness check exists (5-seed mean±std) but only on synthetic data — not yet run against the real-city case study we'll actually demo. | Numbers we show judges should come from the same scenario we demo, not a different synthetic one. |

---

## 3. Roadmap (priority order)

Re-ordered so the items that map directly to named PS deliverables come first — these aren't optional polish, they're explicitly graded requirements.

1. **Stabilize the dashboard.** Run the full API + dashboard flow locally end-to-end. Fix CORS/port issues, confirm network → VRP → solve → benchmark works without manual workarounds. *Nobody should build new features on top of this until it's confirmed working.* (→ deliverable #4)
2. **Make traffic dynamic.** Add time-varying or event-based congestion updates (e.g. congestion changes mid-route, optimizer reacts). This satisfies two PS requirements at once: the "dynamic weight update mechanism" (deliverable #1) and "varying traffic conditions" (deliverable #5). (→ deliverables #1, #5)
3. **Lock a real city case study.** Pick one real place via OSM, define a realistic scenario (N customers, depot, capacities, time windows), and make this the canonical demo scenario — all future benchmark numbers and the live demo should use this, not ad hoc synthetic graphs. (→ deliverable #5)
4. **Write the formal mathematical formulation.** Turn `vrp_problem.py`'s objective/penalty logic into a proper written model: objective function, decision variables, capacity/time-window/flow constraints as equations. This is a document, not code — can be drafted in parallel with other work. (→ deliverable #2)
5. **Add a small-instance exact baseline.** Brute-force or ILP solve on a tiny VRP instance (5–10 customers) purely to have an "exact method" comparison point alongside the metaheuristics. (→ Objectives section requirement)
6. **Write the "why quantum-inspired wins" narrative.** Turn the existing scalability/convergence plots into a clear comparison story (including the honest runtime trade-off) for the report and pitch.
7. **Polish the frontend layer.** Add a clean, judge-facing summary view (before/after route, % savings, congestion avoided) on top of the existing control-room dashboard — it doesn't need to be replaced, just made presentable for a non-technical audience.
8. **Add a demo fallback.** Rehearsed synthetic-network backup + a recorded video run in case live OSM/network access fails on stage.
9. **(Once the above is stable) Add basic tests** around the local search and chromosome encoding so future contributions don't silently reintroduce correctness bugs.

---

## 4. Ground rules for contributing

- Don't touch the QPSO jump-cap or the local-search reinjection logic without reading the docstrings in `qpso_vrp.py` and `local_search.py` first — both encode fixes for real bugs we already hit once.
- Any change to the algorithm core should be re-benchmarked against `app/core/benchmark_vrp.py` before merging — we need to know if a "cleanup" quietly changes our headline numbers.
- Keep the real-city case study (once locked in step 3) as the reference scenario for any new benchmark numbers you generate.
