# SIH26137 — Quantum-Inspired Intelligent Traffic Route Optimization

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-sih--26137.vercel.app-000000?logo=vercel&logoColor=white)](https://sih-26137.vercel.app)
[![Live API](https://img.shields.io/badge/Live%20API-sih26137.onrender.com-46E3B7?logo=render&logoColor=white)](https://sih26137.onrender.com/docs)

A quantum-inspired metaheuristic optimization framework (QPSO) for solving
large-scale Vehicle Routing Problems (VRP) under traffic congestion, benchmarked
against classical metaheuristics (GA, SA, standard PSO) and exact/greedy baselines.

## 🚀 Live Demo

The platform is deployed as two services (see [DEPLOYMENT.md](DEPLOYMENT.md) for why):

| | URL | Hosted on |
| :--- | :--- | :--- |
| **Dashboard (UI)** | **https://sih-26137.vercel.app** | Vercel (static) |
| **Backend API** | **https://sih26137.onrender.com** · [Swagger docs](https://sih26137.onrender.com/docs) | Render (web service) |

Just open the dashboard link — when served from a deployed origin it points itself at the
live API automatically. Run it locally and it defaults to `http://127.0.0.1:8000` instead,
so local development is unaffected. Either way you can override the target in the
**API Base URL** field at the top of the dashboard.

> ⏱️ **First request may take 30–50 seconds.** The API runs on Render's free tier, which
> sleeps after ~15 minutes of inactivity and cold-starts on the next request. It is not
> broken — give the first call a moment, then it responds normally. Warm it up by opening
> the [API docs](https://sih26137.onrender.com/docs) a minute before a demo.

## Setup (on your machine)

```bash
cd sih26137
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger UI — you can
generate a network, generate a VRP instance on it, solve it, and run the full
benchmark, all from the browser.

## API workflow

1. `POST /api/network/generate` → creates a synthetic city road network, returns `network_id`
2. `POST /api/vrp/generate` → creates a CVRPTW instance (customers, demands, time windows) on that network, returns `vrp_id`
3. `POST /api/vrp/solve` → solves the instance with one algorithm (`qpso`, `ga`, `sa`, `standard_pso`, or `greedy`)
4. `POST /api/benchmark/run` → solves with **all** algorithms and returns a side-by-side comparison (for your benchmarking/report deliverable)

Two additional endpoints support real-world data and the dashboard:
- `POST /api/network/from_osm` → loads a REAL city road network from OpenStreetMap (via `osmnx`) instead of a synthetic one — same `network_id` workflow after that
- `GET /api/network/{network_id}` / `GET /api/vrp/{vrp_id}` → refetch a previously generated network/instance

## Running the visualization dashboard

A standalone control-room style dashboard lives at `frontend/dashboard.html` —
network generation (synthetic or real OSM city), VRP instance generation,
solve, and full benchmark comparison, all with a live map and convergence chart.

1. Start the API (`uvicorn app.main:app --reload --port 8000`)
2. Serve the frontend folder (opening the file directly can hit browser
   fetch/CORS restrictions on some setups, so a tiny local server is safer):
   ```bash
   cd frontend
   python -m http.server 5500
   ```
3. Open **http://127.0.0.1:5500/dashboard.html**
4. Click "check" next to the API address field (defaults to
   `http://127.0.0.1:8000`) to confirm it's connected
5. Generate a network (synthetic, or type a real place name for OSM — e.g.
   "Connaught Place, New Delhi, India" — requires `osmnx` + internet), then
   generate a VRP instance on it, then Solve or Run full benchmark

The map auto-switches between a real Leaflet map (OSM networks) and a custom
SVG plot (synthetic networks, which don't have real lat/lon). Routes are drawn
along the actual road-network path (not straight lines) using the shortest
path each vehicle actually takes.

**Note:** this dashboard was built and syntax-checked in an environment
without a live backend to test against (no FastAPI/browser available in the
build sandbox) — the underlying business logic it depends on was validated
separately, but test the actual UI end-to-end first thing on your machine and
expect to fix small integration issues (e.g. CORS, port mismatches, edge
cases in map rendering for very large OSM networks).

## Deploying

Backend and frontend deploy to **different platforms** (Render for the backend,
Vercel/Netlify for the frontend) — see [DEPLOYMENT.md](DEPLOYMENT.md) for why and how.
Do not attempt to deploy the whole app to Vercel.

Both are already live — see [Live Demo](#-live-demo) above for the URLs.

## Running the standalone benchmark scripts (no API needed)

These generate the convergence and scalability plots directly to `data/`:

```bash
# Shortest-path formulation (simpler, for sanity-checking the algorithm)
python -m app.core.benchmark

# Full CVRPTW formulation (the actual deliverable)
python -m app.core.benchmark_vrp
```

## Project structure

```
sih26137/
├── app/
│   ├── core/
│   │   ├── graph_model.py            # Weighted traffic network + synthetic city generator
│   │   ├── qpso.py                   # QPSO for single-vehicle shortest path
│   │   ├── vrp_problem.py            # CVRPTW formulation (capacity + time windows)
│   │   ├── qpso_vrp.py               # QPSO + 2-opt/or-opt local-search hybrid for VRP (core deliverable)
│   │   ├── local_search.py           # 2-opt / or-opt operators
│   │   ├── classical_baselines.py    # Dijkstra, A*, GA, SA, standard PSO (shortest-path)
│   │   ├── classical_baselines_vrp.py# GA, SA, standard PSO, Greedy NN (VRP)
│   │   ├── benchmark.py              # Benchmarking suite (shortest-path)
│   │   ├── benchmark_vrp.py          # Benchmarking suite (VRP) — convergence, scalability, robustness
│   │   ├── osm_network.py            # Real OpenStreetMap network loader (osmnx)
│   │   └── store.py                  # In-memory store for API state (network_id/vrp_id -> objects)
│   ├── models/
│   │   └── schemas.py                # Pydantic request/response models
│   ├── api/
│   │   └── routes.py                 # FastAPI endpoints
│   └── main.py                       # FastAPI app entrypoint
├── frontend/
│   └── dashboard.html                # Standalone visualization dashboard (map + charts + controls)
├── data/                             # Generated plots land here
├── requirements.txt
└── README.md
```

## Key design notes / findings (for your report)

- **Encoding:** VRP is combinatorial, but QPSO/GA/PSO are continuous-space
  algorithms. We use a random-key encoding — one continuous gene per customer
  in `[0, n_vehicles)`; `floor()` picks the vehicle, the fractional part
  orders customers within that vehicle's route.
- **Why CVRPTW and not plain shortest-path:** plain shortest-path on a modest
  graph is trivially solved by every method (including Dijkstra exactly), so
  it can't demonstrate QPSO's advantages. Multi-vehicle VRP with capacity and
  time-window constraints is genuinely NP-hard, which is where quantum-inspired
  search shows real separation from classical baselines.
- **High-dimensional instability fix:** vanilla QPSO's `ln(1/u)` stochastic
  jump term is unbounded and, in high dimensions (many customers), an
  increasingly-likely single large jump on any dimension disrupts otherwise-good
  solutions. We cap this jump term (scaled down as dimensionality grows) to
  restore stable convergence at scale — see the docstring in `qpso_vrp.py`.
- **Memetic hybridization:** even after the jump-cap fix, vanilla QPSO trailed
  Standard PSO at 40–60 customers because it lacked any local refinement step.
  We periodically apply 2-opt/or-opt local search to the swarm's global best
  and reinject the improved solution (Lamarckian learning). This closed the
  gap completely — QPSO+local-search now wins at every tested scale (10–60
  customers), at the cost of higher runtime from the extra refinement passes.
- **Robustness:** single-seed comparisons between metaheuristics can be
  misleading due to randomness. `benchmark_vrp.py` includes a 5-seed robustness
  check reporting mean ± std, not just a single best-case number.
- **Bug found and fixed during development:** the initial or-opt implementation
  iterated over a stale snapshot of route indices while mutating the routes
  list mid-pass, which could silently duplicate a customer across two routes
  (or drop one). Building the visualization dashboard is what surfaced it —
  a duplicate customer was visible in a rendered route. Fixed by looking up
  each customer's current position fresh on every iteration instead of trusting
  stale indices (see `local_search.py`). Worth mentioning in your report as an
  example of why visualization matters for catching silent correctness bugs
  that pure fitness numbers can hide.
