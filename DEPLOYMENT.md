# Deployment Architecture

> Resolves [#43](https://github.com/Dev-saxena11/SIH26137/issues/43). Read this before
> attempting to deploy the app — **do not deploy the whole thing to Vercel.**

## Decision

This app deploys as **two separate services on two different platforms**, not as one
unified deployment:

| Component | Platform | Type | Live URL |
|---|---|---|---|
| Backend (`app/`, FastAPI) | **Render** | Persistent web service | https://sih26137.onrender.com |
| Frontend (`frontend/dashboard.html`) | **Vercel** (or Netlify) | Static site | https://sih-26137.vercel.app |

Both services are deployed and verified working together: the backend answers on
`/api/health`, and CORS is restricted to the Vercel origin (not `*`) via the
`CORS_ORIGINS` env var described below.

> ⏱️ Render's free tier sleeps after ~15 minutes idle, so the first request after a
> quiet period takes 30–50 seconds to cold-start. Warm it up before a live demo.

## Why not "just deploy it all to Vercel"

Vercel's hosting model for a Python backend is serverless functions, and this backend
doesn't fit that model:

- **Cold starts.** Vercel serverless functions spin down when idle and cold-start on the
  next request. The VRP solver (QPSO + local search, benchmark runs across multiple
  algorithms) is CPU-bound and can run for seconds to tens of seconds — a cold start on
  top of that produces a bad first-request experience or an outright timeout.
- **Execution time limits.** Vercel serverless functions have hard execution timeouts
  (10s on the free/Hobby tier, longer on paid plans, but still capped). A full
  `/benchmark` run across QPSO/GA/SA/PSO/greedy baselines is not guaranteed to fit
  inside that window as problem size grows.
- **Geospatial dependencies (`osmnx`, and transitively `geopandas`/`fiona`/`shapely`,
  which depend on GDAL).** These are heavy, sometimes-compiled-from-source packages.
  Vercel's Python serverless build environment is constrained and frequently fails to
  build packages with native/GDAL dependencies, or produces bundles that exceed the
  function size limit. Render's web service runtime is a normal persistent Linux
  container with a full `pip install` — the standard environment these packages are
  built and tested against.

Vercel (or Netlify) is genuinely the right tool for the **frontend** half: `dashboard.html`
is a single static file with no build step, and Vercel/Netlify are optimized exactly for
that (global CDN, instant deploys, generous free tier).

## Backend: Render

Config lives in [`render.yaml`](render.yaml) at the repo root (Render's
[Blueprint](https://render.com/docs/blueprint-spec) format — Render auto-detects it).

- **Build:** `pip install -r requirements.txt`
- **Start:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Runtime:** Python 3.11 (pinned via `PYTHON_VERSION` — osmnx's dependency chain is
  most reliably prebuilt for 3.11 at time of writing; bump this only after confirming
  wheels exist for a newer version, see verification steps below)

### Deploy steps
1. On [render.com](https://render.com), **New → Blueprint**, point it at this repo.
   Render reads `render.yaml` and creates the web service automatically.
2. Wait for the build to finish and open the Render **build logs**.
3. **Verify the `osmnx`/GDAL dependency chain actually built** (see below) — this is
   the single most likely failure point on a fresh Render service.
4. Once live, note the service URL (`https://<service-name>.onrender.com`) — the
   frontend needs it (see below).

### Verifying the osmnx/GDAL build (do this once per environment change)
Render's Python runtime ships manylinux wheels for `shapely`/`fiona`/`pyogrio` in most
cases, so a plain `pip install -r requirements.txt` *should* succeed without needing to
apt-install GDAL manually. Confirm this rather than assuming it:
- Check the Render build log for the `osmnx`, `geopandas`, `fiona` (or `pyogrio`)
  install lines — a successful build shows wheels being downloaded, not `Building wheel
  for fiona (pyproject.toml) ...` compiling from source.
- After deploy, hit `GET /docs` on the live service and exercise an endpoint that
  exercises `osm_network.py` (e.g. generate a real-map network) to confirm the import
  works at runtime, not just at install time.
- If a build does fail on GDAL, the fix is a `render-build.sh` that
  `apt-get install -y gdal-bin libgdal-dev` before `pip install` — not a switch back to
  Vercel. Document any such change here.

## Environment variables

### Render (backend)

Set in the Render dashboard → your service → **Environment** tab (already pre-filled if
deploying via the `render.yaml` Blueprint):

| Key | Value | Required? |
|---|---|---|
| `PYTHON_VERSION` | `3.11.9` | Recommended — pins the runtime so osmnx's dependency wheels resolve predictably. |
| `CORS_ORIGINS` | `*` (default) or a comma-separated list, e.g. `https://sih-26137.vercel.app,http://localhost:5500` | Optional but strongly recommended once the frontend URL is known — see CORS note below. |
| `CORS_ORIGIN_REGEX` | e.g. `https://sih-26137-[a-z0-9-]+\.vercel\.app` | Optional. Needed to allow Vercel preview/branch deploys, whose hostname changes per deploy. |
| `OPENROUTER_API_KEY` | Your key from [openrouter.ai/keys](https://openrouter.ai/keys) | Optional — enables free-text AI Assistant answers. Without it the assistant still works on its local engine. |
| `OPENROUTER_MODEL` | `openrouter/free` (default) | Optional. Must be a free model (`openrouter/free` or `*:free`); a paid id is rejected at startup. |

`PORT` is injected automatically by Render — do not set it yourself.

`OVERPASS_URL` is deliberately **not** set on Render: the local Overpass instance
it points at is something you run on the demo laptop, and the free tier has
neither the RAM nor the disk for it. See
[Running a local Overpass instance](#running-a-local-overpass-instance-for-live-demos).

**Set API keys in the Render dashboard only — never in a committed file.**
`.env` is gitignored; [`.env.example`](.env.example) holds placeholders. See
[docs/AI_ASSISTANT.md](docs/AI_ASSISTANT.md) for the full assistant setup.

### Vercel / Netlify (frontend)

**None required.** `frontend/dashboard.html` is a static file with no build step, so
there's nowhere for a platform env var to be read into it. It has a plain-text **API
Base URL** input field built into the page itself instead — set that by hand (or bake a
default into the HTML) rather than via a Vercel/Netlify env var.

### CORS note

`app/main.py` allows an origin if it matches **either** env var:

| Variable | Purpose |
| :--- | :--- |
| `CORS_ORIGINS` | Comma-separated exact origins, or `*` for all (the local-dev default). |
| `CORS_ORIGIN_REGEX` | Regex matched against the whole `Origin` header. |

Recommended production values:

```
CORS_ORIGINS=https://sih-26137.vercel.app,http://localhost:5500,http://127.0.0.1:5500
CORS_ORIGIN_REGEX=https://sih-26137-[a-z0-9-]+\.vercel\.app
```

Include the localhost entries: without them a dashboard served locally cannot call the
deployed backend.

**Why the regex is needed.** Vercel mints a new hostname for every preview and branch
deploy (`sih-26137-git-<branch>-<team>.vercel.app`), so those origins can't be
enumerated in advance. The regex covers them all.

**How a CORS failure presents.** It does not look like a CORS error from the outside:
the server still returns **HTTP 200**, but with no `access-control-allow-origin` header
the browser discards the response. The dashboard shows a failed request and the API
looks down even though it is healthy. If the API answers `curl` but not the browser,
check this first — confirm with:

```bash
curl -s -i -X OPTIONS https://sih26137.onrender.com/api/network/generate \
  -H "Origin: <the origin you're loading the dashboard from>" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control-allow-origin
```

No output means that origin is blocked.

An invalid `CORS_ORIGIN_REGEX` is logged and ignored rather than crashing the app or
widening access; the exact-origin list still applies. Covered by
`test_cors_config.py`, which also checks the pattern can't be prefix-spoofed by a
lookalike domain (matching uses `fullmatch`).

## Real-city map data

The "Delhi Urban Corridor" demo serves a road network that was downloaded from
OpenStreetMap ahead of time and committed to `data/networks/delhi_central.json`
(481 nodes, 1177 directed edges, 191 of them genuinely one-way).

**Why it is cached rather than fetched live.** Downloading at request time was
measured against this deployed backend at 44s, 169s, and outright failure (502)
within the same hour. OpenStreetMap's public endpoints rate-limit by IP, and
Render's free tier shares an IP with other tenants — so the demo could be
blocked by traffic that isn't ours. The cache loads in ~20 ms and cannot fail
that way.

The roads are real, including one-way restrictions. Only the download moved
offline: congestion is still randomised per run, so traffic conditions are not
frozen along with the geometry.

### Refreshing or adding a network

```bash
python scripts/build_osm_cache.py --name delhi_central
python scripts/build_osm_cache.py --name mumbai_south --label "South Mumbai"     --south 18.90 --north 18.95 --west 72.80 --east 72.85
```

Run it from a machine OpenStreetMap is willing to talk to (a laptop is fine;
the deployed host often is not), then commit the JSON. The script talks to the
Overpass API directly, so it needs no geospatial dependencies.

Map data is © OpenStreetMap contributors, licensed ODbL. The attribution is
stored in the cache file and shown on the dashboard maps; keep it there.

### Running a local Overpass instance (for live demos)

The cache above covers the networks we chose in advance. The *draw your own
boundary* feature cannot be cached, because the whole point is that the judge
picks the box — and that path still goes to the public Overpass mirrors, with
the 44s / 169s / 502 behaviour described above. Running Overpass locally removes
that risk: the same query comes back in well under a second, and no other
tenant's traffic can throttle it.

**This is for the demo machine, not for Render.** The free tier has neither the
RAM nor the disk for it, and the deployed backend should keep using the public
mirrors. The local instance is something you run on the laptop driving the demo.

#### Hardware requirements

| Resource | Needed | Notes |
|---|---|---|
| RAM | 8 GB minimum, 16 GB comfortable | The import is the peak. If it gets OOM-killed, lower `OVERPASS_RULES_LOAD` in the compose file. |
| Disk | 20–30 GB free for India-wide | A city extract needs a couple of GB. The built database is several times the size of the `.osm.pbf` it came from. |
| Time | Minutes to 2+ hours | Entirely dependent on extract size — see the table below. |

The time row is the one that catches people out. **Do the import the day before
the event, not the morning of it.** Once built it lives in a Docker volume and
subsequent starts are instant.

#### Choosing an extract

Pick the smallest extract that covers everywhere a judge might plausibly draw.

| Extract | Size | Import | Use when |
|---|---|---|---|
| City / district | ~50–200 MB | Minutes | The case study is locked to one city and the demo stays there. |
| Single state / zone | ~200–600 MB | ~15–40 min | Recommended default. `central-zone` (335 MB) covers Bareilly, which is what the control room opens on. |
| India-wide | ~1.3 GB | 2 h+ | Only if the demo genuinely roams the country. |

Browse extracts at [download.geofabrik.de/asia/india.html](https://download.geofabrik.de/asia/india.html).

**Geofabrik's India zones are administrative, not compass directions.** Bareilly
is in northern India but Uttar Pradesh belongs to `central-zone`; Delhi is in
`northern-zone`; Bengaluru and Hyderabad are in `southern-zone`. Picking by the
name alone is how you end up importing 335 MB that does not contain the city you
are demoing — and because a box outside the extract returns zero elements and
falls back to the public API silently, nothing will tell you. If in doubt, test
the coordinates against the zone's `.poly` file before starting the import, and
confirm afterwards with `scripts/check_overpass.py`.

The five networks in `data/networks/` span five states, so covering all of them
from one extract means either India-wide or merging several zone extracts with
`osmium merge` beforehand.

#### Runbook

```bash
# 1. Choose the extract (in .env)
OSM_EXTRACT_URL=https://download.geofabrik.de/asia/india/northern-zone-latest.osm.pbf

# 2. Start it. First run downloads and imports; this is the slow part.
docker compose -f docker-compose.overpass.yml up -d

# 3. Watch the import. Config mistakes show up here within the first minute.
docker compose -f docker-compose.overpass.yml logs -f

# 4. Once it is serving, point the app at it (in .env)
OVERPASS_URL=http://localhost:12345/api/interpreter

# 5. Confirm — this is the step that matters
python scripts/check_overpass.py --compare
```

Do a throwaway run with a small city extract first. A typo in the compose file
costs you a minute that way and two hours the other way.

#### How it behaves when it isn't running

| Variable | Default | Effect |
|---|---|---|
| `OVERPASS_URL` | unset | Unset means the public mirrors, exactly as before. Setting it makes the local instance first choice. |
| `OVERPASS_PUBLIC_FALLBACK` | on | A local instance that fails falls through to the public mirrors, so forgetting Docker costs latency rather than the feature. Set to `off` in rehearsal to make misconfiguration loud. |
| `OVERPASS_LOCAL_TIMEOUT` | `20` | Seconds before giving up on a wedged local instance. A healthy one answers in under a second. |

Two details worth knowing. An **empty** response from the local instance is
treated as a failure and falls through to the public mirrors, not as "no roads
here" — because a box outside the imported region returns exactly the same empty
result as genuine farmland, and reporting the first as the second would be wrong
in a way nobody would catch on stage. And the fallback is silent by design,
which is why `scripts/check_overpass.py` exists: run it before the demo, or you
will never notice you're back on the public API until it's slow in front of
judges.

#### If every query comes back "Permission denied"

Symptom: the container is up, `docker logs` shows nginx returning `200`, but any
real query returns XML containing
`runtime error: open64: 13 Permission denied /db/db//osm3s_osm_base`.

Cause: the image creates `/db` as `0700` owned by `overpass`, but serves queries
through `fcgiwrap` running as the `nginx` user, which then cannot traverse into
`/db` to reach the dispatcher socket. The compose file fixes this with a
`post_start` hook that runs `chmod o+x /db`, which grants traversal without
granting read access to the database.

This one is worth knowing because it looks healthy from the outside: the port is
open, nginx logs success, and only the response body says otherwise — which the
silent public-API fallback then hides completely.

### Load order at demo time

1. `POST /api/network/from_cache` — the committed network, ~20 ms
2. `POST /api/network/from_osm` — a live download; sub-second against a local
   Overpass instance, seconds-to-minutes against the public mirrors
3. the synthetic city, with an on-screen notice, if neither is available

## Frontend: Vercel (or Netlify)

`frontend/dashboard.html` is a static file with a configurable **API Base URL** input
field built into the page itself (see the `apiBase` field in the dashboard) — so no
code changes or rebuilds are needed to point it at the deployed Render backend. Deploy
the `frontend/` directory as-is; there is no build step.

- **Vercel:** import the repo, set the project **root directory to `frontend/`**.
  [`frontend/vercel.json`](frontend/vercel.json) rewrites `/` to `/dashboard.html` so the
  dashboard loads at the site root instead of requiring `/dashboard.html` in the URL.
- **Netlify:** import the repo, set **base directory to `frontend/`**, publish directory
  `.`. [`frontend/netlify.toml`](frontend/netlify.toml) provides the equivalent redirect.

After both are deployed, open the frontend URL and set the **API Base URL** field to the
Render backend URL from the previous section.

## Summary — what to tell anyone who suggests "just deploy it all to Vercel"

Point them at this file. The short version: Vercel is great for the static dashboard,
wrong for a long-running, CPU-bound, GDAL-dependent Python backend. Backend → Render,
frontend → Vercel/Netlify, connected via the dashboard's existing API Base URL field.
