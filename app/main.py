"""
main.py
--------
FastAPI application entrypoint for QuantaRoute ENTERPRISE: Quantum-Inspired Intelligent
Traffic Route Optimization.

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="QuantaRoute ENTERPRISE - Quantum-Inspired Traffic Route Optimization",
    description=(
        "Quantum-inspired metaheuristic (QPSO) framework for solving "
        "large-scale Vehicle Routing Problems (VRP) under traffic congestion, "
        "benchmarked against classical metaheuristics (GA, SA, standard PSO) "
        "and exact/greedy baselines."
    ),
    version="1.0.0",
)

# CORS origins are configurable via the CORS_ORIGINS env var (comma-separated
# list of allowed origins, e.g. "https://quantaroute.app"). Defaults to
# "*" (allow all) for local development convenience — set
# CORS_ORIGINS explicitly once deployed so the API isn't wide open to any
# origin. See DEPLOYMENT.md.
_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
_allow_origins = (
    ["*"]
    if _cors_origins_env.strip() == "*"
    else [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "project": "QuantaRoute ENTERPRISE",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
