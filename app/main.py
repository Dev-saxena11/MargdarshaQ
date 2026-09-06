"""
main.py
--------
FastAPI application entrypoint for SIH26137: Quantum-Inspired Intelligent
Traffic Route Optimization.

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="SIH26137 - Quantum-Inspired Traffic Route Optimization",
    description=(
        "Quantum-inspired metaheuristic (QPSO) framework for solving "
        "large-scale Vehicle Routing Problems (VRP) under traffic congestion, "
        "benchmarked against classical metaheuristics (GA, SA, standard PSO) "
        "and exact/greedy baselines."
    ),
    version="1.0.0",
)

# Permissive CORS for local development / hackathon demo purposes.
# Tighten this (specific origins) before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "project": "SIH26137",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
