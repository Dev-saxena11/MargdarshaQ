"""
main.py
--------
FastAPI application entrypoint for MargdarshaQ ENTERPRISE: Quantum-Inspired Intelligent
Traffic Route Optimization.

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import logging
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router

app = FastAPI(
    title="MargdarshaQ ENTERPRISE - Quantum-Inspired Traffic Route Optimization",
    description=(
        "Quantum-inspired metaheuristic (QPSO) framework for solving "
        "large-scale Vehicle Routing Problems (VRP) under traffic congestion, "
        "benchmarked against classical metaheuristics (GA, SA, standard PSO) "
        "and exact/greedy baselines."
    ),
    version="1.0.0",
)

# CORS is configured from two env vars, and an origin is allowed if it matches
# EITHER:
#
#   CORS_ORIGINS       comma-separated exact origins, or "*" for all.
#                      Defaults to "*" for local development convenience — set
#                      it explicitly once deployed so the API isn't wide open.
#   CORS_ORIGIN_REGEX  a regex matched against the whole Origin header.
#
# The regex exists because hosts like Vercel mint a NEW hostname for every
# preview/branch deploy, so those origins cannot be enumerated ahead of time.
# Without it, a dashboard opened from a preview URL gets HTTP 200 responses
# with no access-control-allow-origin header — the browser then discards them,
# which looks exactly like the API being down.
#
# See DEPLOYMENT.md.
_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
_allow_origins = (
    ["*"]
    if _cors_origins_env.strip() == "*"
    else [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
)

_cors_origin_regex = (os.getenv("CORS_ORIGIN_REGEX") or "").strip() or None
if _cors_origin_regex is not None:
    try:
        re.compile(_cors_origin_regex)
    except re.error as exc:
        # A bad pattern must not take the API down, and must not silently widen
        # access either — drop it and carry on with the exact-origin list.
        logging.getLogger(__name__).warning(
            "Ignoring invalid CORS_ORIGIN_REGEX %r: %s", _cors_origin_regex, exc
        )
        _cors_origin_regex = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=_cors_origin_regex,
    # Credentials cannot be combined with a wildcard origin per the CORS spec,
    # so they are only enabled once the allowed origins are actually restricted.
    allow_credentials=_allow_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errs = exc.errors()
    if errs:
        msg = errs[0]["msg"].replace("Value error, ", "")
        return JSONResponse(status_code=422, content={"detail": msg})
    return JSONResponse(status_code=422, content={"detail": "Invalid parameters"})

@app.get("/")
def root():
    return {
        "project": "MargdarshaQ ENTERPRISE",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
