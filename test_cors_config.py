"""
test_cors_config.py
-------------------
Tests for CORS origin configuration (CORS_ORIGINS / CORS_ORIGIN_REGEX).

Why these matter: a CORS misconfiguration does not look like a CORS problem
from the browser. The server still returns HTTP 200, but without an
`access-control-allow-origin` header the browser discards the response — which
presents as "the API is down" even though the API is fine.

Runs fully offline; no deployed backend needed.

Run with:  python test_cors_config.py
"""

import importlib
import os
import sys

sys.path.insert(0, ".")

PROD = "https://sih-26137.vercel.app"
PREVIEW = "https://sih-26137-git-main-rudra-singhs-projects.vercel.app"
LOCAL = "http://localhost:5500"
LOOKALIKE = "https://sih-26137.evil.com"
VERCEL_REGEX = r"https://sih-26137-[a-z0-9-]+\.vercel\.app"

failures = []


def allowed_origin(origin: str, **env) -> str | None:
    """Reimport the app under the given env and return the CORS header, if any."""
    for key in ("CORS_ORIGINS", "CORS_ORIGIN_REGEX"):
        os.environ.pop(key, None)
    os.environ.update(env)
    for mod in [m for m in list(sys.modules) if m.startswith("app.")]:
        del sys.modules[mod]

    from fastapi.testclient import TestClient
    app = importlib.import_module("app.main").app
    resp = TestClient(app).options(
        "/api/network/generate",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    return resp.headers.get("access-control-allow-origin")


def check(label, origin, env, should_allow):
    got = allowed_origin(origin, **env)
    ok = bool(got) == should_allow
    print(f"{'PASS' if ok else 'FAIL'}  {label:32s} "
          f"expected={'allow' if should_allow else 'block':5s} got={got or 'blocked'}")
    if not ok:
        failures.append(label)


print("=" * 78)
print("1. Exact origin list (CORS_ORIGINS)")
print("=" * 78)
check("production origin allowed", PROD, {"CORS_ORIGINS": PROD}, True)
check("unlisted preview blocked", PREVIEW, {"CORS_ORIGINS": PROD}, False)
check("listed localhost allowed", LOCAL, {"CORS_ORIGINS": f"{PROD},{LOCAL}"}, True)
check("wildcard allows anything", PREVIEW, {"CORS_ORIGINS": "*"}, True)

print()
print("=" * 78)
print("2. Regex origins (CORS_ORIGIN_REGEX) — for per-deploy preview hostnames")
print("=" * 78)
_regex_env = {"CORS_ORIGINS": PROD, "CORS_ORIGIN_REGEX": VERCEL_REGEX}
check("preview matched by regex", PREVIEW, _regex_env, True)
check("exact list still honoured", PROD, _regex_env, True)
check("listed localhost still works", LOCAL,
      {"CORS_ORIGINS": f"{PROD},{LOCAL}", "CORS_ORIGIN_REGEX": VERCEL_REGEX}, True)

print()
print("=" * 78)
print("3. The regex must not over-match")
print("=" * 78)
# Starlette matches with fullmatch(), so a lookalike domain that merely shares a
# prefix must not slip through. If this ever fails, the pattern is exploitable.
check("lookalike domain blocked", LOOKALIKE, _regex_env, False)
check("bare vercel.app blocked", "https://vercel.app", _regex_env, False)
check("http scheme blocked", PREVIEW.replace("https://", "http://"), _regex_env, False)

print()
print("=" * 78)
print("4. An invalid regex degrades safely")
print("=" * 78)
_bad = {"CORS_ORIGINS": PROD, "CORS_ORIGIN_REGEX": "([unclosed"}
check("bad pattern does not widen access", PREVIEW, _bad, False)
check("bad pattern keeps exact list", PROD, _bad, True)

print()
print("=" * 78)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
